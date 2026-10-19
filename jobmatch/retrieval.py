"""Local E5 embeddings + Chroma + BM25 with reciprocal-rank fusion."""
import hashlib
import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from jobmatch.config import EMBEDDING_MODEL, INDEX_PATH
from jobmatch.data import load_ads, make_chunks, ads_fingerprint


def embedding_model(name: str = EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w+#.]+", text.casefold())


def write_manifest(target: Path, manifest: dict):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=target, prefix=".manifest-", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(manifest, handle, indent=2)
    try:
        temporary.replace(target / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)


def build_index(source: Path, target: Path = INDEX_PATH, size: int = 320, overlap: int = 60,
                publish: bool = True) -> dict:
    import chromadb
    model = embedding_model()
    ads = load_ads(source)
    chunks = make_chunks(ads, model.tokenizer, size, overlap)
    target.mkdir(parents=True, exist_ok=True)
    old_chunks, old_collection = {}, None
    try:
        previous = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if previous["embedding_model"] == EMBEDDING_MODEL:
            old_generation = target / previous["generation"]
            old_chunks = {c["chunk_id"]: c["text"] for c in json.loads((old_generation / "chunks.json").read_text(encoding="utf-8"))}
            old_client = chromadb.PersistentClient(path=str(old_generation / "chroma"))
            old_collection = old_client.get_collection("jobmatch")
    except (OSError, ValueError, KeyError, chromadb.errors.ChromaError):
        old_chunks = {}
    # A new generation keeps a failed rebuild from corrupting the working index.
    generation = target / f"build-{uuid.uuid4().hex[:12]}"
    generation.mkdir()
    client = chromadb.PersistentClient(path=str(generation / "chroma"))
    collection = client.create_collection("jobmatch", metadata={"hnsw:space": "cosine"})
    for start in range(0, len(chunks), 64):
        batch = chunks[start:start + 64]
        vectors = {}
        reusable = [c["chunk_id"] for c in batch if old_collection is not None and old_chunks.get(c["chunk_id"]) == c["text"]]
        if reusable:
            stored = old_collection.get(ids=reusable, include=["embeddings"])
            vectors.update((cid, vector.tolist()) for cid, vector in zip(stored["ids"], stored["embeddings"]))
        changed = [c for c in batch if c["chunk_id"] not in vectors]
        if changed:
            encoded = model.encode(["passage: " + c["text"] for c in changed], normalize_embeddings=True).tolist()
            vectors.update((c["chunk_id"], vector) for c, vector in zip(changed, encoded))
        collection.add(ids=[c["chunk_id"] for c in batch], documents=[c["text"] for c in batch],
                       metadatas=[c["metadata"] for c in batch], embeddings=[vectors[c["chunk_id"]] for c in batch])
        print(f"Indexed {min(start + 64, len(chunks))}/{len(chunks)} chunks", flush=True)
    (generation / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (generation / "ads.json").write_text(json.dumps(ads, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "generation": generation.name, "embedding_model": EMBEDDING_MODEL,
        "chunk_size": size, "overlap": overlap, "ad_count": len(ads), "chunk_count": len(chunks),
        "source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "ads_fingerprint": ads_fingerprint(ads),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    if publish:
        write_manifest(target, manifest)
    return manifest


class Retriever:
    def __init__(self, path: Path = INDEX_PATH):
        import chromadb
        self.manifest = json.loads((path / "manifest.json").read_text())
        generation = path / self.manifest["generation"]
        self.chunks = json.loads((generation / "chunks.json").read_text())
        self.ads = json.loads((generation / "ads.json").read_text())
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.client = chromadb.PersistentClient(path=str(generation / "chroma"))
        self.collection = self.client.get_collection("jobmatch")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = embedding_model(self.manifest["embedding_model"])
        return self._model

    def filter_ads(self, filters: dict | None = None) -> list[dict]:
        filters = filters or {}
        return [a for a in self.ads if all(a.get(k) == v for k, v in filters.items() if v)]

    def retrieve(self, query: str, k: int = 6, filters: dict | None = None,
                 mode: str = "hybrid", min_similarity: float = 0.72) -> list[dict]:
        from rank_bm25 import BM25L
        if mode not in {"hybrid", "semantic", "keyword"}:
            raise ValueError("Unknown retrieval mode")
        if not query.strip() or k < 1:
            return []
        filters = {key: value for key, value in (filters or {}).items() if value}
        if set(filters) - {"location", "role", "field", "employer", "opportunity_type"}:
            raise ValueError("Unsupported metadata filter")
        candidates = [c for c in self.chunks if all(c["metadata"].get(key) == val for key, val in filters.items())]
        if not candidates:
            return []
        pool = min(len(candidates), max(k * 8, 40))
        semantic, keyword, similarities = [], [], {}
        if mode != "keyword":
            import numpy as np
            model = self.model
            query_tokens = model.tokenizer.encode(query, add_special_tokens=False, verbose=False)
            query_parts = [query] if len(query_tokens) <= 400 else [
                model.tokenizer.decode(query_tokens[start:start + 400], skip_special_tokens=True)
                for start in range(0, len(query_tokens), 400)
            ]
            # Long CVs are represented by all segments, rather than silently
            # discarding everything past the embedding model's context window.
            vectors = model.encode(["query: " + part for part in query_parts], normalize_embeddings=True)
            mean = np.asarray(vectors).mean(axis=0)
            vector = [ (mean / max(float(np.linalg.norm(mean)), 1e-12)).tolist() ]
            where = None
            if len(filters) == 1:
                where = {key: {"$eq": value} for key, value in filters.items()}
            elif filters:
                where = {"$and": [{key: {"$eq": val}} for key, val in filters.items()]}
            kwargs = {"where": where} if where else {}
            result = self.collection.query(query_embeddings=vector, n_results=pool,
                                           include=["distances"], **kwargs)
            for cid, distance in zip(result["ids"][0], result["distances"][0]):
                similarity = 1 - distance
                similarities[cid] = similarity
                if similarity >= min_similarity:
                    semantic.append(cid)
        if mode != "semantic":
            tokens = [tokenize(c["text"]) for c in candidates]
            # BM25L keeps IDF positive in small filtered subsets; Okapi can
            # assign zero/negative scores when a skill occurs in most ads.
            bm25 = BM25L(tokens)
            scores = bm25.get_scores(tokenize(query))
            ranked = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]["chunk_id"]))
            keyword = [candidates[i]["chunk_id"] for i in ranked[:pool] if scores[i] > 0]
        scores = Counter()
        for ranking in (semantic, keyword):
            for rank, cid in enumerate(ranking, 1):
                scores[cid] += 1 / (60 + rank)
        # Limit repeated chunks from a single ad so answers cover multiple jobs.
        output, per_ad = [], Counter()
        for cid in sorted(scores, key=lambda c: (-scores[c], c)):
            chunk = self.by_id[cid]
            ad_id = chunk["metadata"]["ad_id"]
            if per_ad[ad_id] >= 2:
                continue
            output.append({**chunk, "score": scores[cid], "similarity": similarities.get(cid)})
            per_ad[ad_id] += 1
            if len(output) >= k:
                break
        return output
