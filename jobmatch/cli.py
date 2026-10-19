import argparse
import json
from pathlib import Path

from jobmatch.config import ADS_PATH, INDEX_PATH, OPENAI_MODEL


def main():
    parser = argparse.ArgumentParser(description="JobMatch command-line tools")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Download JobTech ads")
    ingest.add_argument("--limit", type=int, default=3000)
    ingest.add_argument("--scope", choices=["all", "it", "data-science", "tech"], default="tech")
    ingest.add_argument("--output", type=Path, default=ADS_PATH)
    build = commands.add_parser("build", help="Clean, chunk, embed and index a JSONL dataset")
    build.add_argument("--input", type=Path, default=ADS_PATH)
    build.add_argument("--index", type=Path, default=INDEX_PATH)
    build.add_argument("--chunk-size", type=int, default=320)
    build.add_argument("--overlap", type=int, default=60)
    search = commands.add_parser("search", help="Retrieve excerpts without calling OpenAI")
    search.add_argument("query")
    search.add_argument("--location")
    search.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid")
    search.add_argument("--k", type=int, default=6)
    search.add_argument("--index", type=Path, default=INDEX_PATH)
    ask = commands.add_parser("ask", help="Stream an English grounded answer")
    ask.add_argument("query")
    ask.add_argument("--model", default=OPENAI_MODEL)
    ask.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    ask.add_argument("--location")
    ask.add_argument("--index", type=Path, default=INDEX_PATH)
    args = parser.parse_args()
    if args.command == "ingest":
        from jobmatch.ingest import download_ads
        if args.limit < 1:
            parser.error("--limit must be positive")
        count = download_ads(args.output, args.limit, args.scope == "it", args.scope == "data-science",
                             args.scope == "tech")
        print(f"Saved {count} ads to {args.output}")
    elif args.command == "build":
        from jobmatch.retrieval import build_index
        print(json.dumps(build_index(args.input, args.index, args.chunk_size, args.overlap), indent=2))
    else:
        from jobmatch.retrieval import Retriever
        from jobmatch.conversation import thesis_opportunity_query
        retriever = Retriever(args.index)
        filters = {"location": args.location}
        if thesis_opportunity_query(args.query):
            filters["opportunity_type"] = "thesis"
        chunks = retriever.retrieve(args.query, k=getattr(args, "k", 6), filters=filters,
                                    mode=getattr(args, "mode", "hybrid"))
        if args.command == "search":
            print(json.dumps(chunks, ensure_ascii=False, indent=2))
        else:
            from jobmatch.generation import stream_answer
            for delta in stream_answer(args.query, chunks, model=args.model, backend=args.backend):
                print(delta, end="", flush=True)
            print("\n\nSources:")
            for i, chunk in enumerate(chunks, 1):
                print(f"[{i}] {chunk['metadata']['headline']} — {chunk['metadata']['url']}")


if __name__ == "__main__":
    main()
