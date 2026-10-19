"""Reproducible retrieval and model experiments; no invented quality scores."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from jobmatch.config import INDEX_PATH, OPENAI_MODEL
from jobmatch.generation import GenerationStats, stream_answer
from jobmatch.retrieval import Retriever


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_questions(retriever: Retriever, output: Path):
    """Make a draft set tied to real IDs; review before treating it as a benchmark."""
    ads = sorted(retriever.ads, key=lambda a: a["ad_id"])
    if len(ads) < 15:
        raise ValueError("At least 15 unique ads are needed to create the draft question set")
    selected = ads[::max(1, len(ads) // 15)][:15]
    questions = []
    for i, ad in enumerate(selected, 1):
        questions.append({"id": f"specific-{i:02}", "category": "specific", "reviewed": False,
                          "question": f"What requirements are stated for the '{ad['headline']}' role at {ad['employer']}?",
                          "gold_ad_ids": [ad["ad_id"]], "expected_facts": [], "filters": {}})
    locations = sorted({a["location"] for a in ads})
    for i in range(8):
        location = locations[i % len(locations)]
        questions.append({"id": f"filter-{i + 1:02}", "category": "filter", "reviewed": False,
                          "question": f"Show examples of data-related roles in {location} and explain their requirements.",
                          "filters": {"location": location},
                          "gold_ad_ids": [a["ad_id"] for a in ads if a["location"] == location], "expected_facts": []})
    profiles = ["Python, SQL, pandas, statistics; MSc student; a classification project.",
                "SQL, Power BI, Excel; two years of dashboard development.",
                "Python, PyTorch, machine learning, Docker; a computer vision project.",
                "SQL, Spark, Azure, ETL; one year of data engineering."]
    for i, cv in enumerate(profiles, 1):
        questions.append({"id": f"cv-{i:02}", "category": "cv", "reviewed": False,
                          "question": "Which roles fit this CV, and which requirements are not evidenced by it?",
                          "cv": cv, "gold_ad_ids": [], "expected_facts": [], "filters": {}})
    for i, question in enumerate([
        "What is the exact salary of every role when salary is not stated in the supplied ads?",
        "Can you guarantee that I will receive an offer for one of these jobs?",
        "What is the weather forecast for Stockholm tomorrow?",
    ], 1):
        questions.append({"id": f"refusal-{i:02}", "category": "unanswerable", "reviewed": False,
                          "question": question, "gold_ad_ids": [], "expected_facts": ["Should refuse unsupported request"], "filters": {}})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {len(questions)} DRAFT questions at {output}. Review labels and expected facts before the report.")


def evaluate(retriever: Retriever, questions: list[dict], output: Path, k: int,
             generation: bool, backend: str, model: str, limit: int,
             input_rate: float | None, output_rate: float | None):
    retrieval_rows, generation_rows = [], []
    output.mkdir(parents=True, exist_ok=True)
    if generation and backend == "openai" and not __import__("os").getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in .env before running model experiments")
    for q in questions[:limit]:
        query = q["question"] + ("\nCV: " + q["cv"] if q.get("cv") else "")
        rankings = {}
        for mode in ["keyword", "semantic", "hybrid"]:
            chunks = retriever.retrieve(query, k, q.get("filters"), mode)
            rankings[mode] = chunks
            retrieved = {c["metadata"]["ad_id"] for c in chunks}
            gold = set(q.get("gold_ad_ids", []))
            retrieval_rows.append({
                "question_id": q["id"], "category": q["category"], "reviewed": q.get("reviewed", False),
                "mode": mode, "chunk_size": retriever.manifest["chunk_size"], "k_chunks": k,
                "hit_at_k": int(bool(gold & retrieved)) if gold else "",
                "ad_recall_at_k": len(gold & retrieved) / len(gold) if gold else "",
                "retrieved_ad_ids": json.dumps(sorted(retrieved)), "gold_ad_ids": json.dumps(sorted(gold)),
            })
        if generation:
            for grounded in [True, False]:
                stats = GenerationStats()
                chunks = rankings["hybrid"] if grounded else []
                try:
                    answer = "".join(stream_answer(q["question"], chunks, backend=backend, model=model,
                                                  cv=q.get("cv", ""), stats=stats, grounded=grounded))
                    error = ""
                except Exception as exc:
                    answer, error = "", type(exc).__name__
                cost = ""
                if input_rate is not None and output_rate is not None and backend == "openai":
                    cost = (stats.input_tokens * input_rate + stats.output_tokens * output_rate) / 1_000_000
                generation_rows.append({
                    "question_id": q["id"], "category": q["category"], "backend": backend, "model": model,
                    "grounded": grounded, "question": q["question"], "answer": answer, "error": error,
                    **asdict(stats), "estimated_cost_usd": cost,
                    "source_ad_ids": json.dumps([c["metadata"]["ad_id"] for c in chunks]),
                    "source_excerpts": json.dumps(chunks, ensure_ascii=False),
                    "supported_claims_manual": "", "unsupported_claims_manual": "",
                    "correct_refusal_manual": "", "notes_manual": "",
                })
        # Checkpoint each question so an interrupted experiment retains completed work.
        write_csv(output / "retrieval.csv", retrieval_rows)
        write_csv(output / f"generation_{backend}.csv", generation_rows)
        print(f"Evaluated {q['id']}", flush=True)
    run = {"index": retriever.manifest, "k": k, "question_count": min(limit, len(questions)),
           "questions": questions[:limit], "backend": backend if generation else None,
           "model": model if generation else None, "input_price_per_million": input_rate,
           "output_price_per_million": output_rate}
    (output / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["seed", "run"])
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--questions", type=Path, default=Path("evaluation/questions.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/default"))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--generation", action="store_true", help="Run paid OpenAI or local Ollama requests")
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", default=OPENAI_MODEL)
    parser.add_argument("--input-rate", type=float, help="USD per million input tokens, from current pricing")
    parser.add_argument("--output-rate", type=float, help="USD per million output tokens, from current pricing")
    args = parser.parse_args()
    if args.k < 1 or args.limit < 1:
        parser.error("--k and --limit must be positive")
    retriever = Retriever(args.index)
    if args.command == "seed":
        if args.questions.exists():
            parser.error("Question file already exists; choose a new --questions path to preserve reviewed labels")
        seed_questions(retriever, args.questions)
    else:
        questions = json.loads(args.questions.read_text())
        evaluate(retriever, questions, args.output, args.k, args.generation, args.backend,
                 args.model, args.limit, args.input_rate, args.output_rate)


if __name__ == "__main__":
    main()
