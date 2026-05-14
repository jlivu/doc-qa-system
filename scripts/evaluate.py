#!/usr/bin/env python3
"""Evaluation script for the Document Q&A system.

Runs the golden dataset through the live API and reports:
- Retrieval recall@5
- Answer similarity (cosine)
- Not-found accuracy

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --api-url http://localhost:8001
    python scripts/evaluate.py --output-dir eval/results/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def embed_text(text: str, api_url: str) -> list[float] | None:
    """Embed text using the same API embedding model via Ollama."""
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
            api_key="ollama",
        )
        response = client.embeddings.create(
            input=text,
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        )
        return response.data[0].embedding
    except Exception:
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def run_query(api_url: str, question: str, filename: str | None) -> dict:
    payload: dict = {"question": question}
    if filename:
        payload["filters"] = {"filename": filename}
    resp = requests.post(f"{api_url}/query", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Document Q&A system")
    parser.add_argument("--api-url", default="http://localhost:8001")
    parser.add_argument("--output-dir", default="eval/results")
    parser.add_argument("--dataset", default="eval/golden_dataset.json")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} questions from {args.dataset}")

    # Pre-flight: verify all required documents are ingested
    required_filenames = {
        item["document_filename"] for item in dataset
        if item.get("document_filename")
    }
    try:
        resp = requests.get(f"{args.api_url}/documents", timeout=30)
        resp.raise_for_status()
        ingested_filenames = {d["filename"] for d in resp.json().get("documents", [])}
    except Exception as exc:
        print(f"ERROR: Could not reach the API at {args.api_url}/documents: {exc}")
        sys.exit(2)

    missing = required_filenames - ingested_filenames
    if missing:
        print("ERROR: The following documents from the golden dataset are not ingested:")
        for name in sorted(missing):
            print(f"  - {name}")
        print("\nIngest the missing documents before running evaluation.")
        sys.exit(2)
    print(f"Pre-flight OK: all {len(required_filenames)} required documents are ingested.")

    results = []
    for item in dataset:
        doc_filter = item.get("document_filename") if item["expected_found"] else None
        start = time.time()
        try:
            result = run_query(args.api_url, item["question"], doc_filter)
            latency = time.time() - start

            sources = result.get("sources") or []
            found = result.get("found", False)
            answer_text = result.get("answer", "")
            retrieved_pages = [s["page"] for s in sources]
            top_score = sources[0]["score"] if sources else None

            results.append({
                **item,
                "actual_answer": answer_text,
                "actual_found": found,
                "actual_sources": sources,
                "retrieved_pages": retrieved_pages,
                "top_score": top_score,
                "confidence": result.get("confidence", "low"),
                "latency": round(latency, 2),
                "error": None,
            })
            print(f"  [{item['id']}] found={found} top_score={top_score} pages={retrieved_pages[:5]} ({latency:.1f}s)")
        except Exception as exc:
            latency = time.time() - start
            results.append({
                **item,
                "actual_answer": "",
                "actual_found": False,
                "actual_sources": [],
                "retrieved_pages": [],
                "top_score": None,
                "confidence": "low",
                "latency": round(latency, 2),
                "error": str(exc),
            })
            print(f"  [{item['id']}] ERROR: {exc} ({latency:.1f}s)")

    # Compute metrics
    answerable = [r for r in results if r["expected_found"]]
    unanswerable = [r for r in results if not r["expected_found"]]

    # Retrieval recall@5
    recall_hits = 0
    for r in answerable:
        top5_pages = set(r["retrieved_pages"][:5])
        expected = set(r["expected_source_pages"])
        if top5_pages & expected:  # any overlap
            recall_hits += 1
    recall = recall_hits / len(answerable) if answerable else 0.0

    # Answer similarity
    similarities = []
    for r in answerable:
        if r["reference_answer"] and r["actual_answer"]:
            emb_ref = embed_text(r["reference_answer"], args.api_url)
            emb_act = embed_text(r["actual_answer"], args.api_url)
            if emb_ref and emb_act:
                similarities.append(cosine_similarity(emb_ref, emb_act))
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    # Not-found accuracy — unanswerable questions should have found==False
    not_found_correct = sum(
        1 for r in unanswerable
        if r["expected_found"] is False and r["actual_found"] is False
    )
    not_found_accuracy = not_found_correct / len(unanswerable) if unanswerable else 0.0

    # Targets
    recall_pass = recall >= 0.80
    similarity_pass = avg_similarity >= 0.70
    not_found_pass = not_found_accuracy >= 1.0
    all_pass = recall_pass and similarity_pass and not_found_pass

    # Print summary
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nEvaluation results — {now}")
    print("=" * 42)
    print(f"Questions evaluated:  {len(results)}")
    print(f"Answerable:           {len(answerable)}")
    print(f"Unanswerable:          {len(unanswerable)}")
    print()
    print(f"Retrieval recall@5:   {recall:.2f}  {'✓' if recall_pass else '✗'}  (target: >= 0.80)")
    print(f"Answer similarity:    {avg_similarity:.2f}  {'✓' if similarity_pass else '✗'}  (target: >= 0.70)")
    print(f"Not-found accuracy:   {not_found_accuracy:.2f}  {'✓' if not_found_pass else '✗'}  (target: 1.00)")
    print()

    # Write results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    json_path = output_dir / f"{timestamp}.json"
    md_path = output_dir / "latest.md"

    output_data = {
        "timestamp": timestamp,
        "metrics": {
            "retrieval_recall_at_5": round(recall, 4),
            "answer_similarity": round(avg_similarity, 4),
            "not_found_accuracy": round(not_found_accuracy, 4),
        },
        "targets_met": all_pass,
        "results": results,
    }

    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    with open(md_path, "w") as f:
        f.write(f"# Evaluation Results — {timestamp}\n\n")
        f.write("| Metric | Value | Target | Pass |\n")
        f.write("|--------|-------|--------|------|\n")
        f.write(f"| Retrieval recall@5 | {recall:.2f} | >= 0.80 | {'✓' if recall_pass else '✗'} |\n")
        f.write(f"| Answer similarity | {avg_similarity:.2f} | >= 0.70 | {'✓' if similarity_pass else '✗'} |\n")
        f.write(f"| Not-found accuracy | {not_found_accuracy:.2f} | 1.00 | {'✓' if not_found_pass else '✗'} |\n\n")
        f.write("## Per-question results\n\n")
        for r in results:
            status = "✓" if r["error"] is None else "✗"
            f.write(f"### [{r['id']}] {r['question']}\n\n")
            f.write(f"- **Expected found:** {r['expected_found']}\n")
            f.write(f"- **Actual found:** {r['actual_found']}\n")
            f.write(f"- **Top score:** {r['top_score']}\n")
            f.write(f"- **Retrieved pages:** {r['retrieved_pages'][:5]}\n")
            f.write(f"- **Expected pages:** {r['expected_source_pages']}\n")
            f.write(f"- **Confidence:** {r['confidence']}\n")
            f.write(f"- **Latency:** {r['latency']}s\n")
            if r["reference_answer"]:
                f.write(f"- **Reference:** {r['reference_answer'][:200]}\n")
            f.write(f"- **Generated:** {r['actual_answer'][:200]}\n\n")

    if all_pass:
        print(f"All targets met. Results saved to {json_path}")
    else:
        print(f"Some targets missed. Results saved to {json_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
