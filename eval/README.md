# Evaluation

## Golden Dataset

`golden_dataset.json` contains 15 question/answer pairs drawn from the ingested Vanuatu government documents. 13 are answerable and 2 are unanswerable.

## Running the Evaluation

```bash
# Ensure the API is running and documents are ingested
python scripts/evaluate.py

# Custom API URL
python scripts/evaluate.py --api-url http://localhost:8001

# Custom output directory
python scripts/evaluate.py --output-dir eval/results/
```

## Metrics

- **Retrieval recall@5** — proportion of answerable questions where at least one expected page appears in top-5 sources. Target: >= 0.80.
- **Answer similarity** — cosine similarity between generated and reference answers. Target: >= 0.70.
- **Not-found accuracy** — proportion of unanswerable questions correctly returned as `found: false`. Target: 1.00.

## Results

Results are written to `eval/results/` as timestamped JSON files and a `latest.md` Markdown summary.
