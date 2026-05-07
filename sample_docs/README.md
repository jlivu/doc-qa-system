# Sample documents

Place one or more publicly available PDF documents here to use for testing
and demonstration.

## Suggested sources for Vanuatu government documents

- **Vanuatu National Budget** — available at https://mof.gov.vu
- **Reserve Bank of Vanuatu Annual Report** — available at https://rbv.gov.vu
- **Vanuatu National Statistics Office reports** — available at https://vnso.gov.vu

Download a PDF, drop it in this folder, and run the ingestion pipeline:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_docs/your_document.pdf"
```

## Why this folder exists in the repo

A reviewer cloning this project should be able to run the full system
immediately. Providing a real document — rather than a synthetic test fixture —
makes the demo authentic and demonstrates the government-document use case
described in the project README.

Note: Do not commit documents that are not publicly available or that contain
sensitive information.
