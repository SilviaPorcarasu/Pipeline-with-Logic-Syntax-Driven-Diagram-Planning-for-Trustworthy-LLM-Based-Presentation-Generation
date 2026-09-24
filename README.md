# Retrieval data

Prepare a JSONL retrieval corpus from your source PDF:

```bash
python scripts/build_chunks_from_pdf.py \
  --pdf-path /path/to/source.pdf \
  --out-path data/book_chunks.jsonl
```

Run this command from the repository root. Each record contains `chunk_id`,
`source`, and `text`. Set `CHUNKS_PATH` to use a different corpus.

Source textbooks, extracted corpora, generated training datasets, embedding
caches, and trained adapters are not included. Dataset construction and LoRA
training scripts are provided in `scripts/`.
