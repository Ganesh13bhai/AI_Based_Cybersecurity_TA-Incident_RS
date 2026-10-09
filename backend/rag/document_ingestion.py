
import json
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENT_PATH = PROJECT_ROOT / "data" / "rag" / "documents.jsonl"


class DocumentIngestion:
    """Store cybersecurity knowledge documents without exact duplicates."""

    def __init__(self, document_path: Path = DOCUMENT_PATH):
        self.document_path = Path(document_path)
        self.document_path.parent.mkdir(parents=True, exist_ok=True)

    def list_documents(self) -> list[dict[str, Any]]:
        if not self.document_path.exists():
            return []

        documents = []

        with self.document_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in document store at line {line_number}."
                    ) from exc

                if isinstance(item, dict) and item.get("text"):
                    documents.append(item)

        return documents

    def ingest(
        self,
        text: str,
        source: str,
        title: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Document text cannot be empty.")

        if not isinstance(source, str) or not source.strip():
            raise ValueError("Document source is required.")

        if not isinstance(title, str) or not title.strip():
            raise ValueError("Document title is required.")

        normalized_text = text.strip()
        normalized_source = source.strip()
        normalized_title = title.strip()

        # Return the existing record for an exact source/title/text match.
        for existing in self.list_documents():
            if (
                existing.get("source") == normalized_source
                and existing.get("title") == normalized_title
                and existing.get("text") == normalized_text
            ):
                return {
                    "document_id": existing["document_id"],
                    "title": existing["title"],
                    "source": existing["source"],
                    "metadata": existing.get("metadata", {}),
                    "ingested_at": existing.get("ingested_at"),
                    "duplicate": True,
                }

        document = {
            "document_id": f"DOC-{uuid.uuid4().hex[:12].upper()}",
            "title": normalized_title,
            "source": normalized_source,
            "text": normalized_text,
            "metadata": metadata or {},
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

        # Validate JSON serialization before writing.
        line = json.dumps(document, ensure_ascii=False, allow_nan=False)

        with self.document_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

        return {
            "document_id": document["document_id"],
            "title": document["title"],
            "source": document["source"],
            "metadata": document["metadata"],
            "ingested_at": document["ingested_at"],
            "duplicate": False,
        }
