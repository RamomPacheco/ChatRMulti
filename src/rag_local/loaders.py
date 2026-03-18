from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from langchain_core.documents import Document


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def load_documents(docs_dir: Path) -> List[Document]:
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Pasta de documentos não existe: {docs_dir}")

    documents: List[Document] = []

    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    for path in iter_files(docs_dir):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            documents.extend(loader.load())
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
            documents.extend(loader.load())
        else:
            # ignora outros tipos por padrão
            continue

    return documents
