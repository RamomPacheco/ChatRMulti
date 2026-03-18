from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_local.config import Settings


def _stable_chunk_id(doc: Document, idx: int) -> str:
    import hashlib

    src = str(doc.metadata.get("source") or "")
    page = str(doc.metadata.get("page") if doc.metadata.get("page") is not None else "")
    h = hashlib.sha1(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{src}:{page}:{idx}:{h}"


def build_embeddings(settings: Settings):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embed_model)


def build_vectorstore(settings: Settings):
    from langchain_chroma import Chroma

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    embeddings = build_embeddings(settings)
    return Chroma(
        collection_name="rag_local",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def split_documents(settings: Settings, docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def index_documents(settings: Settings, docs: List[Document]) -> Tuple[int, int]:
    """
    Retorna (num_docs_originais, num_chunks).
    """
    vs = build_vectorstore(settings)
    chunks = split_documents(settings, docs)
    ids = [_stable_chunk_id(d, i) for i, d in enumerate(chunks)]
    vs.add_documents(chunks, ids=ids)
    return (len(docs), len(chunks))


def build_retriever(settings: Settings):
    vs = build_vectorstore(settings)
    return vs.as_retriever(search_kwargs={"k": settings.top_k})


def answer(settings: Settings, llm, question: str) -> str:
    # LangChain 1.x moveu "chains" para langchain-classic
    from langchain_classic.chains import RetrievalQA

    retriever = build_retriever(settings)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
    )
    res = chain.invoke({"query": question})
    if isinstance(res, dict) and "result" in res:
        return str(res["result"])
    return str(res)


def answer_with_sources(settings: Settings, llm, question: str):
    # LangChain 1.x moveu "chains" para langchain-classic
    from langchain_classic.chains import RetrievalQA

    retriever = build_retriever(settings)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    res = chain.invoke({"query": question})
    return res
