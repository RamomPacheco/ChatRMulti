from __future__ import annotations
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_local.config import Settings

logger = logging.getLogger(__name__)

# Pergunta+contexto: input_variables. Instruções do operador (GUI) vão em partial.
_RAG_STRICT_STUFF_HEAD = (
    "Tarefa: responder de forma fechada usando SOMENTE o contexto abaixo (trechos recuperados dos documentos indexados).\n\n"
    "Regras obrigatórias (cumpra todas):\n"
    "1) Não use conhecimento geral, memória ou suposições fora do contexto. Se a informação não estiver dita (ou de forma clara) no contexto, não a afirme.\n"
    "2) Não invente nomes, datas, números, percentagens, citações legais, URLs, normas, ou detalhes de procedimento que não apareçam no contexto.\n"
    "3) Não atribua ao contexto citações ou conclusões que nela não existam. Em caso de dúvida, sinalize a limitação (ver regra 4).\n"
    "4) Se o contexto for vazio, for insuficiente para responder, ou a pergunta for ambígua sem pistas no contexto, responda com uma única frase, exatamente: "
    '"Não consigo responder com base nos documentos fornecidos."\n'
    "5) Títulos, fontes, páginas: só se constarem claramente no contexto; caso contrário não invente formatação de referência.\n"
    "6) Língua: português, direto e tão curto quanto a pergunta permitir (evite abrir com fórmulas vazias salvo se a pergunta pedir um tom explícito).\n\n"
    "{operator_block}"
    "Contexto:\n{context}\n\n"
    "Pergunta: {question}\n\n"
    "Resposta (baseada exclusivamente no contexto e respeitando as regras acima):"
)


def _build_strict_stuff_prompt(extra_instructions: str) -> PromptTemplate:
    """Build the strict anti-hallucination RAG prompt with optional operator notes.

    Args:
        extra_instructions: Optional text appended in ``partial_variables`` (may be empty).

    Returns:
        A ``PromptTemplate`` with ``context`` and ``question`` as input variables.
    """
    extra = (extra_instructions or "").strip()
    if extra:
        op = (
            "Instruções adicionais do operador (não violem as regras anteriores; nada disso introduz conhecimento além do contexto):\n"
            + extra
            + "\n\n"
        )
    else:
        op = ""
    return PromptTemplate(
        input_variables=["context", "question"],
        partial_variables={"operator_block": op},
        template=_RAG_STRICT_STUFF_HEAD,
    )


def _stable_chunk_id(doc: Document, idx: int) -> str:
    """Build a deterministic chunk id from metadata and content hash.

    Args:
        doc: LangChain document chunk.
        idx: Chunk index within the split list.

    Returns:
        String id suitable for Chroma ``ids=``.
    """
    import hashlib

    src = str(doc.metadata.get("source") or "")
    page = str(doc.metadata.get("page") if doc.metadata.get("page") is not None else "")
    h = hashlib.sha1(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{src}:{page}:{idx}:{h}"


def clear_vectorstore(settings: Settings) -> None:
    """Remove persisted Chroma data on disk for a clean re-index.

    Args:
        settings: Configuration; uses ``settings.chroma_dir``.
    """
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
        logger.debug("Chroma removido: %s", settings.chroma_dir)


def build_embeddings(settings: Settings):
    """Create an embeddings instance for Ollama or Hugging Face.

    Args:
        settings: Must set ``embed_provider`` and ``embed_model`` (and Ollama base URL when needed).

    Returns:
        A LangChain-compatible embeddings object.
    """
    if settings.embed_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        logger.debug("Carregando embeddings Ollama: model=%s", settings.embed_model)
        return OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url,
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    logger.debug("Carregando embeddings Hugging Face: model_name=%s", settings.embed_model)
    return HuggingFaceEmbeddings(model_name=settings.embed_model)


def build_vectorstore(settings: Settings):
    """Open or create a persisted Chroma collection with configured embeddings.

    Args:
        settings: Paths and embedding configuration.

    Returns:
        A ``Chroma`` vector store instance.
    """
    from langchain_chroma import Chroma

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    embeddings = build_embeddings(settings)
    return Chroma(
        collection_name="rag_local",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def split_documents(settings: Settings, docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks.

    Args:
        settings: Supplies ``chunk_size`` and ``chunk_overlap``.
        docs: Original loaded documents.

    Returns:
        Chunked documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def index_documents(
    settings: Settings,
    docs: List[Document],
    *,
    clear_existing: bool = True,
) -> Tuple[int, int]:
    """Index documents into Chroma and return document and chunk counts.

    By default deletes the Chroma directory before indexing to avoid duplicates.
    Use ``clear_existing=False`` to append to an existing index.

    Args:
        settings: Vector store and chunking configuration.
        docs: Documents to chunk and index.
        clear_existing: If True, remove ``settings.chroma_dir`` before adding.

    Returns:
        Tuple ``(number of original documents, number of chunks stored)``.
    """
    if clear_existing:
        clear_vectorstore(settings)

    vs = build_vectorstore(settings)
    chunks = split_documents(settings, docs)
    ids = [_stable_chunk_id(d, i) for i, d in enumerate(chunks)]
    vs.add_documents(chunks, ids=ids)
    return (len(docs), len(chunks))


def build_retriever(settings: Settings):
    """Build a similarity retriever over the persisted vector store.

    Args:
        settings: Uses ``top_k`` as ``search_kwargs['k']``.

    Returns:
        LangChain retriever for the Chroma collection.
    """
    vs = build_vectorstore(settings)
    return vs.as_retriever(search_kwargs={"k": settings.top_k})


def _retrieval_qa(
    settings: Settings,
    llm,
    *,
    return_source_documents: bool,
    extra_instructions: str = "",
):
    """Create a ``RetrievalQA`` chain with the strict prompt template.

    Args:
        settings: RAG and retrieval configuration.
        llm: LangChain LLM instance.
        return_source_documents: Whether to return retrieved docs in the output.
        extra_instructions: Optional operator instructions merged into the prompt.

    Returns:
        A configured ``RetrievalQA`` chain.
    """
    from langchain_classic.chains import RetrievalQA

    retriever = build_retriever(settings)
    prompt = _build_strict_stuff_prompt(extra_instructions)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=return_source_documents,
    )


def answer(
    settings: Settings,
    llm,
    question: str,
    *,
    extra_instructions: str = "",
) -> str:
    """Run retrieval-augmented QA and return only the answer string.

    Retrieval uses ``question``; ``extra_instructions`` are injected into the prompt template.

    Args:
        settings: RAG configuration.
        llm: LangChain LLM.
        question: User query for retrieval and generation.
        extra_instructions: Optional operator notes (same as GUI system prompt section).

    Returns:
        Model answer text from the chain ``result`` field when present.
    """
    chain = _retrieval_qa(
        settings, llm, return_source_documents=False, extra_instructions=extra_instructions
    )
    res = chain.invoke({"query": question})
    if isinstance(res, dict) and "result" in res:
        return str(res["result"])
    return str(res)


def answer_with_sources(
    settings: Settings,
    llm,
    question: str,
    *,
    extra_instructions: str = "",
):
    """Run retrieval-augmented QA and return the full chain output including sources.

    Args:
        settings: RAG configuration.
        llm: LangChain LLM.
        question: User query.
        extra_instructions: Optional operator notes for the prompt.

    Returns:
        Chain invoke result (typically a dict with ``result`` and ``source_documents``).
    """
    chain = _retrieval_qa(
        settings, llm, return_source_documents=True, extra_instructions=extra_instructions
    )
    return chain.invoke({"query": question})
