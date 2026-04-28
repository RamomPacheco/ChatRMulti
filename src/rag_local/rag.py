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
    import hashlib

    src = str(doc.metadata.get("source") or "")
    page = str(doc.metadata.get("page") if doc.metadata.get("page") is not None else "")
    h = hashlib.sha1(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{src}:{page}:{idx}:{h}"


def clear_vectorstore(settings: Settings) -> None:
    """Remove dados persistidos do Chroma (.chroma) para próxima indexação limpa."""
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
        logger.debug("Chroma removido: %s", settings.chroma_dir)


def build_embeddings(settings: Settings):
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


def index_documents(
    settings: Settings,
    docs: List[Document],
    *,
    clear_existing: bool = True,
) -> Tuple[int, int]:
    """
    Retorna (num_docs_originais, num_chunks).

    Por padrão apaga o diretório Chroma antes de indexar (evita duplicatas e garante estado limpo).
    Use clear_existing=False para acrescentar ao índice existente (modo avançado).
    """
    if clear_existing:
        clear_vectorstore(settings)

    vs = build_vectorstore(settings)
    chunks = split_documents(settings, docs)
    ids = [_stable_chunk_id(d, i) for i, d in enumerate(chunks)]
    vs.add_documents(chunks, ids=ids)
    return (len(docs), len(chunks))


def build_retriever(settings: Settings):
    vs = build_vectorstore(settings)
    return vs.as_retriever(search_kwargs={"k": settings.top_k})


def _retrieval_qa(
    settings: Settings,
    llm,
    *,
    return_source_documents: bool,
    extra_instructions: str = "",
):
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
    """A recuperação de trechos usa só a pergunta do utilizador; o prompt rígido e notas do operador vão no template."""
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
    chain = _retrieval_qa(
        settings, llm, return_source_documents=True, extra_instructions=extra_instructions
    )
    return chain.invoke({"query": question})
