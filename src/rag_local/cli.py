from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rag_local.config import get_settings
from rag_local.llm import build_llm, list_ollama_models
from rag_local.loaders import load_documents
from rag_local.rag import answer, answer_with_sources, index_documents


app = typer.Typer(add_completion=False, help="RAG local (Ollama/HF) com LangChain.")
console = Console()


@app.command()
def ingest(
    docs_dir: Optional[Path] = typer.Option(
        None, "--docs", "-d", help="Pasta com .pdf/.txt/.md para indexar."
    ),
    reset: bool = typer.Option(
        False, "--reset", help="Apaga o índice local (.chroma) antes de indexar."
    ),
):
    """
    Indexa documentos no Chroma (persistente).
    """
    settings = get_settings()
    effective_docs_dir = docs_dir or settings.docs_dir

    if reset and settings.chroma_dir.exists():
        import shutil

        shutil.rmtree(settings.chroma_dir)

    console.print(
        Panel.fit(
            f"[bold]Indexando[/bold]\n"
            f"- provider: [bold]{settings.rag_provider}[/bold]\n"
            f"- docs_dir: [bold]{effective_docs_dir}[/bold]\n"
            f"- chroma_dir: [bold]{settings.chroma_dir}[/bold]\n"
            f"- embed_model: [bold]{settings.embed_model}[/bold]\n"
        )
    )

    docs = load_documents(effective_docs_dir)
    try:
        from collections import Counter

        c = Counter(d.metadata.get("source") for d in docs)
        breakdown = "\n".join(f"- {k}: {v}" for k, v in c.items())
        console.print(Panel.fit(breakdown or "(vazio)", title="Docs carregados por arquivo"))
    except Exception:
        pass
    n_docs, n_chunks = index_documents(settings, docs)
    console.print(f"[bold green]OK[/bold green] docs={n_docs} chunks={n_chunks}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Pergunta para o seu RAG."),
    sources: bool = typer.Option(
        False, "--sources", help="Mostra os trechos/fontes recuperados."
    ),
):
    """
    Faz uma pergunta usando recuperação + LLM local.
    """
    settings = get_settings()
    try:
        llm = build_llm(settings)
    except Exception as e:
        console.print(Panel(str(e), title="Erro ao configurar LLM", style="red"))
        raise typer.Exit(code=2) from e

    used = getattr(llm, "_rag_provider_used", settings.rag_provider)

    console.print(
        Panel.fit(
            f"[bold]Pergunta[/bold]\n"
            f"- provider: [bold]{used}[/bold]\n"
            f"- top_k: [bold]{settings.top_k}[/bold]\n"
            f"- chroma_dir: [bold]{settings.chroma_dir}[/bold]\n"
            f"\n[bold]{question}[/bold]"
        )
    )

    try:
        if sources:
            res = answer_with_sources(settings, llm, question)
            out = res.get("result") if isinstance(res, dict) else str(res)
            console.print(Panel(str(out), title="Resposta", expand=False))

            src_docs = res.get("source_documents", []) if isinstance(res, dict) else []
            if src_docs:
                lines = []
                for i, d in enumerate(src_docs, 1):
                    src = d.metadata.get("source")
                    page = d.metadata.get("page")
                    snippet = (d.page_content or "").strip().replace("\n", " ")
                    if len(snippet) > 240:
                        snippet = snippet[:240] + "…"
                    lines.append(f"{i}. source={src} page={page} :: {snippet}")
                console.print(Panel("\n".join(lines), title="Sources (top hits)", expand=False))
        else:
            out = answer(settings, llm, question)
            console.print(Panel(out, title="Resposta", expand=False))
    except Exception as e:
        msg = str(e)
        if "model" in msg.lower() and "not found" in msg.lower() and settings.rag_provider == "ollama":
            models = list_ollama_models(settings.ollama_base_url)
            hint = "\n".join(
                [
                    "O Ollama respondeu que o modelo não existe.",
                    f"OLLAMA_MODEL={settings.ollama_model!r}",
                    f"Modelos locais: {models if models else '(não consegui listar)'}",
                    "Dica: rode `ollama pull <nome-do-modelo>` e tente novamente.",
                ]
            )
            console.print(Panel(hint, title="Erro do Ollama", style="red"))
            raise typer.Exit(code=3) from e

        console.print(Panel(msg, title="Erro ao gerar resposta", style="red"))
        raise typer.Exit(code=1) from e


@app.command()
def models():
    """
    Lista modelos disponíveis para o provider atual.
    """
    s = get_settings()
    if s.rag_provider == "ollama":
        ms = list_ollama_models(s.ollama_base_url)
        console.print(Panel("\n".join(ms) if ms else "(nenhum encontrado)", title="Ollama models"))
        return

    console.print(Panel(s.hf_model, title="HF model"))


@app.command()
def info():
    """
    Mostra a configuração efetiva (via .env / env vars).
    """
    s = get_settings()
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"provider={s.rag_provider}",
                    f"ollama_model={s.ollama_model}",
                    f"ollama_base_url={s.ollama_base_url}",
                    f"hf_model={s.hf_model}",
                    f"hf_device={s.hf_device}",
                    f"hf_max_new_tokens={s.hf_max_new_tokens}",
                    f"embed_model={s.embed_model}",
                    f"docs_dir={s.docs_dir}",
                    f"chroma_dir={s.chroma_dir}",
                    f"chunk_size={s.chunk_size}",
                    f"chunk_overlap={s.chunk_overlap}",
                    f"top_k={s.top_k}",
                ]
            ),
            title="Config",
        )
    )


if __name__ == "__main__":
    app()
