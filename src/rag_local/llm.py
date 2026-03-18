from __future__ import annotations
from typing import Any
from rag_local.config import Settings
from pathlib import Path


def list_ollama_models(base_url: str) -> list[str]:
    try:
        import ollama
    except Exception:
        return []

    try:
        client = ollama.Client(host=base_url)
        data = client.list()
    except Exception:
        return []

    models: list[str] = []
    # A lib `ollama` pode retornar um dict-like ou objetos Model.
    raw_models = None
    if isinstance(data, dict):
        raw_models = data.get("models")
    else:
        raw_models = getattr(data, "models", None)

    for m in (raw_models or []):
        if isinstance(m, dict):
            name = (m.get("name") or m.get("model") or "").strip()
        else:
            name = (getattr(m, "name", None) or getattr(m, "model", None) or "").strip()
        if name:
            models.append(name)
    return models


def _build_hf_llm(settings: Settings) -> Any:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import HuggingFacePipeline

    settings.hf_cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        settings.hf_model,
        cache_dir=str(settings.hf_cache_dir),
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.hf_model,
        cache_dir=str(settings.hf_cache_dir),
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=settings.hf_device,
        max_new_tokens=settings.hf_max_new_tokens,
        do_sample=True,
        temperature=0.2,
    )
    llm = HuggingFacePipeline(pipeline=gen)
    setattr(llm, "_rag_provider_used", "hf")
    return llm


def _build_gguf_llm(settings: Settings) -> Any:
    from langchain_community.llms import LlamaCpp

    model_path = (settings.gguf_model_path or "").strip()
    if not model_path:
        raise ValueError("GGUF_MODEL_PATH não definido (caminho para um .gguf).")

    p = Path(model_path)
    if not p.exists() or p.suffix.lower() != ".gguf":
        raise ValueError(f"GGUF_MODEL_PATH inválido: {model_path!r} (esperado arquivo .gguf)")

    llm = LlamaCpp(
        model_path=str(p),
        n_ctx=settings.gguf_n_ctx,
        n_gpu_layers=settings.gguf_n_gpu_layers,
        temperature=0.2,
        max_tokens=settings.hf_max_new_tokens,
        verbose=False,
    )
    setattr(llm, "_rag_provider_used", "gguf")
    return llm


def _ollama_reachable(base_url: str) -> bool:
    try:
        import ollama
    except Exception:
        return False

    try:
        client = ollama.Client(host=base_url)
        client.list()
        return True
    except Exception:
        return False


def build_llm(settings: Settings) -> Any:
    """
    Retorna um LLM compatível com LangChain.

    - provider=ollama: usa `ChatOllama` (LLM local via Ollama)
    - provider=hf: usa `HuggingFacePipeline` (modelo local via transformers)
    - provider=gguf: usa `LlamaCpp` (llama.cpp) com arquivo .gguf local
    """
    provider = (settings.rag_provider or "").lower().strip()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # Fallback automático para HF quando o servidor Ollama não estiver acessível.
        if not _ollama_reachable(settings.ollama_base_url):
            return _build_hf_llm(settings)

        available = list_ollama_models(settings.ollama_base_url)
        desired = (settings.ollama_model or "").strip()
        resolved_model = desired

        if available:
            if not desired:
                resolved_model = available[0]
            elif desired not in available:
                # Tenta casar "llama3.1" com "llama3.1:8b", etc.
                pref = [m for m in available if m == desired or m.startswith(desired + ":")]
                if len(pref) == 1:
                    resolved_model = pref[0]
                else:
                    raise ValueError(
                        "Modelo do Ollama não encontrado. "
                        f"OLLAMA_MODEL={desired!r}. "
                        f"Modelos disponíveis: {sorted(set(available))}. "
                        "Dica: use exatamente um nome da lista (ex.: 'llama3.1:8b') "
                        f"ou rode `ollama pull {desired}`."
                    )
        elif desired:
            # Sem lista disponível (ex.: falha no /api/tags), segue e deixa a validação para o invoke
            resolved_model = desired

        return ChatOllama(
            model=resolved_model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )

    if provider in {"hf", "huggingface", "transformers"}:
        return _build_hf_llm(settings)

    if provider in {"gguf", "llamacpp", "llama.cpp"}:
        return _build_gguf_llm(settings)

    raise ValueError(
        "RAG_PROVIDER inválido. Use 'ollama', 'hf' ou 'gguf'. "
        f"Recebido: {settings.rag_provider!r}"
    )
