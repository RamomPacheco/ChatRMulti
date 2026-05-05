from __future__ import annotations
from typing import Any
from rag_local.config import Settings
from pathlib import Path
from langchain_core.language_models.llms import LLM


class OpenAISdkLLM(LLM):
    """LangChain LLM using the official OpenAI Chat Completions API."""

    api_key: str
    model: str

    @property
    def _llm_type(self) -> str:
        return "openai-sdk"

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager=None, **kwargs: Any) -> str:
        """Run a chat completion with a single user message.

        Args:
            prompt: Full prompt text sent as user content.
            stop: Stop sequences (ignored; LangChain hook compatibility).
            run_manager: Callback manager (ignored).
            **kwargs: Extra arguments (ignored).

        Returns:
            Trimmed assistant message content.
        """
        del stop, run_manager, kwargs
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        res = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        msg = (res.choices[0].message.content or "").strip() if getattr(res, "choices", None) else ""
        return msg


class MistralSdkLLM(LLM):
    """LangChain LLM using the Mistral client chat API."""

    api_key: str
    model: str

    @property
    def _llm_type(self) -> str:
        return "mistral-sdk"

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager=None, **kwargs: Any) -> str:
        """Run chat completion, supporting multiple Mistral SDK variants.

        Args:
            prompt: User message content.
            stop: Unused (compatibility).
            run_manager: Unused.
            **kwargs: Unused.

        Returns:
            Trimmed assistant text.
        """
        del stop, run_manager, kwargs
        from mistralai import Mistral

        client = Mistral(api_key=self.api_key)
        # SDK atual
        if hasattr(client, "chat") and hasattr(client.chat, "complete"):
            res = client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            if getattr(res, "choices", None):
                content = getattr(res.choices[0].message, "content", "")
                return content.strip() if isinstance(content, str) else str(content).strip()
        # Compatibilidade com versões antigas
        res = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (res.choices[0].message.content or "").strip()


class GoogleSdkLLM(LLM):
    """LangChain LLM using Google Generative AI ``GenerativeModel.generate_content``."""

    api_key: str
    model: str

    @property
    def _llm_type(self) -> str:
        return "google-generativeai-sdk"

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager=None, **kwargs: Any) -> str:
        """Generate text from ``prompt`` via Gemini.

        Args:
            prompt: Raw prompt string.
            stop: Unused.
            run_manager: Unused.
            **kwargs: Unused.

        Returns:
            Generated text, or empty string if none.
        """
        del stop, run_manager, kwargs
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        gm = genai.GenerativeModel(self.model)
        res = gm.generate_content(prompt)
        txt = getattr(res, "text", None)
        if txt:
            return str(txt).strip()
        # fallback para objetos com candidates/parts
        candidates = getattr(res, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", []) if getattr(candidates[0], "content", None) else []
            joined = "".join(getattr(p, "text", "") for p in parts)
            return joined.strip()
        return ""


def list_ollama_models(base_url: str) -> list[str]:
    """List installed Ollama models exposed by the server.

    Args:
        base_url: Ollama API base URL (e.g. ``http://localhost:11434``).

    Returns:
        Model names, or an empty list if the client is missing or the call fails.
    """
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


def list_openai_models(api_key: str) -> list[str]:
    """List OpenAI model ids visible to the given API key.

    Args:
        api_key: OpenAI API key.

    Returns:
        Sorted unique model ids, or empty list if unavailable.
    """
    if not api_key:
        return []
    try:
        from openai import OpenAI
    except Exception:
        return []
    try:
        client = OpenAI(api_key=api_key)
        data = client.models.list()
        return sorted({m.id for m in data.data if getattr(m, "id", "")})
    except Exception:
        return []


def list_mistral_models(api_key: str) -> list[str]:
    """List Mistral model ids when the client supports ``models.list``.

    Args:
        api_key: Mistral API key.

    Returns:
        Sorted unique model ids, or empty list on failure.
    """
    if not api_key:
        return []
    try:
        from mistralai import Mistral
    except Exception:
        return []
    try:
        client = Mistral(api_key=api_key)
        if hasattr(client, "models") and hasattr(client.models, "list"):
            data = client.models.list()
            items = getattr(data, "data", data) or []
            return sorted({getattr(m, "id", "") for m in items if getattr(m, "id", "")})
    except Exception:
        return []
    return []


def list_google_models(api_key: str) -> list[str]:
    """List Google Generative AI models that support generation, without ``models/`` prefix.

    Args:
        api_key: Google API key.

    Returns:
        Sorted model name strings, or empty list on failure.
    """
    if not api_key:
        return []
    try:
        import google.generativeai as genai
    except Exception:
        return []
    try:
        genai.configure(api_key=api_key)
        out: list[str] = []
        for m in genai.list_models():
            name = getattr(m, "name", "")
            methods = set(getattr(m, "supported_generation_methods", []) or [])
            if not name or ("generateContent" not in methods and "generateText" not in methods):
                continue
            out.append(name.replace("models/", ""))
        return sorted(set(out))
    except Exception:
        return []


def build_sdk_llm(api_provider: str, *, api_key: str, model: str) -> Any:
    """Build an SDK-backed LangChain LLM for cloud APIs.

    Args:
        api_provider: One of ``openai``, ``mistral``, ``google``.
        api_key: Provider API key.
        model: Model name or id for that provider.

    Returns:
        ``OpenAISdkLLM``, ``MistralSdkLLM``, or ``GoogleSdkLLM``.

    Raises:
        ValueError: If API key or model is missing, or provider is unknown.
    """
    p = (api_provider or "").strip().lower()
    if not api_key:
        raise ValueError(f"API key ausente para provider {p!r}.")
    if not model:
        raise ValueError(f"Modelo ausente para provider {p!r}.")

    if p == "openai":
        llm = OpenAISdkLLM(api_key=api_key, model=model)
        setattr(llm, "_rag_provider_used", "api:openai")
        return llm
    if p == "mistral":
        llm = MistralSdkLLM(api_key=api_key, model=model)
        setattr(llm, "_rag_provider_used", "api:mistral")
        return llm
    if p == "google":
        llm = GoogleSdkLLM(api_key=api_key, model=model)
        setattr(llm, "_rag_provider_used", "api:google")
        return llm
    raise ValueError("API_PROVIDER inválido. Use 'openai', 'mistral' ou 'google'.")


def _build_hf_llm(settings: Settings) -> Any:
    """Build a Hugging Face causal LM pipeline wrapped as ``HuggingFacePipeline``.

    Args:
        settings: HF model id, cache dir, device, and generation limits.

    Returns:
        LangChain ``HuggingFacePipeline``; sets ``_rag_provider_used`` to ``hf``.
    """
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
    """Build a ``LlamaCpp`` LLM from a local GGUF file path.

    Args:
        settings: Must define ``gguf_model_path`` and GGUF-related options.

    Returns:
        ``LlamaCpp`` instance; sets ``_rag_provider_used`` to ``gguf``.

    Raises:
        ValueError: If path is missing, missing file, or not a ``.gguf`` file.
    """
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
    """Return whether the Ollama HTTP API responds to ``list`` at ``base_url``.

    Args:
        base_url: Ollama server URL.

    Returns:
        True if listing succeeds; False if client missing or server unreachable.
    """
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
    """Return a LangChain LLM according to ``settings.rag_provider``.

    Behavior by provider:

    * ``ollama``: ``ChatOllama``; if the server is unreachable, falls back to HF pipeline.
    * ``hf`` / ``huggingface`` / ``transformers``: Hugging Face pipeline.
    * ``gguf`` / ``llamacpp`` / ``llama.cpp``: llama.cpp via ``LlamaCpp``.
    * ``api``: SDK wrapper for OpenAI, Mistral, or Google.

    Args:
        settings: Full configuration including provider-specific fields.

    Returns:
        A LangChain-compatible runnable LLM.

    Raises:
        ValueError: Unknown ``rag_provider``, missing Ollama model match, invalid GGUF path, etc.
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

    if provider == "api":
        keys = {
            "openai": settings.openai_api_key,
            "mistral": settings.mistral_api_key,
            "google": settings.google_api_key,
        }
        models = {
            "openai": settings.openai_model,
            "mistral": settings.mistral_model,
            "google": settings.google_model,
        }
        p = settings.api_provider
        return build_sdk_llm(p, api_key=keys.get(p, ""), model=models.get(p, ""))

    raise ValueError(
        "RAG_PROVIDER inválido. Use 'ollama', 'hf', 'gguf' ou 'api'. "
        f"Recebido: {settings.rag_provider!r}"
    )
