from __future__ import annotations
from typing import Any
from rag_local.config import Settings
from pathlib import Path
from langchain_core.language_models.llms import LLM


class OpenAISdkLLM(LLM):
    api_key: str
    model: str

    @property
    def _llm_type(self) -> str:
        return "openai-sdk"

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager=None, **kwargs: Any) -> str:
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
    api_key: str
    model: str

    @property
    def _llm_type(self) -> str:
        return "mistral-sdk"

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager=None, **kwargs: Any) -> str:
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
    api_key: str
    model: str

    @property
    def _llm_type(self) -> str:
        return "google-generativeai-sdk"

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager=None, **kwargs: Any) -> str:
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
    - provider=api: usa SDK oficial (OpenAI, Mistral ou Google Generative AI)
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
