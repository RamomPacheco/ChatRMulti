from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration loaded from environment variables.

    See Also:
        get_settings: Factory that reads ``os.environ`` and returns ``Settings``.
    """

    rag_provider: str

    ollama_model: str
    ollama_base_url: str

    hf_model: str
    hf_device: int
    hf_max_new_tokens: int
    hf_cache_dir: Path

    gguf_model_path: str
    gguf_n_ctx: int
    gguf_n_gpu_layers: int
    api_provider: str
    openai_api_key: str
    openai_model: str
    mistral_api_key: str
    mistral_model: str
    google_api_key: str
    google_model: str

    embed_model: str
    embed_provider: str

    docs_dir: Path
    chroma_dir: Path

    chunk_size: int
    chunk_overlap: int
    top_k: int


def _get_int(name: str, default: int) -> int:
    """Parse an environment variable as ``int``.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or empty.

    Returns:
        Parsed integer, or ``default`` when unset/empty.

    Raises:
        ValueError: If the variable is set but not a valid integer.
    """
    raw = os.getenv(name, "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Variável {name} deve ser int (recebido: {raw!r})") from e


def _get_str(name: str, default: str) -> str:
    """Return a string environment variable or a default.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or empty.

    Returns:
        Stripped non-empty value from the environment, or ``default``.
    """
    raw = os.getenv(name, "").strip()
    return raw if raw else default


def get_settings() -> Settings:
    """Load ``.env`` (without overriding existing variables) and build ``Settings``.

    Returns:
        Validated ``Settings`` instance.

    Raises:
        ValueError: If ``EMBED_PROVIDER``, ``API_PROVIDER``, or related values are invalid.
    """
    load_dotenv(override=False)

    raw_ep = os.getenv("EMBED_PROVIDER", "").strip()
    raw_em = os.getenv("EMBED_MODEL", "").strip()
    if not raw_ep:
        if not raw_em:
            embed_model, ep_norm = "nomic-embed-text:latest", "ollama"
        # Caminhos org/modelo (HF) costumam ter "/"; modelos Ollama usam tag "nome:tag".
        elif "/" in raw_em and ":" not in raw_em:
            embed_model, ep_norm = raw_em, "huggingface"
        else:
            embed_model, ep_norm = raw_em, "ollama"
    else:
        ep = raw_ep.lower()
        if ep in ("huggingface", "hf", "hugging_face"):
            ep_norm = "huggingface"
        elif ep == "ollama":
            ep_norm = "ollama"
        else:
            raise ValueError(
                "EMBED_PROVIDER deve ser 'ollama' ou 'huggingface' (ou hf)."
            )
        default_em = (
            "sentence-transformers/all-MiniLM-L6-v2"
            if ep_norm == "huggingface"
            else "nomic-embed-text:latest"
        )
        embed_model = _get_str("EMBED_MODEL", default_em)

    docs_dir = Path(_get_str("DOCS_DIR", "docs"))
    chroma_dir = Path(_get_str("CHROMA_DIR", ".chroma"))
    hf_cache_dir = Path(_get_str("HF_CACHE_DIR", r"E:\ComfyUI\models\LLM"))
    api_provider = _get_str("API_PROVIDER", "openai").lower()
    if api_provider not in {"openai", "mistral", "google"}:
        raise ValueError("API_PROVIDER deve ser 'openai', 'mistral' ou 'google'.")

    return Settings(
        rag_provider=_get_str("RAG_PROVIDER", "ollama").lower(),
        ollama_model=_get_str("OLLAMA_MODEL", "llama3.1"),
        ollama_base_url=_get_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        hf_model=_get_str("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        hf_device=_get_int("HF_DEVICE", -1),
        hf_max_new_tokens=_get_int("HF_MAX_NEW_TOKENS", 512),
        hf_cache_dir=hf_cache_dir,
        gguf_model_path=_get_str("GGUF_MODEL_PATH", ""),
        gguf_n_ctx=_get_int("GGUF_N_CTX", 4096),
        gguf_n_gpu_layers=_get_int("GGUF_N_GPU_LAYERS", 0),
        api_provider=api_provider,
        openai_api_key=_get_str("OPENAI_API_KEY", ""),
        openai_model=_get_str("OPENAI_MODEL", "gpt-4o-mini"),
        mistral_api_key=_get_str("MISTRAL_API_KEY", ""),
        mistral_model=_get_str("MISTRAL_MODEL", "mistral-small-latest"),
        google_api_key=_get_str("GOOGLE_API_KEY", ""),
        google_model=_get_str("GOOGLE_MODEL", "gemini-1.5-flash"),
        embed_model=embed_model,
        embed_provider=ep_norm,
        docs_dir=docs_dir,
        chroma_dir=chroma_dir,
        chunk_size=_get_int("CHUNK_SIZE", 1000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
        top_k=_get_int("TOP_K", 4),
    )
