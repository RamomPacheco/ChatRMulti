# RAG Local (LangChain + Ollama + HF + APIs)

Projeto Python para rodar um **sistema RAG** (Retrieval-Augmented Generation) usando **LLMs locais e APIs**, com os backends:

- **Ollama** (ex.: Llama 3.1, Mistral, etc.)
- **Hugging Face Transformers** (pipeline local)
- **API (SDK oficial)**: OpenAI, Mistral e Google Generative AI

O armazenamento vetorial é o **Chroma** (persistente em disco).

## Requisitos

- Python **3.10+**
- (Opcional) **Ollama** instalado e rodando, se você for usar `RAG_PROVIDER=ollama`

## Instalação

No Windows (PowerShell), dentro da pasta do projeto:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Crie seu `.env` a partir do exemplo:

```bash
copy .env.example .env
```

## Uso rápido

### 1) Indexar documentos

Coloque arquivos em `docs/` (suporta `.pdf`, `.txt`, `.md`) e rode:

```bash
rag ingest
```

Ou indique outra pasta:

```bash
rag ingest --docs caminho\para\documentos
```

### 2) Perguntar

```bash
rag ask "Quais backends de LLM estão disponíveis?"
```

## Usando Ollama

1. Instale o Ollama e baixe um modelo (exemplo):

```bash
ollama pull llama3.1
```

2. No `.env`:

```text
RAG_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

## Usando Hugging Face (transformers)

No `.env`:

```text
RAG_PROVIDER=hf
HF_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
HF_DEVICE=-1
```

- **HF_DEVICE=-1**: CPU
- **HF_DEVICE=0**: GPU (se disponível)

## Usando API com SDK oficial

No `.env`, habilite provider `api` e escolha o SDK:

```text
RAG_PROVIDER=api
API_PROVIDER=openai   # openai | mistral | google
```

### OpenAI

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

### Mistral

```text
MISTRAL_API_KEY=...
MISTRAL_MODEL=mistral-small-latest
```

### Google Generative AI

```text
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-1.5-flash
```

Na GUI, use a aba **API**, selecione o provider SDK e informe a chave correspondente.

## Comandos úteis

Ver configuração efetiva:

```bash
rag info
```

## Estrutura

- `src/rag_local/cli.py`: CLI (`ingest`, `ask`, `info`)
- `src/rag_local/loaders.py`: carregamento de documentos
- `src/rag_local/rag.py`: embeddings, Chroma e chain de QA
- `src/rag_local/llm.py`: seleção do backend (Ollama/HF/GGUF/API SDK)
- `src/rag_local/config.py`: configuração via `.env`
