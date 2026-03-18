# RAG Local (LangChain + Ollama + HF + GGUF + API)

Projeto Python para rodar um **sistema RAG** (Retrieval-Augmented Generation) usando **LLMs locais** (e opcionalmente API paga).

Backends suportados:

- **Ollama** (ex.: Llama 3.1, Qwen, etc.)
- **Hugging Face Transformers** (pipeline local)
- **GGUF / llama.cpp** (modelos `.gguf` via `llama-cpp-python`)
- **API OpenAI-compatible** (opcional, modelos pagos)

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
rag ingest --reset
```

Ou indique outra pasta:

```bash
rag ingest --docs caminho\para\documentos --reset
```

### 2) Perguntar

```bash
rag ask "Quais backends de LLM estão disponíveis?"
```

Ver as fontes (trechos recuperados):

```bash
rag ask "Sobre o que é o PDF?" --sources
```

## Interface gráfica (PySide6)

Para abrir a interface:

```bash
rag-gui
```

Na GUI:

- Cada provedor (aba) tem seu próprio **chat**: Pergunta/Resposta/Fontes
- Cada provedor tem seu próprio **histórico** (clicável) e ele é persistido
- Há bloco de **Indexação** com reset e escolha da pasta `docs`

## Usando Ollama

1. Instale o Ollama e baixe um modelo (exemplo):

```bash
ollama pull llama3.1:8b
```

2. No `.env`:

```text
RAG_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```

Dica: para ver os nomes exatos dos modelos locais, rode:

```bash
rag models
```

## Usando Hugging Face (transformers)

No `.env`:

```text
RAG_PROVIDER=hf
HF_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
HF_DEVICE=-1
HF_CACHE_DIR=E:\ComfyUI\models\LLM
```

- **HF_DEVICE=-1**: CPU
- **HF_DEVICE=0**: GPU (se disponível)

Por padrão, os modelos do Hugging Face são baixados/cacheados em `E:\ComfyUI\models\LLM` (pode alterar via `HF_CACHE_DIR`).

No Windows, você pode ver um aviso sobre **symlinks** no cache do Hugging Face. Isso não impede de funcionar; para melhorar, ative o **Developer Mode** do Windows ou rode o Python como administrador. Para apenas silenciar o aviso, defina `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

## Usando GGUF (llama.cpp)

Se você tem um modelo `.gguf` baixado (ex.: dentro de `E:\ComfyUI\models\LLM`), você pode usar o backend GGUF:

```text
RAG_PROVIDER=gguf
GGUF_MODEL_PATH=E:\ComfyUI\models\LLM\seu-modelo\modelo.gguf
GGUF_N_CTX=4096
GGUF_N_GPU_LAYERS=0
```

Observações:

- **GGUF_N_GPU_LAYERS=0**: CPU
- Se você tiver build com suporte CUDA, aumente `GGUF_N_GPU_LAYERS` para offload na GPU (depende do modelo/VRAM).

## Usando API (OpenAI-compatible)

Na GUI (aba **API**), preencha `base_url`, `api_key` e clique em **Listar modelos**.

Exemplo de `base_url`:

- `https://api.openai.com/v1` (OpenAI)
- ou um endpoint compatível (ex.: servidores OpenAI-like)

## Fallback automático (Ollama → HF)

Se `RAG_PROVIDER=ollama` mas o servidor Ollama não estiver acessível, o projeto faz **fallback automático** para Hugging Face.

## Comandos úteis

Ver configuração efetiva:

```bash
rag info
```

Listar modelos do Ollama:

```bash
rag models
```

## Estrutura

- `src/rag_local/cli.py`: CLI (`ingest`, `ask`, `info`)
- `src/rag_local/loaders.py`: carregamento de documentos
- `src/rag_local/rag.py`: embeddings, Chroma e chain de QA
- `src/rag_local/llm.py`: seleção do backend (Ollama/HF/GGUF)
- `src/rag_local/config.py`: configuração via `.env`
- `src/rag_local/gui.py`: interface gráfica (PySide6)

## Documentação extra

- `RAG_detalhes.md`: bibliotecas usadas, instalação e utilidade de cada uma
