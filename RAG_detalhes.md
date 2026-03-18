# RAG_detalhes — bibliotecas, instalação e papel de cada uma

Este documento descreve **todas as bibliotecas utilizadas** no projeto `rag_project`, **como instalar** e **qual a utilidade** de cada dependência dentro do sistema RAG.

## Visão geral do projeto

O projeto implementa um fluxo de **RAG (Retrieval-Augmented Generation)**:

- **Carregamento** de documentos (`.pdf`, `.txt`, `.md`)
- **Quebra em chunks** (splitter)
- **Embeddings** (vetorização)
- **Vector store** persistente (**Chroma**) para busca semântica
- **Retriever** (top-k)
- **LLM** para gerar resposta usando os trechos recuperados
- Interface:
  - **CLI** (`rag ingest`, `rag ask`, etc.)
  - **GUI PySide6** (`rag-gui`)

Backends de LLM suportados:

- **Ollama** (local)
- **Hugging Face Transformers** (local)
- **GGUF / llama.cpp** (local, via `llama-cpp-python`)
- **API OpenAI-compatible** (pago, opcional)

## Instalação (Windows / PowerShell)

Na raiz do projeto (`E:\projetos_python\rag_project`):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Crie seu `.env`:

```bash
copy .env.example .env
```

Comandos principais:

```bash
rag ingest --reset
rag ask "Sua pergunta aqui"
rag-gui
```

## Bibliotecas usadas (requirements.txt)

Abaixo está a lista de bibliotecas do `requirements.txt` e o **papel** de cada uma no projeto.

### Núcleo LangChain (orquestração do RAG)

- **`langchain`**
  - **Papel**: APIs principais e estruturas de alto nível para montar chains (no projeto, usamos `langchain-classic` indiretamente via `langchain-community`/`langchain` para `RetrievalQA`).
  - **Utilidade no RAG**: orquestra a chamada “retrieval + geração”.

- **`langchain-community`**
  - **Papel**: integrações “community” (loaders, vectorstores e alguns LLMs).
  - **Utilidade no RAG**: loaders (ex.: `PyPDFLoader`, `TextLoader`) e integrações auxiliares.

- **`langchain-text-splitters`**
  - **Papel**: splitters oficiais.
  - **Utilidade no RAG**: `RecursiveCharacterTextSplitter` para quebrar documentos em chunks com overlap.

### Vector store (busca semântica)

- **`chromadb`**
  - **Papel**: banco vetorial Chroma.
  - **Utilidade no RAG**: persistência e consulta de embeddings.

- **`langchain-chroma`**
  - **Papel**: integração oficial LangChain ↔ Chroma (substitui import antigo de `langchain_community.vectorstores.Chroma`).
  - **Utilidade no RAG**: cria/abre coleção, faz `add_documents` e `as_retriever`.

### Embeddings (vetorização)

- **`sentence-transformers`**
  - **Papel**: modelos de embeddings (ex.: `all-MiniLM-L6-v2`).
  - **Utilidade no RAG**: transformar texto em vetores para busca semântica.

- **`langchain-huggingface`**
  - **Papel**: integrações Hugging Face “oficiais” para LangChain.
  - **Utilidade no RAG**:
    - `HuggingFaceEmbeddings` (embeddings)
    - `HuggingFacePipeline` (LLM via transformers na GUI/CLI)

### LLM local via Transformers (Hugging Face)

- **`transformers`**
  - **Papel**: carregar tokenizer/modelo e rodar geração local (`pipeline("text-generation")`).
  - **Utilidade no RAG**: backend de LLM local (quando provider = `hf`).

- **`torch`**
  - **Papel**: runtime de deep learning para o `transformers`.
  - **Utilidade no RAG**: executar o modelo em CPU/GPU (quando disponível).

### LLM local via Ollama

- **`langchain-ollama`**
  - **Papel**: wrapper LangChain para Ollama.
  - **Utilidade no RAG**: `ChatOllama` para chamar modelos locais servidos pelo Ollama.

> Observação: o executável/serviço **Ollama** não vem pelo `pip`; ele é instalado separadamente.

### LLM local via GGUF (llama.cpp)

- **`llama-cpp-python`**
  - **Papel**: bindings Python do **llama.cpp**.
  - **Utilidade no RAG**: rodar modelos `.gguf` localmente (quantizados) via `LlamaCpp` (LangChain community).

> Observação: no Windows, essa lib pode compilar durante o `pip install` (leva alguns minutos).  
> Para usar GPU com GGUF, você normalmente precisa de build com suporte CUDA e configurar `GGUF_N_GPU_LAYERS`.

### Leitura de documentos

- **`pypdf`**
  - **Papel**: biblioteca base para leitura de PDF.
  - **Utilidade no RAG**: utilizada por `PyPDFLoader` para extrair texto de `.pdf`.

### Configuração e DX (developer experience)

- **`python-dotenv`**
  - **Papel**: carregar variáveis do `.env`.
  - **Utilidade no RAG**: configuração de provider/modelos/pastas sem hardcode.

- **`typer`**
  - **Papel**: framework de CLI.
  - **Utilidade no RAG**: comandos `rag ingest`, `rag ask`, `rag info`, `rag models`.

- **`rich`**
  - **Papel**: UI/saída bonita no terminal.
  - **Utilidade no RAG**: painéis e mensagens formatadas na CLI.

### Interface gráfica (GUI)

- **`pyside6`**
  - **Papel**: toolkit Qt para GUI.
  - **Utilidade no RAG**: cria a interface `rag-gui` com abas por provedor, chat e histórico.

### Exportar resposta (TXT / PDF)

- **`reportlab`**
  - **Papel**: gerar PDFs.
  - **Utilidade no RAG**: salvar resposta em `.pdf` via GUI.

### API paga / OpenAI-compatible

- **`langchain-openai`**
  - **Papel**: integração LangChain para APIs OpenAI-compatible.
  - **Utilidade no RAG**: permitir provider “api” com `base_url`, `api_key` e listagem de modelos.

- **`httpx`**
  - **Papel**: cliente HTTP moderno.
  - **Utilidade no RAG**: listar modelos em APIs compatíveis (endpoint `/models`) pela GUI.

## Onde cada parte é usada no código

- **CLI**: `src/rag_local/cli.py`
  - Indexação (`ingest`) e perguntas (`ask`)
- **GUI**: `src/rag_local/gui.py`
  - Abas de provider + chat/histórico + salvar TXT/PDF
- **Config (.env)**: `src/rag_local/config.py`
  - lê `RAG_PROVIDER`, `HF_*`, `GGUF_*`, paths etc.
- **LLMs**: `src/rag_local/llm.py`
  - cria o LLM (Ollama / HF / GGUF)
- **RAG core**: `src/rag_local/rag.py`
  - embeddings, Chroma, retriever, chain para responder
- **Loaders**: `src/rag_local/loaders.py`
  - carrega `.pdf/.md/.txt`

