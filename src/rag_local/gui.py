from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QProcess, QSettings, Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QListWidget,
    QListWidgetItem,
    QTabWidget as QTabWidgetInner,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from rag_local.config import get_settings
from rag_local.llm import (
    build_sdk_llm,
    list_google_models,
    list_mistral_models,
    list_ollama_models,
    list_openai_models,
)
from rag_local.loaders import load_documents
from rag_local.rag import answer_with_sources, index_documents


def _configure_logging() -> None:
    lvl = logging.DEBUG if os.getenv("RAG_DEBUG", "").strip() in {"1", "true", "True", "yes"} else logging.INFO
    logging.basicConfig(level=lvl, format="%(levelname)s %(name)s: %(message)s")


logger = logging.getLogger(__name__)


@dataclass
class ApiConfig:
    provider: str
    model: str
    openai_api_key: str
    mistral_api_key: str
    google_api_key: str


def _list_hf_models_in_dir(root: Path) -> list[str]:
    """
    Lista modelos dentro de uma pasta de cache HF.
    Suporta estrutura padrão do huggingface_hub: models--org--name
    """
    root = Path(root)
    if not root.exists():
        return []
    models: list[str] = []
    for p in root.glob("models--*"):
        if p.is_dir():
            # models--ORG--NAME  -> ORG/NAME
            name = p.name.replace("models--", "").replace("--", "/")
            models.append(name)
    return sorted(set(models))


def _list_folder_models_simple(root: Path) -> list[str]:
    """
    Lista modelos quando o usuário aponta para uma pasta "direta" de modelos,
    por exemplo: E:\\ComfyUI\\models\\LLM contendo subpastas de modelos.
    """
    root = Path(root)
    if not root.exists():
        return []
    out: list[str] = []
    for p in root.iterdir():
        if p.is_dir():
            # heurística: uma pasta com config/tokenizer ou arquivos .gguf
            has_cfg = (p / "config.json").exists() or (p / "tokenizer.json").exists()
            has_gguf = any(x.suffix.lower() == ".gguf" for x in p.glob("*.gguf"))
            if has_cfg or has_gguf:
                out.append(p.name)
    return sorted(set(out))


def _discover_llm_models_recursive(root: Path) -> list[str]:
    """
    Descobre modelos baixados dentro de uma pasta, mesmo em subpastas.

    Heurísticas:
    - diretório contendo `config.json` e/ou `tokenizer.json`
    - diretório contendo arquivos `.safetensors` ou `.gguf`

    Retorna identificadores no formato de caminho relativo (com '/').
    Ex.: "Qwen-VL/Qwen3-VL-4B-Instruct"
    """
    root = Path(root)
    if not root.exists():
        return []

    found_dirs: set[Path] = set()
    found_gguf_files: set[Path] = set()

    # 1) arquivos típicos de transformers
    for cfg in root.rglob("config.json"):
        if cfg.is_file():
            found_dirs.add(cfg.parent)
    for tok in root.rglob("tokenizer.json"):
        if tok.is_file():
            found_dirs.add(tok.parent)

    # 2) pesos
    for w in root.rglob("*.safetensors"):
        if w.is_file():
            found_dirs.add(w.parent)
    for w in root.rglob("*.gguf"):
        if w.is_file():
            found_gguf_files.add(w)

    # normaliza: se tiver subdir tipo "snapshots/<hash>", subimos para um nível mais semântico
    normalized: set[Path] = set()
    for d in found_dirs:
        parts = {p.lower() for p in d.parts}
        if "snapshots" in parts:
            # .../models--ORG--NAME/snapshots/<hash> -> .../models--ORG--NAME
            try:
                idx = [p.lower() for p in d.parts].index("snapshots")
                normalized.add(Path(*d.parts[:idx]))
                continue
            except Exception:
                pass
        normalized.add(d)

    labels: set[str] = set()
    for d in normalized:
        try:
            rel = d.relative_to(root)
        except Exception:
            rel = d
        label = str(rel).replace("\\", "/").strip("/")
        if not label:
            continue
        labels.add(label)

    # inclui arquivos gguf como seleção direta
    for f in found_gguf_files:
        try:
            relf = f.relative_to(root)
        except Exception:
            relf = f
        label = str(relf).replace("\\", "/").strip("/")
        if label:
            labels.add(label)

    return sorted(labels)


def _save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _save_pdf(path: Path, text: str, title: str = "Resposta") -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    x = 2 * cm
    y = height - 2 * cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 1 * cm
    c.setFont("Helvetica", 11)

    # Quebra simples por linhas
    max_chars = 105
    for paragraph in (text or "").splitlines() or [""]:
        s = paragraph
        while len(s) > max_chars:
            line = s[:max_chars]
            s = s[max_chars:]
            if y < 2 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 11)
            c.drawString(x, y, line)
            y -= 0.55 * cm
        if y < 2 * cm:
            c.showPage()
            y = height - 2 * cm
            c.setFont("Helvetica", 11)
        c.drawString(x, y, s)
        y -= 0.65 * cm

    c.save()


def _resolve_hf_selection_to_path(root: Path, selection: str) -> Optional[Path]:
    """
    Se a seleção apontar para um .gguf, tenta resolver para um caminho real.
    Aceita:
    - caminho absoluto para .gguf
    - caminho relativo dentro de root
    - apenas nome do arquivo (procura recursivamente)
    """
    s = (selection or "").strip()
    if not s:
        return None
    p = Path(s)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return p

    candidate = root / s
    if candidate.is_file() and candidate.suffix.lower() == ".gguf":
        return candidate

    # tenta achar pelo nome em subpastas
    if s.lower().endswith(".gguf"):
        for f in root.rglob(Path(s).name):
            if f.is_file() and f.suffix.lower() == ".gguf":
                return f
    return None


class AskWorker(QThread):
    done = Signal(str, list)
    failed = Signal(str)

    def __init__(
        self,
        question: str,
        system_prompt: str,
        provider: str,
        ollama_model: str,
        ollama_base_url: str,
        hf_model: str,
        hf_cache_dir: str,
        hf_device: int,
        hf_max_new_tokens: int,
        api_cfg: Optional[ApiConfig],
        top_k: int,
    ) -> None:
        super().__init__()
        self.question = question
        self.system_prompt = system_prompt
        self.provider = provider
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.hf_model = hf_model
        self.hf_cache_dir = hf_cache_dir
        self.hf_device = hf_device
        self.hf_max_new_tokens = hf_max_new_tokens
        self.api_cfg = api_cfg
        self.top_k = top_k

    def _build_llm_runtime(self):
        provider = (self.provider or "").lower().strip()
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                temperature=0.2,
            )

        if provider == "hf":
            # Se o "modelo" selecionado for um GGUF, usa llama.cpp (LlamaCpp).
            root = Path(self.hf_cache_dir)
            gguf_path = _resolve_hf_selection_to_path(root, self.hf_model)
            if gguf_path:
                from langchain_community.llms import LlamaCpp

                # Valores default razoáveis; pode evoluir para UI/env depois.
                return LlamaCpp(
                    model_path=str(gguf_path),
                    n_ctx=4096,
                    temperature=0.2,
                    max_tokens=self.hf_max_new_tokens,
                    verbose=False,
                )

            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            from langchain_huggingface import HuggingFacePipeline

            cache_dir = Path(self.hf_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(self.hf_model, cache_dir=str(cache_dir))
            model = AutoModelForCausalLM.from_pretrained(self.hf_model, cache_dir=str(cache_dir))
            gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=self.hf_device,
                max_new_tokens=self.hf_max_new_tokens,
                do_sample=True,
                temperature=0.2,
            )
            return HuggingFacePipeline(pipeline=gen)

        if provider == "api":
            if not self.api_cfg:
                raise ValueError("Configuração de API não definida.")
            api_key_map = {
                "openai": self.api_cfg.openai_api_key,
                "mistral": self.api_cfg.mistral_api_key,
                "google": self.api_cfg.google_api_key,
            }
            return build_sdk_llm(
                self.api_cfg.provider,
                api_key=api_key_map.get(self.api_cfg.provider, ""),
                model=self.api_cfg.model,
            )

        raise ValueError("Provider inválido. Use ollama/hf/api.")

    def run(self) -> None:
        try:
            # override temporário para top_k
            settings = get_settings()
            object.__setattr__(settings, "top_k", self.top_k)  # type: ignore[attr-defined]
            llm = self._build_llm_runtime()

            q = self.question.strip()
            if not q:
                raise ValueError("Pergunta vazia.")

            res = answer_with_sources(
                settings, llm, q, extra_instructions=(self.system_prompt or "").strip()
            )
            out = res.get("result") if isinstance(res, dict) else str(res)
            src_docs = res.get("source_documents", []) if isinstance(res, dict) else []
            sources = []
            for d in src_docs:
                src = d.metadata.get("source")
                page = d.metadata.get("page")
                snippet = (d.page_content or "").strip().replace("\n", " ")
                if len(snippet) > 260:
                    snippet = snippet[:260] + "…"
                sources.append({"source": src, "page": page, "snippet": snippet})
            self.done.emit(str(out), sources)
        except Exception as e:
            logging.getLogger(__name__).exception("Falha no AskWorker")
            self.failed.emit(str(e))


class IngestWorker(QThread):
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, docs_dir: str) -> None:
        super().__init__()
        self.docs_dir = docs_dir

    def run(self) -> None:
        try:
            s = get_settings()
            docs_path = Path(self.docs_dir)
            docs = load_documents(docs_path)
            n_docs, n_chunks = index_documents(s, docs)

            from collections import Counter

            c = Counter(d.metadata.get("source") for d in docs)
            breakdown = "\n".join(f"- {k}: {v}" for k, v in c.items())
            msg = (
                f"Índice recriado (Chroma zerado antes de indexar). "
                f"docs={n_docs}, chunks={n_chunks}\n\nArquivos carregados:\n{breakdown}"
            )
            self.done.emit(msg)
        except Exception as e:
            logging.getLogger(__name__).exception("Falha no IngestWorker")
            self.failed.emit(str(e))


@dataclass
class ChatEntry:
    question: str
    answer: str
    sources: list


@dataclass
class SessionQARow:
    at: str
    provider: str
    question: str
    answer: str
    sources: list


@dataclass
class ChatPane:
    provider_key: str
    te_question: QTextEdit
    te_answer: QTextEdit
    te_sources: QTextEdit
    lst_history: QListWidget
    btn_ask: QPushButton
    btn_save_txt: QPushButton
    btn_save_pdf: QPushButton


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RAG Local — perguntas com seus documentos")
        self.resize(1100, 700)

        self.settings = QSettings("rag_local", "rag_gui")
        self.ollama_process: Optional[QProcess] = None
        self.worker: Optional[AskWorker] = None
        self.ingest_worker: Optional[IngestWorker] = None
        self.chat_panes: dict[str, ChatPane] = {}
        self.chat_history: dict[str, list[ChatEntry]] = {"ollama": [], "hf": [], "api": []}
        self.session_log: list[SessionQARow] = []
        self._pending_provider_for_answer: Optional[str] = None

        self._build_ui()
        self._refresh_embed_info()
        self._load_state()
        self._refresh_models()

    def _build_ui(self) -> None:
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        act_refresh = QAction("Atualizar lista de modelos", self)
        act_refresh.triggered.connect(self._refresh_models)
        toolbar.addAction(act_refresh)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        lbl_flow = QLabel(
            "Fluxo rápido: configure o modelo de IA acima → na seção “Base de documentos” escolha a pasta dos "
            "arquivos e clique “Indexar” → digite sua pergunta na aba Chat e envie."
        )
        lbl_flow.setWordWrap(True)
        lbl_flow.setStyleSheet("padding: 4px 0 8px 0; color: palette(mid);")
        layout.addWidget(lbl_flow)

        # Painel principal
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        gb_provider = QGroupBox("Modelo de linguagem (gera a resposta)")
        vprov = QVBoxLayout(gb_provider)
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(lambda _i: self._refresh_models())
        vprov.addWidget(self.tabs)

        # --- Aba Ollama ---
        tab_ollama = QWidget()
        tab_ollama_layout = QVBoxLayout(tab_ollama)

        gb_ol_cfg = QGroupBox("Configuração (Ollama)")
        f_ol = QFormLayout(gb_ol_cfg)
        self.le_ollama_url = QLineEdit()
        self.le_ollama_url.setPlaceholderText("http://localhost:11434")
        f_ol.addRow("Ollama URL", self.le_ollama_url)

        self.cb_ollama_model = QComboBox()
        f_ol.addRow("Modelo", self.cb_ollama_model)

        self.chk_ollama_auto = QCheckBox("Iniciar `ollama serve` automaticamente ao perguntar")
        f_ol.addRow("", self.chk_ollama_auto)

        self.btn_ollama_start = QPushButton("Iniciar Ollama (serve) — abrir Prompt")
        self.btn_ollama_stop = QPushButton("Parar Ollama (serve)")
        self.btn_ollama_start.clicked.connect(self._start_ollama)
        self.btn_ollama_stop.clicked.connect(self._stop_ollama)
        row = QHBoxLayout()
        row.addWidget(self.btn_ollama_start)
        row.addWidget(self.btn_ollama_stop)
        f_ol.addRow("Controle", row)

        tab_ollama_layout.addWidget(gb_ol_cfg)
        tab_ollama_layout.addWidget(self._build_chat_block("ollama"))
        tab_ollama_layout.addStretch(1)
        self.tabs.addTab(tab_ollama, "Ollama")

        # --- Aba Hugging Face ---
        tab_hf = QWidget()
        tab_hf_layout = QVBoxLayout(tab_hf)

        gb_hf_cfg = QGroupBox("Configuração (Hugging Face / GGUF)")
        f_hf = QFormLayout(gb_hf_cfg)
        self.le_hf_cache = QLineEdit()
        self.btn_hf_pick = QPushButton("Selecionar pasta…")
        self.btn_hf_pick.clicked.connect(self._pick_hf_dir)
        row2 = QHBoxLayout()
        row2.addWidget(self.le_hf_cache)
        row2.addWidget(self.btn_hf_pick)
        f_hf.addRow("Pasta LLM", row2)

        self.chk_hf_simple = QCheckBox("Procurar recursivamente em subpastas (recomendado)")
        self.chk_hf_simple.setChecked(True)
        self.chk_hf_simple.stateChanged.connect(lambda _x: self._refresh_models())
        f_hf.addRow("", self.chk_hf_simple)

        self.cb_hf_model = QComboBox()
        self.cb_hf_model.setEditable(True)
        f_hf.addRow("Modelo", self.cb_hf_model)

        self.sp_hf_device = QSpinBox()
        self.sp_hf_device.setMinimum(-1)
        self.sp_hf_device.setMaximum(16)
        f_hf.addRow("Device", self.sp_hf_device)

        self.sp_hf_max = QSpinBox()
        self.sp_hf_max.setMinimum(64)
        self.sp_hf_max.setMaximum(8192)
        f_hf.addRow("Max tokens", self.sp_hf_max)

        tab_hf_layout.addWidget(gb_hf_cfg)
        tab_hf_layout.addWidget(self._build_chat_block("hf"))
        tab_hf_layout.addStretch(1)
        self.tabs.addTab(tab_hf, "Hugging Face")

        # --- Aba API ---
        tab_api = QWidget()
        tab_api_layout = QVBoxLayout(tab_api)

        gb_api_cfg = QGroupBox("Configuração (API)")
        f_api = QFormLayout(gb_api_cfg)
        self.cb_api_provider = QComboBox()
        self.cb_api_provider.addItems(["openai", "mistral", "google"])
        self.cb_api_provider.currentIndexChanged.connect(lambda _x: self._refresh_models())
        f_api.addRow("Provider SDK", self.cb_api_provider)

        self.le_openai_key = QLineEdit()
        self.le_openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        f_api.addRow("OPENAI_API_KEY", self.le_openai_key)

        self.le_mistral_key = QLineEdit()
        self.le_mistral_key.setEchoMode(QLineEdit.EchoMode.Password)
        f_api.addRow("MISTRAL_API_KEY", self.le_mistral_key)

        self.le_google_key = QLineEdit()
        self.le_google_key.setEchoMode(QLineEdit.EchoMode.Password)
        f_api.addRow("GOOGLE_API_KEY", self.le_google_key)

        self.cb_api_model = QComboBox()
        self.cb_api_model.setEditable(True)
        f_api.addRow("Modelo", self.cb_api_model)

        self.btn_api_list = QPushButton("Listar modelos")
        self.btn_api_list.clicked.connect(self._refresh_models)
        f_api.addRow("", self.btn_api_list)

        tab_api_layout.addWidget(gb_api_cfg)
        tab_api_layout.addWidget(self._build_chat_block("api"))
        tab_api_layout.addStretch(1)
        self.tabs.addTab(tab_api, "API")

        # RAG
        gb_rag = QGroupBox("Recuperação (busca nos documentos)")
        form2 = QFormLayout(gb_rag)
        self.sp_topk = QSpinBox()
        self.sp_topk.setMinimum(1)
        self.sp_topk.setMaximum(20)
        self.sp_topk.setToolTip("Quantos trechos recuperar antes de responder (maior = mais contexto, mais texto).")
        form2.addRow("Trechos a recuperar (top_k)", self.sp_topk)

        self.lbl_embed_model = QLabel("")
        self.lbl_embed_model.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_embed_model.setWordWrap(True)
        form2.addRow("Modelo de embeddings", self.lbl_embed_model)

        self.lbl_chroma_path = QLabel("")
        self.lbl_chroma_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form2.addRow("Pasta do índice (Chroma)", self.lbl_chroma_path)

        # Indexação
        gb_ingest = QGroupBox("Base de documentos (PDF, TXT ou MD)")
        form_ing = QFormLayout(gb_ingest)
        self.le_docs_dir = QLineEdit()
        self.le_docs_dir.setPlaceholderText("Pasta com os arquivos a consultar…")
        self.btn_docs_pick = QPushButton("Selecionar pasta…")
        self.btn_docs_pick.clicked.connect(self._pick_docs_dir)
        row_docs = QHBoxLayout()
        row_docs.addWidget(self.le_docs_dir)
        row_docs.addWidget(self.btn_docs_pick)
        form_ing.addRow("Pasta dos documentos", row_docs)

        hint_ix = QLabel(
            "Ao indexar, o sistema apaga o banco vetorial anterior e monta um índice novo com os "
            "arquivos encontrados nesta pasta (subpastas inclusas)."
        )
        hint_ix.setWordWrap(True)
        hint_ix.setStyleSheet("color: palette(mid);")
        form_ing.addRow("", hint_ix)

        self.btn_ingest = QPushButton("Indexar documentos")
        self.btn_ingest.clicked.connect(self._ingest)
        form_ing.addRow("", self.btn_ingest)

        self.lbl_ingest_status = QLabel("")
        self.lbl_ingest_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form_ing.addRow("Última indexação", self.lbl_ingest_status)

        # notas adicionais (o RAG aplica em rag.py prompt antialucinação fixo; a recuperação usa só a pergunta)
        gb_prompt = QGroupBox("Notas adicionais ao modelo (opcional; além do prompt antialucinação do RAG)")
        v = QVBoxLayout(gb_prompt)
        self.te_system = QTextEdit()
        self.te_system.setPlaceholderText(
            "Ex.: Pedir tabela em markdown, ou reforçar o tom. Não use isto para “colar” o contexto — a busca usa só a pergunta."
        )
        v.addWidget(self.te_system)

        gb_session = QGroupBox("Todas as interações (esta execução)")
        vsess = QVBoxLayout(gb_session)
        self.lbl_session_note = QLabel(
            "Cada pergunta/resposta fica listada abaixo para consulta. Ao fechar a janela, este registo é apagado e não fica em disco."
        )
        self.lbl_session_note.setWordWrap(True)
        self.lbl_session_note.setStyleSheet("color: palette(mid);")
        vsess.addWidget(self.lbl_session_note)
        sess_split = QSplitter(Qt.Orientation.Horizontal)
        self.lst_session = QListWidget()
        self.lst_session.setMinimumWidth(260)
        self.te_session_detail = QTextEdit()
        self.te_session_detail.setReadOnly(True)
        self.te_session_detail.setPlaceholderText("Selecione um item do registo para ver a pergunta, a resposta e as fontes…")
        sess_split.addWidget(self.lst_session)
        sess_split.addWidget(self.te_session_detail)
        sess_split.setStretchFactor(0, 0)
        sess_split.setStretchFactor(1, 1)
        sess_split.setSizes([300, 520])
        vsess.addWidget(sess_split, 1)
        self.lst_session.currentRowChanged.connect(self._on_session_row)

        left_layout.addWidget(gb_provider)
        left_layout.addWidget(gb_rag)
        left_layout.addWidget(gb_ingest)
        left_layout.addWidget(gb_prompt)
        left_layout.addWidget(gb_session, 1)
        left_layout.addStretch(0)

        main_layout.addWidget(left, 1)

    def _build_chat_block(self, provider_key: str) -> QWidget:
        gb = QGroupBox("Chat")
        outer = QVBoxLayout(gb)

        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split, 1)

        # Histórico (painel lateral compacto)
        hist = QWidget()
        hv = QVBoxLayout(hist)
        hv.setContentsMargins(0, 0, 0, 0)
        hv.addWidget(QLabel("Histórico"))
        lst = QListWidget()
        lst.setMinimumWidth(230)
        hv.addWidget(lst, 1)
        btn_clear = QPushButton("Limpar")
        hv.addWidget(btn_clear)

        # Área principal (pergunta + resultados)
        main = QWidget()
        mv = QVBoxLayout(main)
        mv.setContentsMargins(0, 0, 0, 0)

        te_q = QTextEdit()
        te_q.setPlaceholderText("Digite sua pergunta…")
        te_q.setFixedHeight(95)

        btn_row = QHBoxLayout()
        btn_ask = QPushButton("Perguntar")
        btn_ask.setMinimumHeight(34)
        btn_ask.setToolTip("Atalho na caixa da pergunta: Ctrl+Enter")
        btn_save_txt = QPushButton("Salvar TXT")
        btn_save_pdf = QPushButton("Salvar PDF")
        btn_row.addWidget(btn_ask)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_save_txt)
        btn_row.addWidget(btn_save_pdf)

        tabs = QTabWidgetInner()
        te_a = QTextEdit()
        te_a.setReadOnly(True)
        te_a.setPlaceholderText("A resposta aparecerá aqui…")
        te_a.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        te_a.setMinimumHeight(260)

        te_s = QTextEdit()
        te_s.setReadOnly(True)
        te_s.setPlaceholderText("Fontes (trechos recuperados) aparecerão aqui…")
        te_s.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        tabs.addTab(te_a, "Resposta")
        tabs.addTab(te_s, "Fontes")

        mv.addWidget(QLabel("Pergunta"))
        mv.addWidget(te_q)
        mv.addLayout(btn_row)
        mv.addWidget(tabs, 1)

        split.addWidget(hist)
        split.addWidget(main)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([250, 820])

        pane = ChatPane(
            provider_key=provider_key,
            te_question=te_q,
            te_answer=te_a,
            te_sources=te_s,
            lst_history=lst,
            btn_ask=btn_ask,
            btn_save_txt=btn_save_txt,
            btn_save_pdf=btn_save_pdf,
        )
        self.chat_panes[provider_key] = pane

        btn_ask.clicked.connect(lambda: self._ask(provider_key))
        btn_save_txt.clicked.connect(lambda: self._save_response(provider_key, "txt"))
        btn_save_pdf.clicked.connect(lambda: self._save_response(provider_key, "pdf"))
        lst.currentRowChanged.connect(lambda idx: self._load_history_item(provider_key, idx))
        btn_clear.clicked.connect(lambda: self._clear_history(provider_key))

        sc = QShortcut(QKeySequence("Ctrl+Return"), te_q)
        sc.activated.connect(lambda k=provider_key: self._ask(k))

        return gb

    def _load_state(self) -> None:
        s = get_settings()
        tab = self.settings.value("provider_tab", "Ollama")
        tab_map = {"ollama": "Ollama", "hf": "Hugging Face", "api": "API"}
        tab_name = tab_map.get(str(tab).lower(), str(tab))
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == tab_name:
                self.tabs.setCurrentIndex(i)
                break
        self.le_ollama_url.setText(self.settings.value("ollama_url", s.ollama_base_url))
        self.le_hf_cache.setText(self.settings.value("hf_cache", str(s.hf_cache_dir)))
        self.le_docs_dir.setText(self.settings.value("docs_dir", str(s.docs_dir)))
        self.sp_hf_device.setValue(int(self.settings.value("hf_device", s.hf_device)))
        self.sp_hf_max.setValue(int(self.settings.value("hf_max", s.hf_max_new_tokens)))
        self.sp_topk.setValue(int(self.settings.value("topk", s.top_k)))
        self.te_system.setPlainText(self.settings.value("system_prompt", ""))
        self.cb_api_provider.setCurrentText(self.settings.value("api_provider", s.api_provider))
        self.le_openai_key.setText(self.settings.value("openai_api_key", s.openai_api_key))
        self.le_mistral_key.setText(self.settings.value("mistral_api_key", s.mistral_api_key))
        self.le_google_key.setText(self.settings.value("google_api_key", s.google_api_key))
        self.cb_hf_model.setCurrentText(self.settings.value("hf_model", s.hf_model))
        default_api_model = {
            "openai": s.openai_model,
            "mistral": s.mistral_model,
            "google": s.google_model,
        }.get(self.cb_api_provider.currentText().strip().lower(), s.openai_model)
        self.cb_api_model.setCurrentText(self.settings.value("api_model", default_api_model))
        self.chk_ollama_auto.setChecked(self.settings.value("ollama_auto", "false") == "true")
        self.chk_hf_simple.setChecked(self.settings.value("hf_simple", "true") == "true")
        # Histórico de chat: só em memória; não reabre do disco (e é apagado ao fechar a aplicação).
        self._refresh_history_lists()

    def _save_state(self) -> None:
        self.settings.setValue("provider_tab", self.tabs.tabText(self.tabs.currentIndex()))
        self.settings.setValue("ollama_url", self.le_ollama_url.text().strip())
        self.settings.setValue("hf_cache", self.le_hf_cache.text().strip())
        self.settings.setValue("docs_dir", self.le_docs_dir.text().strip())
        self.settings.setValue("hf_device", self.sp_hf_device.value())
        self.settings.setValue("hf_max", self.sp_hf_max.value())
        self.settings.setValue("topk", self.sp_topk.value())
        self.settings.setValue("system_prompt", self.te_system.toPlainText())
        self.settings.setValue("api_provider", self.cb_api_provider.currentText().strip().lower())
        self.settings.setValue("openai_api_key", self.le_openai_key.text())
        self.settings.setValue("mistral_api_key", self.le_mistral_key.text())
        self.settings.setValue("google_api_key", self.le_google_key.text())
        self.settings.setValue("hf_model", self.cb_hf_model.currentText().strip())
        self.settings.setValue("api_model", self.cb_api_model.currentText().strip())
        self.settings.setValue("ollama_auto", "true" if self.chk_ollama_auto.isChecked() else "false")
        self.settings.setValue("hf_simple", "true" if self.chk_hf_simple.isChecked() else "false")

    def _wipe_all_qa_data(self) -> None:
        self.session_log.clear()
        for k in self.chat_history:
            self.chat_history[k] = []
        if hasattr(self, "lst_session"):
            self.lst_session.clear()
        if hasattr(self, "te_session_detail"):
            self.te_session_detail.clear()

    def closeEvent(self, event):  # noqa: N802
        self._wipe_all_qa_data()
        self._save_state()
        try:
            self.settings.remove("chat_history_json")
        except Exception:
            self.settings.setValue("chat_history_json", "")
        super().closeEvent(event)

    def _refresh_embed_info(self) -> None:
        s = get_settings()
        self.lbl_embed_model.setText(
            f"{s.embed_provider} → {s.embed_model}\n"
            "(.env: EMBED_PROVIDER=ollama|huggingface, EMBED_MODEL; padrão Ollama nomic-embed-text:latest)"
        )
        try:
            self.lbl_chroma_path.setText(str(s.chroma_dir.resolve()))
        except OSError:
            self.lbl_chroma_path.setText(str(s.chroma_dir))

    def _pick_hf_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Selecione a pasta de modelos/cache HF")
        if d:
            self.le_hf_cache.setText(d)
            self._refresh_models()

    def _pick_docs_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Selecione a pasta docs (PDF/TXT/MD)")
        if d:
            self.le_docs_dir.setText(d)

    def _refresh_history_lists(self) -> None:
        for key, pane in self.chat_panes.items():
            pane.lst_history.blockSignals(True)
            pane.lst_history.clear()
            for i, entry in enumerate(self.chat_history.get(key, []), 1):
                title = entry.question.strip().replace("\n", " ")
                if len(title) > 60:
                    title = title[:60] + "…"
                pane.lst_history.addItem(QListWidgetItem(f"{i}. {title}"))
            pane.lst_history.blockSignals(False)

    def _clear_history(self, provider_key: str) -> None:
        self.session_log = [r for r in self.session_log if r.provider != provider_key]
        self._refresh_session_list()
        if not self.session_log and hasattr(self, "te_session_detail"):
            self.te_session_detail.clear()
        self.chat_history[provider_key] = []
        self._refresh_history_lists()
        pane = self.chat_panes[provider_key]
        pane.te_question.clear()
        pane.te_answer.clear()
        pane.te_sources.clear()

    def _refresh_session_list(self) -> None:
        if not hasattr(self, "lst_session"):
            return
        self.lst_session.blockSignals(True)
        self.lst_session.clear()
        for i, row in enumerate(self.session_log, 1):
            prev = row.question.strip().replace("\n", " ")
            if len(prev) > 72:
                prev = prev[:72] + "…"
            self.lst_session.addItem(QListWidgetItem(f"{i}. [{row.provider}] {row.at} — {prev}"))
        self.lst_session.blockSignals(False)

    def _on_session_row(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.session_log):
            return
        row = self.session_log[idx]
        src_text = (
            json.dumps(row.sources, ensure_ascii=False, indent=2) if row.sources else "(nenhuma fonte)"
        )
        self.te_session_detail.setPlainText(
            f"Provedor: {row.provider}\nHora: {row.at}\n\n--- Pergunta ---\n{row.question}\n\n"
            f"--- Resposta ---\n{row.answer}\n\n--- Fontes (bruto) ---\n{src_text}"
        )

    def _load_history_item(self, provider_key: str, idx: int) -> None:
        if idx < 0:
            return
        items = self.chat_history.get(provider_key, [])
        if idx >= len(items):
            return
        entry = items[idx]
        pane = self.chat_panes[provider_key]
        pane.te_question.setPlainText(entry.question)
        pane.te_answer.setPlainText(entry.answer)
        pane.te_sources.setPlainText(json.dumps(entry.sources, ensure_ascii=False, indent=2))

    def _refresh_models(self) -> None:
        provider = self.tabs.tabText(self.tabs.currentIndex()).lower()
        if provider.startswith("hugging"):
            provider = "hf"
        if provider.startswith("api"):
            provider = "api"
        if provider.startswith("ollama"):
            provider = "ollama"

        if provider == "ollama":
            url = self.le_ollama_url.text().strip() or "http://localhost:11434"
            models = list_ollama_models(url)
            self.cb_ollama_model.clear()
            self.cb_ollama_model.addItems(models)
            if self.cb_ollama_model.count() == 0:
                self.cb_ollama_model.addItem("(nenhum encontrado)")
            return

        if provider == "hf":
            root = Path(self.le_hf_cache.text().strip() or r"E:\ComfyUI\models\LLM")
            if self.chk_hf_simple.isChecked():
                models = _discover_llm_models_recursive(root)
            else:
                models = _list_hf_models_in_dir(root)
            current = self.cb_hf_model.currentText().strip()
            self.cb_hf_model.clear()
            if models:
                self.cb_hf_model.addItems(models)
            if current:
                self.cb_hf_model.setCurrentText(current)
            return

        if provider == "api":
            api_provider = self.cb_api_provider.currentText().strip().lower() or "openai"
            key_map = {
                "openai": self.le_openai_key.text(),
                "mistral": self.le_mistral_key.text(),
                "google": self.le_google_key.text(),
            }
            key = key_map.get(api_provider, "")
            if not key:
                return
            try:
                if api_provider == "openai":
                    models = list_openai_models(key)
                elif api_provider == "mistral":
                    models = list_mistral_models(key)
                else:
                    models = list_google_models(key)
                current = self.cb_api_model.currentText().strip()
                self.cb_api_model.clear()
                self.cb_api_model.addItems(models)
                if current:
                    self.cb_api_model.setCurrentText(current)
            except Exception as e:
                QMessageBox.warning(self, "API", f"Falha ao listar modelos: {e}")
            return

    def _start_ollama(self) -> None:
        # Abre uma janela separada do Prompt executando o `ollama serve`.
        # Guardamos o PID para permitir "Parar" via taskkill.
        try:
            if getattr(self, "ollama_serve_pid", None):
                QMessageBox.information(self, "Ollama", "Parece que o `ollama serve` já foi iniciado pela GUI.")
                self._refresh_models()
                return

            cmd = ["cmd.exe", "/k", "ollama serve"]
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
            p = subprocess.Popen(cmd, creationflags=creationflags)
            self.ollama_serve_pid = p.pid
            models = []
            try:
                url = self.le_ollama_url.text().strip() or "http://localhost:11434"
                models = list_ollama_models(url)
            except Exception:
                models = []

            # Preenche o combo imediatamente se já conseguimos listar
            if models:
                self.cb_ollama_model.clear()
                self.cb_ollama_model.addItems(models)
            else:
                self.cb_ollama_model.clear()
                self.cb_ollama_model.addItem("(carregando modelos...)")

            # E agenda refresh: o servidor pode demorar um pouco para aceitar conexões
            QTimer.singleShot(1500, self._refresh_models)
            QTimer.singleShot(3500, self._refresh_models)
            QMessageBox.information(
                self,
                "Ollama",
                "`ollama serve` iniciado em uma janela do Prompt.\n\n"
                + ("Modelos encontrados:\n- " + "\n- ".join(models) if models else "Modelos: (nenhum encontrado ainda)"),
            )
        except Exception as e:
            QMessageBox.warning(self, "Ollama", f"Falha ao iniciar `ollama serve`: {e}")

    def _stop_ollama(self) -> None:
        pid = getattr(self, "ollama_serve_pid", None)
        if not pid:
            QMessageBox.information(self, "Ollama", "Nenhum `ollama serve` iniciado pela GUI.")
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, capture_output=True, text=True)
            self.ollama_serve_pid = None
            QMessageBox.information(self, "Ollama", "Processo `ollama serve` finalizado.")
        except Exception as e:
            QMessageBox.warning(self, "Ollama", f"Falha ao parar: {e}")

    def _api_cfg(self) -> Optional[ApiConfig]:
        provider = self.tabs.tabText(self.tabs.currentIndex()).lower()
        if not provider.startswith("api"):
            return None
        api_provider = self.cb_api_provider.currentText().strip().lower() or "openai"
        model = self.cb_api_model.currentText().strip()
        if not model:
            return None
        return ApiConfig(
            provider=api_provider,
            model=model,
            openai_api_key=self.le_openai_key.text(),
            mistral_api_key=self.le_mistral_key.text(),
            google_api_key=self.le_google_key.text(),
        )

    def _ask(self, provider: str) -> None:
        self._save_state()
        provider = (provider or "").lower().strip()

        if provider == "ollama" and self.chk_ollama_auto.isChecked():
            self._start_ollama()

        pane = self.chat_panes[provider]
        question = pane.te_question.toPlainText()
        system_prompt = self.te_system.toPlainText()

        ollama_url = self.le_ollama_url.text().strip() or "http://localhost:11434"
        ollama_model = self.cb_ollama_model.currentText().strip()
        if ollama_model == "(nenhum encontrado)":
            ollama_model = ""

        hf_cache = self.le_hf_cache.text().strip() or r"E:\ComfyUI\models\LLM"
        hf_model = self.cb_hf_model.currentText().strip()

        api_cfg = self._api_cfg()

        pane.btn_ask.setEnabled(False)
        pane.te_answer.setPlainText("Gerando resposta…")
        pane.te_sources.clear()
        self._pending_provider_for_answer = provider

        self.worker = AskWorker(
            question=question,
            system_prompt=system_prompt,
            provider=provider,
            ollama_model=ollama_model,
            ollama_base_url=ollama_url,
            hf_model=hf_model,
            hf_cache_dir=hf_cache,
            hf_device=self.sp_hf_device.value(),
            hf_max_new_tokens=self.sp_hf_max.value(),
            api_cfg=api_cfg,
            top_k=self.sp_topk.value(),
        )
        self.worker.done.connect(self._on_answer)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _ingest(self) -> None:
        self._save_state()
        docs_dir = self.le_docs_dir.text().strip() or "docs"
        self.btn_ingest.setEnabled(False)
        self.lbl_ingest_status.setText("Indexando… (recriando banco vetorial)")

        self.ingest_worker = IngestWorker(docs_dir=docs_dir)
        self.ingest_worker.done.connect(self._on_ingest_done)
        self.ingest_worker.failed.connect(self._on_ingest_fail)
        self.ingest_worker.start()

    def _on_ingest_done(self, msg: str) -> None:
        self.btn_ingest.setEnabled(True)
        self.lbl_ingest_status.setText(msg)

    def _on_ingest_fail(self, msg: str) -> None:
        self.btn_ingest.setEnabled(True)
        self.lbl_ingest_status.setText("")
        QMessageBox.critical(self, "Indexação", msg)

    def _on_answer(self, text: str, sources: list) -> None:
        provider = self._pending_provider_for_answer or "ollama"
        pane = self.chat_panes.get(provider)
        if pane:
            pane.btn_ask.setEnabled(True)
            pane.te_answer.setPlainText(text)
            pane.te_sources.setPlainText(json.dumps(sources, ensure_ascii=False, indent=2))

            q = pane.te_question.toPlainText()
            self.chat_history[provider].append(ChatEntry(question=q, answer=text, sources=sources))
            self.session_log.append(
                SessionQARow(
                    at=datetime.now().strftime("%H:%M:%S"),
                    provider=provider,
                    question=q,
                    answer=text,
                    sources=sources,
                )
            )
            self._refresh_history_lists()
            self._refresh_session_list()
            if self.lst_session.count() > 0:
                self.lst_session.setCurrentRow(self.lst_session.count() - 1)
            pane.lst_history.setCurrentRow(len(self.chat_history[provider]) - 1)
        self._pending_provider_for_answer = None

    def _on_fail(self, msg: str) -> None:
        provider = self._pending_provider_for_answer or "ollama"
        pane = self.chat_panes.get(provider)
        if pane:
            pane.btn_ask.setEnabled(True)
            pane.te_answer.setPlainText("")
        self._pending_provider_for_answer = None
        QMessageBox.critical(self, "Erro", msg)

    def _save_response(self, provider: str, kind: str) -> None:
        pane = self.chat_panes[provider]
        text = pane.te_answer.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Salvar", "Não há resposta para salvar.")
            return

        if kind == "txt":
            path, _ = QFileDialog.getSaveFileName(self, "Salvar TXT", "resposta.txt", "Text (*.txt)")
            if not path:
                return
            _save_text(Path(path), text)
            QMessageBox.information(self, "Salvar", f"Salvo em {path}")
            return

        if kind == "pdf":
            path, _ = QFileDialog.getSaveFileName(self, "Salvar PDF", "resposta.pdf", "PDF (*.pdf)")
            if not path:
                return
            _save_pdf(Path(path), text, title="Resposta do RAG")
            QMessageBox.information(self, "Salvar", f"Salvo em {path}")
            return


def main() -> None:
    _configure_logging()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

