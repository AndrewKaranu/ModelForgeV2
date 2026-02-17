"""
ModelForge V2 - Split-Stack Hybrid Cloud Architecture
Frontend (Cloud): Streamlit Community Cloud - UI, Auth, Synthetic Data, Data Viewing
Backend (Local): Docker + ngrok - Fine-Tuning (Unsloth), Inference
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
import json
import asyncio
import time
import os
import uuid
import io

from supabase import create_client, Client as SupabaseClient

from src.backboard_manager import BackboardManager, run_async
from src.generator import DataGenerator
import httpx

# ==================== SUPABASE CLIENT ====================

@st.cache_resource
def get_supabase_client() -> SupabaseClient:
    """Initialize and cache the Supabase client from Streamlit secrets.
    Uses service_role key because this is a server-side app (key is never
    exposed to the browser) and RLS JWT claims are unavailable."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


# ==================== RAM KEY CACHE ====================
# Persists across Streamlit session refreshes within the same server process.
# Keyed by user email so multiple users stay isolated.

@st.cache_resource
def _get_key_store() -> dict:
    """Returns a process-level dict: { email: { field: value, ... } }"""
    return {}

def _save_to_ram(email: str, field: str, value):
    """Write a key/value into the RAM cache for a user."""
    store = _get_key_store()
    store.setdefault(email, {})[field] = value

def _load_from_ram(email: str) -> dict:
    """Read all cached keys for a user. Returns {} if nothing cached."""
    return _get_key_store().get(email, {})

def _populate_ram_from_profile(profile: dict):
    """Seed the RAM cache from a Supabase profile dict on login."""
    email = profile.get("email", "")
    if not email:
        return
    key_fields = ["backboard_api_key", "hf_token", "hf_username"]
    for f in key_fields:
        val = profile.get(f)
        if val:
            _save_to_ram(email, f, val)


def ensure_user_profile(email: str, display_name: str = None) -> dict:
    """Upsert user profile in Supabase. Returns the profile row."""
    sb = get_supabase_client()
    # Try to find existing profile
    result = sb.table("profiles").select("*").eq("email", email).execute()
    if result.data:
        profile = result.data[0]
        st.session_state.user_profile = profile
        _populate_ram_from_profile(profile)
        return profile

    # Create new profile
    new_profile = {
        "email": email,
        "display_name": display_name or email.split("@")[0],
    }
    result = sb.table("profiles").insert(new_profile).execute()
    profile = result.data[0]
    st.session_state.user_profile = profile
    _populate_ram_from_profile(profile)
    return profile


def save_profile_field(field_name: str, value):
    """Save a single field to the user's Supabase profile AND RAM cache."""
    profile = st.session_state.get("user_profile")
    if not profile:
        return
    sb = get_supabase_client()
    sb.table("profiles").update({
        field_name: value,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", profile["id"]).execute()
    profile[field_name] = value
    # Also persist to RAM cache so it survives browser refresh
    email = profile.get("email", "")
    if email:
        _save_to_ram(email, field_name, value)


def check_backend_connection(user_id: str) -> Optional[str]:
    """Query Supabase for the user's active backend connection.
    Returns the ngrok URL if alive (heartbeat < 60s), else None."""
    sb = get_supabase_client()
    result = sb.table("backend_connections").select("*").eq("user_id", user_id).execute()
    if not result.data:
        return None
    conn = result.data[0]
    heartbeat = datetime.fromisoformat(conn["last_heartbeat"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
    if age < 90:  # generous window for heartbeat (30s interval + buffer)
        return conn["ngrok_url"]
    return None


# ==================== INDUSTRIAL FORGE THEME ====================

def load_forge_theme():
    """Load industrial forge theme CSS"""
    st.markdown("""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    /* Main app styling */
    .stApp {
        background-color: #0b0c0d;
        color: #a0a0a0;
        font-family: 'Rajdhani', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Rajdhani', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}

    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #000000; }
    ::-webkit-scrollbar-thumb { background: #333333; border: 1px solid #1a1a1a; }
    ::-webkit-scrollbar-thumb:hover { background: #f96124; }

    /* Industrial plate styling */
    .industrial-plate {
        background: #141414;
        border: 1px solid #333;
        border-left: 4px solid #f96124;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.5);
        position: relative;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .industrial-plate::before {
        content: "";
        position: absolute;
        top: 0; right: 0;
        width: 20px; height: 20px;
        background:
            linear-gradient(to bottom, #222 1px, transparent 1px),
            linear-gradient(to right, #222 1px, transparent 1px);
        background-size: 4px 4px;
        opacity: 0.3;
    }

    /* Primary color (forge orange) */
    .text-primary { color: #f96124 !important; }
    .bg-primary { background-color: #f96124 !important; }

    /* Button styling */
    .stButton > button {
        background: #1a1a1a;
        border: 1px solid #333;
        color: #888;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.7rem;
        border-radius: 0px;
        transition: all 0.1s;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background: #252525;
        color: #fff;
        border-color: #f96124;
        box-shadow: 0 0 8px rgba(249, 97, 36, 0.2);
    }
    .stButton > button:active { background: #f96124; color: #000; }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #f96124;
        border: 1px solid #f96124;
        color: #000;
    }
    .stButton > button[kind="primary"]:hover {
        background: #ff7b42;
        box-shadow: 0 0 15px rgba(249, 97, 36, 0.4);
    }

    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #080808;
        border: 1px solid #333;
        color: #ccc;
        border-radius: 0px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }
    .stTextInput > div > div > input:focus { border-color: #f96124; color: white; }

    /* Select box styling */
    .stSelectbox > div > div {
        background-color: #080808;
        border: 1px solid #333;
        color: #ccc;
        border-radius: 0px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Pipeline card */
    .pipeline-card {
        background: #111; border: 1px solid #222;
        border-radius: 0px; padding: 1rem;
        margin-bottom: 0.5rem; border-left: 2px solid #333;
    }
    .pipeline-card:hover { border-left: 2px solid #f96124; background: #161616; }

    /* Status badges */
    .status-badge {
        display: inline-block; padding: 0.1rem 0.5rem;
        background: #000; border: 1px solid #333;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.1em;
    }
    .status-online { color: #22c55e; border-color: #22c55e; }
    .status-running { color: #f96124; border-color: #f96124; }
    .status-complete { color: #3b82f6; border-color: #3b82f6; }
    .status-idle { color: #666; border-color: #444; }

    /* Terminal/Log styling */
    .terminal-log {
        background: #050505; border: 1px solid #333;
        padding: 1rem; font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; color: #bbb;
        max-height: 400px; overflow-y: auto;
    }
    .log-time { color: #555; margin-right: 0.5rem; }
    .log-level-info { color: #3b82f6; }
    .log-level-warn { color: #eab308; }
    .log-level-error { color: #ef4444; }
    .log-level-success { color: #22c55e; }

    /* Model card in selector */
    .model-card-compact {
        background: #111; border: 1px solid #222;
        padding: 0.5rem; margin-bottom: 0.25rem;
        transition: all 0.2s; cursor: pointer;
    }
    .model-card-compact:hover { border-color: #f96124; background: #1a1a1a; }

    /* Stat card */
    .stat-card {
        background: #111; border: 1px solid #222;
        padding: 1rem; position: relative;
    }
    .stat-card::before {
        content: ""; position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: #333;
    }
    .stat-value {
        font-size: 2rem; font-weight: 900;
        color: white; line-height: 1; margin-bottom: 0.5rem;
    }
    .stat-label {
        font-size: 0.65rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.15em; color: #8c6b5d;
    }

    /* Progress bar */
    .progress-bar-container {
        background: #0f0907; border: 1px solid #38302a;
        border-radius: 4px; height: 12px;
        overflow: hidden; position: relative;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #7c2d12 0%, #ff4d00 50%, #fbbf24 100%);
        height: 100%;
        box-shadow: 0 0 10px rgba(255, 77, 0, 0.5);
        transition: width 0.3s ease;
    }

    /* Hide sidebar completely */
    [data-testid="stSidebar"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }
    button[kind="header"] { display: none; }

    /* Top nav bar styling */
    .topnav-bar {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.5rem;
        padding: 0.25rem 0;
        border-bottom: 1px solid #222;
        margin-bottom: 1rem;
    }
    .topnav-bar .nav-status {
        color: #666;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== SESSION STATE MANAGEMENT ====================

def init_session_state():
    """Initialize session state variables, restoring from RAM cache > Supabase profile."""
    profile = st.session_state.get("user_profile", {})
    # RAM cache: survives browser refresh within same server process
    # Try profile email first, then st.user.email (available via OAuth on refresh)
    email = profile.get("email", "")
    if not email:
        try:
            email = st.user.email or ""
        except Exception:
            email = ""
    ram = _load_from_ram(email) if email else {}

    if 'view' not in st.session_state:
        st.session_state.view = 'dashboard'

    if 'api_key' not in st.session_state:
        # Priority: RAM cache > secrets > Supabase profile > None
        cached = ram.get("backboard_api_key")
        secret_key = None
        try:
            secret_key = st.secrets.get("BACKBOARD_API_KEY", "") or None
        except Exception:
            pass
        st.session_state.api_key = cached or secret_key or profile.get("backboard_api_key")

    if 'user_logged_in' not in st.session_state:
        st.session_state.user_logged_in = False

    if 'pipelines' not in st.session_state:
        st.session_state.pipelines = []

    if 'active_pipeline' not in st.session_state:
        st.session_state.active_pipeline = None

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None

    # Cache for validated models: { "provider/model": {info_dict} or None }
    if 'validated_models' not in st.session_state:
        st.session_state.validated_models = {}

    # Dataset management
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}

    if 'active_dataset' not in st.session_state:
        st.session_state.active_dataset = None

    # Document cache for RAG add-on
    if 'global_documents' not in st.session_state:
        st.session_state.global_documents = []

    if 'shared_backboard_manager' not in st.session_state:
        st.session_state.shared_backboard_manager = None

    # Fine-tuning state
    if 'finetuning_manager' not in st.session_state:
        st.session_state.finetuning_manager = None

    if 'finetuning_status' not in st.session_state:
        st.session_state.finetuning_status = None

    if 'finetuning_progress' not in st.session_state:
        st.session_state.finetuning_progress = {}

    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = []

    # V2: Backend connection state (auto-discovered from Supabase)
    if 'backend_url' not in st.session_state:
        st.session_state.backend_url = ""

    # HuggingFace (RAM cache > Supabase profile)
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = ram.get("hf_token") or profile.get("hf_token", "") or ""
    if 'hf_username' not in st.session_state:
        st.session_state.hf_username = ram.get("hf_username") or profile.get("hf_username", "") or ""

    # Backend URL (RAM cache so tunnel URL survives refresh)
    if 'backend_url' not in st.session_state or not st.session_state.backend_url:
        st.session_state.backend_url = ram.get("backend_url", "")


# ==================== DATASET MANAGEMENT ====================

@dataclass
class Dataset:
    """Represents a dataset stored in Supabase (metadata in DB, JSONL in Storage)"""
    id: str
    name: str
    mode: int
    mode_name: str
    storage_path: str
    samples: List[Dict]
    created_at: datetime
    thread_id: Optional[str] = None
    assistant_id: Optional[str] = None
    document_ids: List[str] = None

    def __post_init__(self):
        if self.document_ids is None:
            self.document_ids = []

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def add_samples(self, new_samples):
        """Add new samples to the dataset."""
        added_count = 0
        for sample in new_samples:
            if hasattr(sample, 'to_alpaca'):
                data = sample.to_alpaca()
            else:
                data = sample
            instruction = data.get('instruction', '').strip()
            output = data.get('output', '').strip()
            if instruction or output:
                self.samples.append(data)
                added_count += 1
        if added_count < len(new_samples):
            print(f"Warning: Filtered out {len(new_samples) - added_count} empty samples")
        self.save()

    def save(self):
        """Save dataset to Supabase: JSONL to Storage, metadata to DB."""
        sb = get_supabase_client()
        profile = st.session_state.get("user_profile", {})
        user_id = profile.get("id")
        if not user_id:
            return

        # Build JSONL content
        jsonl_content = "\n".join([json.dumps(s, ensure_ascii=False) for s in self.samples])
        jsonl_bytes = jsonl_content.encode("utf-8")

        # Upload to Supabase Storage (upsert)
        try:
            sb.storage.from_("datasets").upload(
                self.storage_path, jsonl_bytes,
                file_options={"content-type": "application/jsonl", "upsert": "true"}
            )
        except Exception as e:
            print(f"Storage upload error: {e}")

        # Upsert metadata in datasets table
        metadata = {
            "id": self.id,
            "user_id": user_id,
            "name": self.name,
            "mode": self.mode,
            "mode_name": self.mode_name,
            "sample_count": len(self.samples),
            "storage_path": self.storage_path,
            "thread_id": str(self.thread_id) if self.thread_id else None,
            "assistant_id": str(self.assistant_id) if self.assistant_id else None,
            "document_ids": json.dumps(self.document_ids),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            sb.table("datasets").upsert(metadata, on_conflict="id").execute()
        except Exception as e:
            print(f"Dataset metadata save error: {e}")

    def delete(self):
        """Delete dataset from Supabase Storage and DB."""
        sb = get_supabase_client()
        try:
            sb.storage.from_("datasets").remove([self.storage_path])
        except Exception as e:
            print(f"Storage delete error: {e}")
        try:
            sb.table("datasets").delete().eq("id", self.id).execute()
        except Exception as e:
            print(f"Dataset DB delete error: {e}")

    @classmethod
    def from_supabase_row(cls, row: dict, samples: List[Dict] = None) -> 'Dataset':
        """Create Dataset from a Supabase datasets table row."""
        created_at = datetime.now()
        if row.get("created_at"):
            try:
                created_at = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            except:
                pass
        doc_ids = row.get("document_ids", [])
        if isinstance(doc_ids, str):
            try:
                doc_ids = json.loads(doc_ids)
            except:
                doc_ids = []
        return cls(
            id=row["id"],
            name=row["name"],
            mode=row.get("mode", 1),
            mode_name=row.get("mode_name", "Imported Dataset"),
            storage_path=row.get("storage_path", ""),
            samples=samples or [],
            created_at=created_at,
            thread_id=row.get("thread_id"),
            assistant_id=row.get("assistant_id"),
            document_ids=doc_ids,
        )


def load_existing_datasets():
    """Load all datasets from Supabase for the current user."""
    profile = st.session_state.get("user_profile", {})
    user_id = profile.get("id")
    if not user_id:
        return

    # Skip if we already loaded this session
    if st.session_state.get("_datasets_loaded"):
        return

    sb = get_supabase_client()
    try:
        result = sb.table("datasets").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    except Exception as e:
        print(f"Error loading datasets from Supabase: {e}")
        return

    for row in result.data:
        dataset_id = row["id"]
        if dataset_id not in st.session_state.datasets:
            # Download JSONL from storage
            samples = []
            try:
                file_bytes = sb.storage.from_("datasets").download(row["storage_path"])
                for line in file_bytes.decode("utf-8").strip().split("\n"):
                    if line.strip():
                        samples.append(json.loads(line))
            except Exception as e:
                print(f"Error downloading dataset {row['name']}: {e}")

            dataset = Dataset.from_supabase_row(row, samples)
            st.session_state.datasets[dataset.id] = dataset

    st.session_state._datasets_loaded = True


# ==================== PIPELINE MANAGEMENT ====================

class Pipeline:
    """Represents a data generation pipeline"""
    def __init__(self, name: str, pipeline_id: str):
        self.id = pipeline_id
        self.name = name
        self.created_at = datetime.now()
        self.status = "idle"
        self.samples_generated = 0
        self.target_samples = 0
        self.model = None
        self.mode = 1
        self.mode_name = "Mode 1: General Chat (Memory-Driven)"
        self.logs = []
        self.dataset_id = None
        self.thread_id = None
        self.assistant_id = None
        self.use_rag = False
        self.rag_documents = []
        self.config = {
            "topic": "", "style": "", "document_ids": [],
            "code_language": "python", "tools": [], "uploaded_files": []
        }

    def to_dict(self):
        return {
            'id': self.id, 'name': self.name,
            'created_at': self.created_at.isoformat(),
            'status': self.status, 'samples_generated': self.samples_generated,
            'target_samples': self.target_samples, 'model': self.model,
            'mode': self.mode, 'mode_name': self.mode_name,
            'dataset_id': self.dataset_id, 'use_rag': self.use_rag
        }


def create_pipeline(name: str, dataset_id: str = None):
    """Create a new pipeline, optionally linked to existing dataset"""
    import uuid
    pipeline_id = str(uuid.uuid4())[:8]
    pipeline = Pipeline(name, pipeline_id)
    if dataset_id and dataset_id in st.session_state.datasets:
        dataset = st.session_state.datasets[dataset_id]
        pipeline.dataset_id = dataset_id
        pipeline.thread_id = dataset.thread_id
        pipeline.assistant_id = dataset.assistant_id
        pipeline.mode = dataset.mode
        pipeline.mode_name = dataset.mode_name
        pipeline.rag_documents = [{'id': doc_id} for doc_id in dataset.document_ids]
    st.session_state.pipelines.append(pipeline)
    st.session_state.active_pipeline = pipeline
    st.session_state.view = 'data_generation'


def add_log(pipeline: Pipeline, level: str, message: str):
    """Add a log entry to the pipeline"""
    pipeline.logs.append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'level': level, 'message': message
    })


# ==================== MODEL VALIDATION ====================

def validate_model(model_id: str) -> dict | None:
    """Validate a model exists via Backboard GET /models/{model_name}.
    Returns the model info dict on success, None if not found.
    Results are cached in session state."""
    if not model_id or '/' not in model_id:
        return None

    # Check cache first
    cache = st.session_state.get('validated_models', {})
    if model_id in cache:
        return cache[model_id]

    api_key = st.session_state.get('api_key')
    if not api_key:
        return None

    # The API expects the full model name (e.g. "openai/gpt-4o")
    url = f"https://app.backboard.io/api/models/{model_id}"
    headers = {"X-API-Key": api_key}

    try:
        resp = httpx.get(url, headers=headers, timeout=10.0)
        if resp.status_code == 200:
            info = resp.json()
            cache[model_id] = info
            st.session_state.validated_models = cache
            return info
        else:
            cache[model_id] = None
            st.session_state.validated_models = cache
            return None
    except Exception as e:
        print(f"Model validation error: {e}")
        return None


# ==================== VIEW: DASHBOARD ====================

def render_dashboard():
    """Render main dashboard with pipelines"""
    load_existing_datasets()

    # Header (render immediately, models load after)
    col1, col2 = st.columns([5, 1])
    with col1:
        api_ok = bool(st.session_state.api_key)
        status_text = "SYSTEM ONLINE" if api_ok else "API KEY REQUIRED"
        status_class = "status-online" if api_ok else "status-idle"
        dot_color = "#22c55e" if api_ok else "#666"
        st.markdown(f"""
        <div style="margin-bottom: 2rem;">
            <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
                SYSTEM DASHBOARD
            </h1>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span class="status-badge {status_class}">
                    <span style="width: 6px; height: 6px; background: {dot_color}; display: inline-block;"></span>
                    {status_text}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("[NEW_PIPELINE]", type="primary", use_container_width=True):
            st.session_state.show_create_pipeline = True
            st.rerun()

    # API key warning
    if not st.session_state.api_key:
        st.warning("No Backboard API Key configured. Go to **Settings** to add one and start generating data.")

    # Stats cards
    st.markdown("### SYSTEM METRICS")
    stat_cols = st.columns(4)

    with stat_cols[0]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Pipelines</div>
            <div class="stat-value">{len(st.session_state.pipelines)}</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_cols[1]:
        running = len([p for p in st.session_state.pipelines if p.status == "running"])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Running</div>
            <div class="stat-value" style="color: #f96124;">{running}</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_cols[2]:
        complete = len([p for p in st.session_state.pipelines if p.status == "complete"])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Complete</div>
            <div class="stat-value" style="color: #22c55e;">{complete}</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_cols[3]:
        total_samples = sum(p.samples_generated for p in st.session_state.pipelines)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Samples</div>
            <div class="stat-value">{total_samples}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline list
    st.markdown("### ACTIVE PROCESSES")

    if not st.session_state.pipelines:
        st.info("[INFO] No pipelines initialized. Create a new pipeline to begin.")
    else:
        for pipeline in st.session_state.pipelines:
            render_pipeline_card(pipeline)

    if st.session_state.get('show_create_pipeline', False):
        render_create_pipeline_modal()


def render_pipeline_card(pipeline: Pipeline):
    """Render a pipeline card"""
    status_class = f"status-{pipeline.status}"
    status_text = pipeline.status.upper()
    progress = 0
    if pipeline.target_samples > 0:
        progress = (pipeline.samples_generated / pipeline.target_samples) * 100

    with st.container():
        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            st.markdown(f"""
            <div class="pipeline-card">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                    <div>
                        <h3 style="color: white; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;">
                            {pipeline.name}
                        </h3>
                        <span class="status-badge {status_class}">{status_text}</span>
                    </div>
                    <span style="color: #666; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;">
                        ID: {pipeline.id}
                    </span>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #888; margin-bottom: 0.25rem;">
                        <span>Progress</span>
                        <span>{pipeline.samples_generated} / {pipeline.target_samples} samples</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: {progress}%;"></div>
                    </div>
                </div>
                <div style="display: flex; gap: 1rem; font-size: 0.7rem; color: #666;">
                    <span>DATE: {pipeline.created_at.strftime('%Y-%m-%d %H:%M')}</span>
                    {f'<span>MODEL: {pipeline.model}</span>' if pipeline.model else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("Open", key=f"open_{pipeline.id}", use_container_width=True):
                st.session_state.active_pipeline = pipeline
                st.session_state.view = 'data_generation'
                st.rerun()
        with col3:
            if st.button("Delete", key=f"delete_{pipeline.id}", use_container_width=True):
                st.session_state.pipelines.remove(pipeline)
                st.rerun()


def render_create_pipeline_modal():
    """Render create pipeline modal"""
    if not st.session_state.api_key:
        st.warning("You need a Backboard API Key to create pipelines. Go to **Settings** to add one.")
        if st.button("Go to Settings", key="go_settings_from_modal"):
            st.session_state.show_create_pipeline = False
            st.session_state.view = 'settings'
            st.rerun()
        return

    with st.container():
        st.markdown("---")
        st.markdown("### [CREATE_NEW_PIPELINE]")
        st.markdown("Name your data generation pipeline and select the generation mode.")

        pipeline_name = st.text_input(
            "Pipeline Name",
            placeholder="e.g., Customer Support Training Data",
            key="new_pipeline_name"
        )

        mode = st.selectbox(
            "Generation Mode",
            [
                "Mode 1: General Chat (Memory-Driven)",
                "Mode 2: Knowledge Injection (RAG)",
                "Mode 3: Code Specialist",
                "Mode 4: Agent / Tool Use",
                "Mode 5: Reasoning (CoT)"
            ],
            key="new_pipeline_mode"
        )

        mode_descriptions = {
            "Mode 1: General Chat (Memory-Driven)": "Generate unique conversational Q&A using memory deduplication",
            "Mode 2: Knowledge Injection (RAG)": "Generate data grounded in uploaded documents",
            "Mode 3: Code Specialist": "Generate code challenges and solutions with specialized models",
            "Mode 4: Agent / Tool Use": "Generate function calling / tool usage training data",
            "Mode 5: Reasoning (CoT)": "Generate reasoning traces with <think> tags"
        }

        st.info(f"[INFO] {mode_descriptions[mode]}")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_create_pipeline = False
                st.rerun()
        with col2:
            if st.button("Create Pipeline", type="primary", use_container_width=True):
                if pipeline_name:
                    mode_num = int(mode.split(":")[0].replace("Mode ", ""))
                    import uuid
                    pipeline_id = str(uuid.uuid4())[:8]
                    pipeline = Pipeline(pipeline_name, pipeline_id)
                    pipeline.mode = mode_num
                    pipeline.mode_name = mode
                    st.session_state.pipelines.append(pipeline)
                    st.session_state.active_pipeline = pipeline
                    st.session_state.show_create_pipeline = False
                    st.session_state.view = 'data_generation'
                    st.rerun()
                else:
                    st.error("Pipeline name required!")

        st.markdown("---")


# ==================== VIEW: DATA GENERATION ====================

def render_data_generation():
    """Render data generation view with chat interface and logs"""
    pipeline = st.session_state.active_pipeline

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div style="margin-bottom: 2rem;">
            <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em; margin: 0.5rem 0;">
                {pipeline.name}
            </h1>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span class="status-badge status-{pipeline.status}">{pipeline.status.upper()}</span>
                <span style="color: #666; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                    ID: {pipeline.id}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("[< DASHBOARD]", use_container_width=True):
            st.session_state.view = 'dashboard'
            st.rerun()

    col_chat, col_logs = st.columns([1, 1])
    with col_chat:
        render_chat_interface(pipeline)
    with col_logs:
        render_processing_logs(pipeline)


def render_chat_interface(pipeline: Pipeline):
    """Render chat-style interface for data generation"""
    st.markdown(f"""
    <div class="industrial-plate">
        <h3 style="color: #d6c0b6; font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
            FORGE OPERATOR
        </h3>
        <div style="color: #8c6b5d; font-size: 0.7rem; margin-bottom: 1rem;">
            {pipeline.mode_name}
        </div>
    """, unsafe_allow_html=True)

    # Model selection
    st.markdown("#### MODEL SELECTION")
    st.markdown("""
    <div style="background: #111; border: 1px solid #333; padding: 0.75rem; margin-bottom: 1rem; font-size: 0.8rem;">
        <div style="color: #f96124; font-weight: bold; margin-bottom: 0.5rem;">Direct Model Input</div>
        <div style="color: #888;">
            Enter any model from <a href="https://app.backboard.io/dashboard/model-library" target="_blank" style="color: #f96124;">Backboard Model Library</a><br>
            Format: <code style="color: #22c55e;">provider/model-name</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_input = st.text_input(
        "Model Name",
        value=pipeline.model if pipeline.model else "openai/gpt-4o",
        key="direct_model_input",
        placeholder="openai/gpt-4o"
    )

    if model_input and '/' in model_input:
        if model_input != pipeline.model:
            pipeline.model = model_input
            add_log(pipeline, "INFO", f"Model set: {model_input}")

        # Validate the model via Backboard API
        model_info = validate_model(model_input)
        if model_info:
            ctx = model_info.get('context_limit', 'N/A')
            mtype = model_info.get('model_type', 'llm')
            tools = model_info.get('supports_tools', False)
            st.markdown(f"""
            <div style="background: #111; border: 1px solid #22c55e; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;">
                <span style="color: #22c55e; font-size: 0.7rem;">VERIFIED</span>
                <span style="color: white; font-weight: 700; margin-left: 0.5rem;">{model_info.get('provider', '')}/{model_info.get('name', '')}</span>
                <span style="color: #666; font-size: 0.7rem; margin-left: 0.75rem;">ctx: {ctx:,} | {mtype}{' | tools' if tools else ''}</span>
            </div>
            """, unsafe_allow_html=True)
        elif model_info is None and st.session_state.get('api_key'):
            st.markdown(f"""
            <div style="background: #111; border: 1px solid #eab308; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;">
                <span style="color: #eab308; font-size: 0.7rem;">UNVERIFIED</span>
                <span style="color: #ccc; font-weight: 700; margin-left: 0.5rem;">{model_input}</span>
                <span style="color: #666; font-size: 0.7rem; margin-left: 0.75rem;">Model not found in Backboard registry</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #111; border: 1px solid #f96124; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;">
                <span style="color: #888; font-size: 0.7rem;">ACTIVE MODEL:</span>
                <span style="color: #f96124; font-weight: 700; margin-left: 0.5rem;">{pipeline.model}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Enter model as `provider/model-name`")

    with st.expander("Common Models (Quick Select)", expanded=False):
        common_models = [
            ("openai/gpt-4o", "GPT-4o - Fast & capable"),
            ("openai/gpt-4o-mini", "GPT-4o Mini - Fast & cheap"),
            ("anthropic/claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet - Best for coding"),
            ("anthropic/claude-3-5-haiku-20241022", "Claude 3.5 Haiku - Fast & cheap"),
            ("google/gemini-2.0-flash-exp", "Gemini 2.0 Flash - Free & fast"),
            ("deepseek/deepseek-chat", "DeepSeek Chat - Great value"),
            ("qwen/qwen-2.5-coder-32b-instruct", "Qwen 2.5 Coder 32B - Code specialist"),
            ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B - Open source"),
        ]
        cols = st.columns(2)
        for idx, (model_id, desc) in enumerate(common_models):
            with cols[idx % 2]:
                if st.button(f"{model_id.split('/')[1][:20]}", key=f"quick_{idx}", use_container_width=True, help=desc):
                    pipeline.model = model_id
                    st.rerun()

    st.markdown("---")

    # Generation parameters - Mode specific
    st.markdown("#### Generation Parameters")

    if pipeline.mode == 1:
        topic = st.text_input("Topic/Domain", placeholder="e.g., Python programming, Customer support",
                              help="What domain should the data cover?", key="gen_topic",
                              value=pipeline.config.get("topic", ""))
        pipeline.config["topic"] = topic
        style = st.selectbox("Generation Style", ["general_qa", "conversation", "instruction_following"],
                             help="What type of conversational style?", key="gen_style")
        pipeline.config["style"] = style
        custom_instructions = st.text_area("Custom Instructions (Optional)",
            placeholder="Add specific instructions for the AI",
            key="custom_instructions_1", value=pipeline.config.get("custom_instructions", ""))
        pipeline.config["custom_instructions"] = custom_instructions

    elif pipeline.mode == 2:
        topic = st.text_input("Topic/Domain", placeholder="e.g., Product documentation, Research papers",
                              key="gen_topic", value=pipeline.config.get("topic", ""))
        pipeline.config["topic"] = topic
        st.markdown("#### [DOCUMENT_UPLOAD]")
        uploaded_files = st.file_uploader("Upload documents for RAG", accept_multiple_files=True,
                                          type=["pdf", "txt", "md", "docx"], key="doc_upload")
        if uploaded_files:
            st.success(f"[READY] {len(uploaded_files)} document(s) uploaded")
            pipeline.config["uploaded_files"] = uploaded_files
        style = st.selectbox("Generation Style", ["qa_from_docs", "summarization"], key="gen_style")
        pipeline.config["style"] = style
        custom_instructions = st.text_area("[CUSTOM_INSTRUCTIONS]",
            placeholder="Add specific instructions",
            key="custom_instructions_2", value=pipeline.config.get("custom_instructions", ""))
        pipeline.config["custom_instructions"] = custom_instructions

    elif pipeline.mode == 3:
        topic = st.text_input("Coding Topic", placeholder="e.g., Binary trees, API design",
                              key="gen_topic", value=pipeline.config.get("topic", ""))
        pipeline.config["topic"] = topic
        code_language = st.selectbox("Programming Language",
            ["python", "javascript", "java", "cpp", "rust", "go", "typescript"], key="code_lang")
        pipeline.config["code_language"] = code_language
        style = st.selectbox("Code Style", ["algorithm", "debugging", "implementation"], key="gen_style")
        pipeline.config["style"] = style
        custom_instructions = st.text_area("[CUSTOM_INSTRUCTIONS]",
            placeholder="Add specific instructions",
            key="custom_instructions_3", value=pipeline.config.get("custom_instructions", ""))
        pipeline.config["custom_instructions"] = custom_instructions
        if not pipeline.model or "qwen" not in pipeline.model.lower():
            st.info("[TIP] Select a Qwen code model for best results")

    elif pipeline.mode == 4:
        topic = st.text_input("Tool Use Scenario", placeholder="e.g., Web search, Calculator",
                              key="gen_topic", value=pipeline.config.get("topic", ""))
        pipeline.config["topic"] = topic
        st.markdown("#### [TOOL_DEFINITIONS]")
        with st.expander("[EXAMPLE_SCHEMAS]"):
            st.code('{\n  "type": "function",\n  "function": {\n    "name": "web_search",\n    "description": "Search the web",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "query": {"type": "string", "description": "Search query"}\n      },\n      "required": ["query"]\n    }\n  }\n}', language="json")
        tools_json = st.text_area("Tool Definitions (JSON array)", height=150,
                                   placeholder='[{"type": "function", "function": {...}}]', key="tools_json")
        if tools_json:
            try:
                pipeline.config["tools"] = json.loads(tools_json)
                st.success(f"[READY] {len(pipeline.config['tools'])} tool(s) configured")
            except json.JSONDecodeError as e:
                st.error(f"[ERROR] Invalid JSON: {e}")
        style = st.selectbox("Agent Style", ["tool_selection", "multi_step"], key="gen_style")
        pipeline.config["style"] = style
        custom_instructions = st.text_area("[CUSTOM_INSTRUCTIONS]",
            placeholder="Add specific instructions",
            key="custom_instructions_4", value=pipeline.config.get("custom_instructions", ""))
        pipeline.config["custom_instructions"] = custom_instructions

    elif pipeline.mode == 5:
        topic = st.text_input("Reasoning Topic", placeholder="e.g., Logic puzzles, Math problems",
                              key="gen_topic", value=pipeline.config.get("topic", ""))
        pipeline.config["topic"] = topic
        style = st.selectbox("Reasoning Style", ["logical_reasoning", "math_reasoning", "analysis"], key="gen_style")
        pipeline.config["style"] = style
        custom_instructions = st.text_area("[CUSTOM_INSTRUCTIONS]",
            placeholder="Add specific instructions",
            key="custom_instructions_5", value=pipeline.config.get("custom_instructions", ""))
        pipeline.config["custom_instructions"] = custom_instructions
        if not pipeline.model or ("deepseek" not in pipeline.model.lower() and "o1" not in pipeline.model.lower()):
            st.info("[TIP] Select DeepSeek R1 or OpenAI O1 for reasoning with <think> tags")

    # RAG Add-on (all modes except Mode 2)
    if pipeline.mode != 2:
        st.markdown("---")
        st.markdown("#### KNOWLEDGE INJECTION (RAG)")
        pipeline.use_rag = st.checkbox("Enable RAG / Knowledge Injection", value=pipeline.use_rag,
            help="Ground generation in uploaded documents", key="enable_rag")
        if pipeline.use_rag:
            st.markdown("""
            <div style="background: #1a1008; border-left: 3px solid #ff6b35; padding: 0.5rem; margin-bottom: 1rem; font-size: 0.85rem; color: #d6c0b6;">
                [DOC] Upload documents to ground the generated data in real content.
            </div>
            """, unsafe_allow_html=True)
            rag_files = st.file_uploader("Upload Reference Documents", accept_multiple_files=True,
                                          type=["pdf", "txt", "md", "docx"], key="rag_addon_docs")
            if rag_files:
                st.success(f"[READY] {len(rag_files)} document(s) ready for knowledge injection")
                pipeline.config["uploaded_files"] = rag_files
                with st.expander("[UPLOADED_DOCS]"):
                    for f in rag_files:
                        st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")

    # Web Search
    st.markdown("---")
    st.markdown("#### WEB SEARCH CONTEXT")
    enable_web_search = st.checkbox("Enable Web Search",
        value=pipeline.config.get("enable_web_search", False),
        help="Search the web for relevant context (uses Perplexity API)", key="enable_web_search")
    pipeline.config["enable_web_search"] = enable_web_search

    if enable_web_search:
        st.markdown("""
        <div style="background: #0a1628; border-left: 3px solid #3b82f6; padding: 0.5rem; margin-bottom: 1rem; font-size: 0.85rem; color: #a0c4ff;">
            Web search will be performed when generation starts.
        </div>
        """, unsafe_allow_html=True)
        web_search_query = st.text_area("Search Query",
            placeholder="Enter your search query",
            key="web_search_query",
            value=pipeline.config.get("web_search_query", pipeline.config.get("topic", "")),
            height=80)
        pipeline.config["web_search_query"] = web_search_query
        if not web_search_query.strip():
            st.warning("Enter a search query to enable web search")
        else:
            st.success(f"Will search: '{web_search_query[:50]}{'...' if len(web_search_query) > 50 else ''}'")

    # Common parameters
    st.markdown("---")
    num_samples = st.number_input("Number of Samples", min_value=1, max_value=10000, value=100, step=10, key="gen_samples")

    # Dataset append option
    st.markdown("#### DATASET_OPTIONS")
    load_existing_datasets()
    dataset_options = ["Create New Dataset"] + [
        f"{ds.name} ({ds.sample_count} samples)"
        for ds in st.session_state.datasets.values()
    ]
    dataset_ids = [None] + list(st.session_state.datasets.keys())
    selected_dataset_idx = st.selectbox("Target Dataset", range(len(dataset_options)),
        format_func=lambda i: dataset_options[i],
        help="Append to existing dataset or create new", key="target_dataset")
    if selected_dataset_idx > 0:
        pipeline.dataset_id = dataset_ids[selected_dataset_idx]
        dataset = st.session_state.datasets[pipeline.dataset_id]
        pipeline.thread_id = dataset.thread_id
        pipeline.assistant_id = dataset.assistant_id
        st.info(f"[NOTE] Will append to '{dataset.name}' (currently {dataset.sample_count} samples).")
    else:
        pipeline.dataset_id = None

    with st.expander("[DEBUG_CONFIG]", expanded=False):
        st.write(f"**Model:** {pipeline.model or 'Not selected'}")
        st.write(f"**Topic:** '{pipeline.config.get('topic', '')}'")
        st.write(f"**Mode:** {pipeline.mode} - {pipeline.mode_name}")
        st.write(f"**RAG Enabled:** {pipeline.use_rag}")
        st.write(f"**Dataset:** {pipeline.dataset_id or 'New'}")

    # Pre-validation
    st.markdown("---")
    ready_to_generate = True
    issues = []
    if not pipeline.model:
        issues.append("[ERR] No model selected")
        ready_to_generate = False
    if not pipeline.config.get("topic"):
        issues.append("[ERR] No topic provided")
        ready_to_generate = False
    if pipeline.mode == 2:
        has_docs = pipeline.config.get("uploaded_files")
        has_web = pipeline.config.get("enable_web_search") and pipeline.config.get("web_search_query", "").strip()
        if not has_docs and not has_web:
            issues.append("[ERR] Upload documents or enable web search for RAG mode")
            ready_to_generate = False
    if pipeline.mode == 4 and not pipeline.config.get("tools"):
        issues.append("[ERR] No tools defined for Agent mode")
        ready_to_generate = False

    if issues:
        st.warning("**Before generating:**\n" + "\n".join(issues))
    else:
        st.success("[READY] Ready to generate!")

    if st.button("[INITIATE_GENERATION]", type="primary", use_container_width=True, key="start_gen_btn", disabled=not ready_to_generate):
        _run_generation(pipeline, num_samples)

    st.markdown("</div>", unsafe_allow_html=True)


def _run_generation(pipeline: Pipeline, num_samples: int):
    """Execute the data generation pipeline"""
    with st.spinner("[PROCESSING] Generating data..."):
        pipeline.status = "running"
        pipeline.target_samples = num_samples
        add_log(pipeline, "INFO", f"Starting {pipeline.mode_name}")
        add_log(pipeline, "INFO", f"Target: {num_samples} samples on '{pipeline.config['topic']}'")
        add_log(pipeline, "INFO", f"Using model: {pipeline.model}")

        existing_thread_id = pipeline.thread_id if pipeline.dataset_id else None
        existing_assistant_id = pipeline.assistant_id if pipeline.dataset_id else None
        if existing_thread_id:
            add_log(pipeline, "INFO", f"Appending with memory continuity (thread: {str(existing_thread_id)[:8]}...)")

        try:
            manager = BackboardManager(api_key=st.session_state.api_key)
            generator = DataGenerator(manager)
            add_log(pipeline, "INFO", "Starting sample generation...")

            model_parts = pipeline.model.split("/")
            llm_provider = model_parts[0] if len(model_parts) > 1 else "openai"
            model_name = model_parts[1] if len(model_parts) > 1 else pipeline.model

            # Web search
            web_search_context = None
            if pipeline.config.get("enable_web_search") and pipeline.config.get("web_search_query"):
                search_query = pipeline.config["web_search_query"]
                add_log(pipeline, "INFO", f"Performing web search: '{search_query[:50]}...'")
                try:
                    search_results = asyncio.run(manager.web_search(search_query))
                    web_search_context = search_results.get("content", "")
                    sources = search_results.get("sources", [])
                    if web_search_context:
                        add_log(pipeline, "INFO", f"Web search completed - {len(sources)} sources found")
                        pipeline.config["web_search_results"] = search_results
                    else:
                        add_log(pipeline, "WARN", "Web search returned no content")
                except Exception as e:
                    add_log(pipeline, "ERROR", f"Web search failed: {str(e)}")

            # RAG documents for non-Mode-2 modes
            document_context = None
            if pipeline.mode != 2 and pipeline.use_rag and pipeline.config.get("uploaded_files"):
                uploaded_files = pipeline.config.get("uploaded_files", [])
                add_log(pipeline, "INFO", f"Processing {len(uploaded_files)} document(s) for RAG...")
                doc_contents = []
                for file in uploaded_files:
                    try:
                        doc_id = asyncio.run(manager.upload_document(
                            file_path=file.name, file_content=file.read(), file_name=file.name))
                        if hasattr(manager, '_document_cache') and doc_id in manager._document_cache:
                            doc_data = manager._document_cache[doc_id]
                            doc_content = doc_data.get('content', '')
                            if doc_content:
                                doc_contents.append(f"=== DOCUMENT: {doc_data.get('name', file.name)} ===\n{doc_content}")
                                add_log(pipeline, "SUCCESS", f"Extracted content from: {file.name}")
                    except Exception as e:
                        add_log(pipeline, "ERROR", f"Failed to process {file.name}: {str(e)}")
                if doc_contents:
                    document_context = "\n\n".join(doc_contents)
                    add_log(pipeline, "INFO", f"Document context ready: {len(document_context)} characters")

            # Combine contexts
            if web_search_context and document_context:
                web_search_context = f"{document_context}\n\n{web_search_context}"
            elif document_context:
                web_search_context = document_context

            # Route to generation method
            custom_instructions = pipeline.config.get("custom_instructions", "")
            common_kwargs = dict(
                num_samples=num_samples,
                topic=pipeline.config["topic"],
                llm_provider=llm_provider, model_name=model_name,
                custom_prompt=custom_instructions if custom_instructions else None,
                existing_thread_id=existing_thread_id,
                existing_assistant_id=existing_assistant_id,
                web_search_context=web_search_context
            )

            samples = []
            if pipeline.mode == 1:
                add_log(pipeline, "INFO", f"Mode 1: Generating {num_samples} chat samples...")
                samples = generator.generate_mode1_samples(
                    style=pipeline.config.get("style", "general_qa"), **common_kwargs)
            elif pipeline.mode == 2:
                document_ids = []
                for file in pipeline.config.get("uploaded_files", []):
                    doc_id = asyncio.run(manager.upload_document(
                        file_path=file.name, file_content=file.read(), file_name=file.name))
                    document_ids.append(doc_id)
                    add_log(pipeline, "SUCCESS", f"Uploaded document: {file.name}")
                pipeline.config["document_ids"] = document_ids
                add_log(pipeline, "INFO", f"Mode 2: Generating {num_samples} RAG samples...")
                samples = generator.generate_mode2_samples(
                    document_ids=document_ids,
                    style=pipeline.config.get("style", "qa_from_docs"), **common_kwargs)
            elif pipeline.mode == 3:
                add_log(pipeline, "INFO", f"Mode 3: Generating {num_samples} code samples...")
                samples = generator.generate_mode3_samples(
                    style=pipeline.config.get("style", "algorithm"),
                    code_language=pipeline.config.get("code_language", "python"), **common_kwargs)
            elif pipeline.mode == 4:
                add_log(pipeline, "INFO", f"Mode 4: Generating {num_samples} agent samples...")
                samples = generator.generate_mode4_samples(
                    tools=pipeline.config.get("tools", []),
                    style=pipeline.config.get("style", "tool_selection"), **common_kwargs)
            elif pipeline.mode == 5:
                add_log(pipeline, "INFO", f"Mode 5: Generating {num_samples} reasoning samples...")
                samples = generator.generate_mode5_samples(
                    style=pipeline.config.get("style", "logical_reasoning"), **common_kwargs)

            pipeline.samples_generated = len(samples)
            pipeline.status = "complete"
            add_log(pipeline, "SUCCESS", f"Generation complete! Generated {len(samples)} samples")

            new_thread_id = generator.last_thread_id
            new_assistant_id = generator.last_assistant_id

            profile = st.session_state.get("user_profile", {})
            user_id = profile.get("id", "local")

            if pipeline.dataset_id:
                dataset = st.session_state.datasets[pipeline.dataset_id]
                dataset.add_samples(samples)
                dataset.save()
                add_log(pipeline, "SUCCESS", f"Appended {len(samples)} samples to '{dataset.name}' (total: {dataset.sample_count})")
            else:
                dataset_id = str(uuid.uuid4())
                storage_path = f"{user_id}/{dataset_id}.jsonl"
                new_dataset = Dataset(
                    id=dataset_id, name=pipeline.name,
                    mode=pipeline.mode, mode_name=pipeline.mode_name,
                    storage_path=storage_path,
                    samples=[s.to_alpaca() for s in samples],
                    created_at=datetime.now(),
                    thread_id=new_thread_id, assistant_id=new_assistant_id)
                st.session_state.datasets[new_dataset.id] = new_dataset
                new_dataset.save()
                add_log(pipeline, "SUCCESS", f"Created new dataset: {new_dataset.name}")

            add_log(pipeline, "SUCCESS", "Dataset saved to cloud storage")
            st.success(f"[SUCCESS] Generated {len(samples)} samples!")
            st.rerun()

        except Exception as e:
            pipeline.status = "error"
            add_log(pipeline, "ERROR", f"Generation failed: {str(e)}")
            st.error(f"[ERROR] {str(e)}")
            import traceback
            add_log(pipeline, "ERROR", traceback.format_exc())


def render_processing_logs(pipeline: Pipeline):
    """Render processing logs terminal"""
    st.markdown("""
    <div class="industrial-plate">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h3 style="color: #d6c0b6; font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">
                SYSTEM LOGS
            </h3>
            <span class="status-badge status-running" style="font-size: 0.65rem;">LIVE</span>
        </div>
    """, unsafe_allow_html=True)

    log_html = '<div class="terminal-log">'
    if not pipeline.logs:
        log_html += '<div style="color: #5c3a2e; font-style: italic;">Waiting for operations...</div>'
    else:
        for log in pipeline.logs[-20:]:
            level_class = f"log-level-{log['level'].lower()}"
            log_html += f"""
            <div class="log-entry">
                <span class="log-time">{log['time']}</span>
                <span class="{level_class}">{log['level']}</span>
                <span>{log['message']}</span>
            </div>
            """
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==================== MEMORY VISUALIZATION ====================

def render_memory_visualization(dataset):
    """Render interactive node-based memory visualization for a dataset"""
    st.markdown("---")
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #f96124;">MEMORY_NETWORK</h3>
        <span style="color: #8c6b5d; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
            [BACKBOARD.IO MEMORY SYSTEM]
        </span>
    </div>
    """, unsafe_allow_html=True)

    if not dataset.assistant_id:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 4px solid #666;">
            <div style="color: #888; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                <span style="color: #eab308;">[WARNING]</span> No assistant ID linked to this dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    api_key = st.session_state.get('api_key') or os.getenv("BACKBOARD_API_KEY")
    if not api_key:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 4px solid #ef4444;">
            <div style="color: #888; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                <span style="color: #ef4444;">[ERROR]</span> Backboard API key not configured.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner("Fetching memories from Backboard..."):
        try:
            mem_manager = BackboardManager(api_key=api_key)
            memories_data = run_async(mem_manager.get_all_memories(dataset.assistant_id))
        except Exception as e:
            st.error(f"[ERROR] Failed to fetch memories: {e}")
            return

    memories = memories_data.get("memories", [])
    total_count = memories_data.get("total_count", len(memories))

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color: #f96124;">{total_count}</div><div class="stat-label">MEMORIES STORED</div></div>', unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color: #22c55e;">{dataset.sample_count}</div><div class="stat-label">SAMPLES GENERATED</div></div>', unsafe_allow_html=True)
    with col_stat3:
        efficiency = (total_count / dataset.sample_count * 100) if dataset.sample_count > 0 else 0
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color: #3b82f6;">{efficiency:.1f}%</div><div class="stat-label">MEMORY DENSITY</div></div>', unsafe_allow_html=True)
    with col_stat4:
        thread_display = str(dataset.thread_id)[:8] if dataset.thread_id else "N/A"
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color: #a855f7; font-size: 1.2rem;">{thread_display}...</div><div class="stat-label">THREAD ID</div></div>', unsafe_allow_html=True)

    if memories:
        display_memories = memories[:50]
        nodes_data = []
        for i, memory in enumerate(display_memories):
            content = str(memory.get("content", ""))
            memory_id = str(memory.get("id", "unknown"))
            score = memory.get("score", 0) or 0
            created_at = str(memory.get("created_at", ""))[:19]
            escaped_content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
            nodes_data.append({"id": i, "memoryId": memory_id[:12], "content": escaped_content[:300],
                               "score": float(score) if score else 0, "createdAt": created_at})
        nodes_json = json.dumps(nodes_data)

        graph_html = f'''
        <div id="memory-graph-container" style="position: relative; width: 100%; height: 500px; background: #0a0a0a; border: 1px solid #333; overflow: hidden;">
            <canvas id="memoryCanvas" style="width: 100%; height: 100%;"></canvas>
            <div id="tooltip" style="display: none; position: absolute; background: #141414; border: 1px solid #f96124; border-left: 4px solid #f96124; padding: 12px; max-width: 350px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #ccc; z-index: 1000; box-shadow: 0 4px 20px rgba(249, 97, 36, 0.3); pointer-events: none;"></div>
            <div style="position: absolute; bottom: 10px; left: 10px; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #666;">[HOVER NODES TO VIEW MEMORY CONTENT]</div>
            <div style="position: absolute; top: 10px; right: 10px; font-family: 'JetBrains Mono', monospace; font-size: 10px;">
                <span style="color: #22c55e;">HIGH</span> <span style="color: #eab308; margin-left: 10px;">MED</span> <span style="color: #f96124; margin-left: 10px;">LOW</span>
            </div>
        </div>
        <script>
        (function() {{
            const nodes = {nodes_json};
            const canvas = document.getElementById('memoryCanvas');
            const container = document.getElementById('memory-graph-container');
            const tooltip = document.getElementById('tooltip');
            const ctx = canvas.getContext('2d');
            function resizeCanvas() {{ canvas.width = container.clientWidth; canvas.height = container.clientHeight; }}
            resizeCanvas(); window.addEventListener('resize', resizeCanvas);
            const nodeRadius = 8; const centerX = canvas.width / 2; const centerY = canvas.height / 2;
            nodes.forEach((node, i) => {{
                const angle = (i / nodes.length) * Math.PI * 2; const radius = 120 + Math.random() * 100;
                node.x = centerX + Math.cos(angle) * radius + (Math.random() - 0.5) * 60;
                node.y = centerY + Math.sin(angle) * radius + (Math.random() - 0.5) * 60;
            }});
            const connections = [];
            for (let i = 0; i < nodes.length; i++) {{
                if (i < nodes.length - 1) connections.push([i, i + 1]);
                for (let j = i + 2; j < Math.min(i + 4, nodes.length); j++) {{ if (Math.random() > 0.5) connections.push([i, j]); }}
            }}
            function getNodeColor(score) {{ if (score >= 0.8) return '#22c55e'; if (score >= 0.5) return '#eab308'; return '#f96124'; }}
            let hoveredNode = null; let animationFrame = 0;
            function draw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height); animationFrame++;
                ctx.strokeStyle = '#1a1a1a'; ctx.lineWidth = 1;
                for (let x = 0; x < canvas.width; x += 30) {{ ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }}
                for (let y = 0; y < canvas.height; y += 30) {{ ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }}
                connections.forEach(([i, j]) => {{
                    const a = nodes[i]; const b = nodes[j]; const g = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
                    g.addColorStop(0, getNodeColor(a.score) + '40'); g.addColorStop(1, getNodeColor(b.score) + '40');
                    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.strokeStyle = g; ctx.lineWidth = 1; ctx.stroke();
                }});
                nodes.forEach((node, i) => {{
                    const isH = hoveredNode === i; const c = getNodeColor(node.score); const r = isH ? nodeRadius * 1.8 : nodeRadius;
                    if (isH) {{ ctx.beginPath(); ctx.arc(node.x, node.y, r + 15, 0, Math.PI * 2); const gg = ctx.createRadialGradient(node.x, node.y, r, node.x, node.y, r + 15); gg.addColorStop(0, c + '60'); gg.addColorStop(1, 'transparent'); ctx.fillStyle = gg; ctx.fill(); }}
                    ctx.beginPath(); ctx.arc(node.x, node.y, r + 2, 0, Math.PI * 2); ctx.strokeStyle = c; ctx.lineWidth = isH ? 2 : 1; ctx.stroke();
                    ctx.beginPath(); ctx.arc(node.x, node.y, r, 0, Math.PI * 2); ctx.fillStyle = isH ? c : '#0a0a0a'; ctx.fill();
                    ctx.beginPath(); ctx.arc(node.x, node.y, 3, 0, Math.PI * 2); ctx.fillStyle = c; ctx.fill();
                    ctx.fillStyle = isH ? '#000' : '#666'; ctx.font = '9px JetBrains Mono, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                    ctx.fillText(String(i + 1).padStart(2, '0'), node.x, node.y);
                }});
                const p = (animationFrame % 100) / 100;
                connections.forEach(([i, j]) => {{ const a = nodes[i]; const b = nodes[j]; ctx.beginPath(); ctx.arc(a.x + (b.x - a.x) * p, a.y + (b.y - a.y) * p, 2, 0, Math.PI * 2); ctx.fillStyle = '#f9612440'; ctx.fill(); }});
                requestAnimationFrame(draw);
            }}
            canvas.addEventListener('mousemove', (e) => {{
                const rect = canvas.getBoundingClientRect(); const sX = canvas.width / rect.width; const sY = canvas.height / rect.height;
                const mX = (e.clientX - rect.left) * sX; const mY = (e.clientY - rect.top) * sY;
                hoveredNode = null;
                for (let i = 0; i < nodes.length; i++) {{ const n = nodes[i]; if (Math.sqrt((mX - n.x) ** 2 + (mY - n.y) ** 2) < nodeRadius + 5) {{ hoveredNode = i; break; }} }}
                if (hoveredNode !== null) {{
                    const n = nodes[hoveredNode]; const c = getNodeColor(n.score);
                    tooltip.innerHTML = '<div style="color: ' + c + '; margin-bottom: 8px; font-size: 12px;">[NODE ' + String(hoveredNode + 1).padStart(3, '0') + ']</div><div style="color: #666; margin-bottom: 6px; font-size: 10px;">ID: ' + n.memoryId + '... | ' + (n.createdAt || 'N/A') + '</div><div style="color: #aaa; line-height: 1.5; font-size: 11px; max-height: 150px; overflow: hidden;">' + n.content + '</div><div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #333;"><span style="color: ' + c + ';">SCORE: ' + n.score.toFixed(2) + '</span></div>';
                    tooltip.style.display = 'block';
                    const tX = e.clientX - container.getBoundingClientRect().left + 15;
                    const tY = e.clientY - container.getBoundingClientRect().top + 15;
                    tooltip.style.left = Math.min(tX, container.clientWidth - tooltip.offsetWidth - 10) + 'px';
                    tooltip.style.top = Math.min(tY, container.clientHeight - tooltip.offsetHeight - 10) + 'px';
                    canvas.style.cursor = 'pointer';
                }} else {{ tooltip.style.display = 'none'; canvas.style.cursor = 'default'; }}
            }});
            canvas.addEventListener('mouseleave', () => {{ hoveredNode = null; tooltip.style.display = 'none'; }});
            draw();
        }})();
        </script>
        '''
        components.html(graph_html, height=550)
        if total_count > 50:
            st.markdown(f'<div style="text-align: center; padding: 0.5rem; color: #666; font-family: \'JetBrains Mono\', monospace; font-size: 0.75rem;">Displaying 50 of {total_count} memories</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 4px solid #666; text-align: center; padding: 2rem;">
            <div style="color: #666; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">[NO MEMORIES FOUND]</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("[CLOSE_MEMORY_VIEW]", use_container_width=True):
        memory_key = f"show_memory_{st.session_state.active_dataset}"
        st.session_state[memory_key] = False
        st.rerun()


# ==================== VIEW: DATA VIEWER ====================

def render_data_viewer():
    """Render the data viewer for browsing and managing datasets"""
    load_existing_datasets()

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.markdown('<h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em;">DATA VIEWER</h1>', unsafe_allow_html=True)
    with col2:
        if st.button("[UPLOAD]", use_container_width=True, key="dv_upload_btn"):
            st.session_state.show_upload_modal = not st.session_state.get("show_upload_modal", False)
    with col3:
        if st.button("[NEW_PIPELINE]", use_container_width=True, key="dv_new_pipeline"):
            st.session_state.show_create_modal = True

    st.markdown("---")

    # ---- Upload Dataset Modal ----
    if st.session_state.get("show_upload_modal", False):
        with st.container():
            st.markdown("""
            <div class="industrial-plate" style="border-left: 4px solid #f96124;">
                <div style="color: #f96124; font-weight: bold;">UPLOAD DATASET</div>
                <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.3rem;">
                    Upload a JSONL file with Alpaca-format samples. Each line must be a JSON object
                    with <code>instruction</code> and <code>output</code> fields (and optional <code>input</code>).
                </div>
            </div>
            """, unsafe_allow_html=True)

            upload_name = st.text_input("Dataset Name", placeholder="e.g., My Custom Dataset", key="upload_ds_name")
            uploaded_file = st.file_uploader(
                "Upload JSONL file",
                type=["jsonl", "json"],
                key="upload_ds_file",
                help='Each line: {"instruction": "...", "output": "...", "input": "..."}'
            )

            if uploaded_file is not None:
                try:
                    raw = uploaded_file.read().decode("utf-8")
                    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
                    samples = []
                    errors = []
                    for i, line in enumerate(lines, 1):
                        try:
                            obj = json.loads(line)
                            if not isinstance(obj, dict):
                                errors.append(f"Line {i}: not a JSON object")
                                continue
                            if not obj.get("instruction") and not obj.get("output"):
                                errors.append(f"Line {i}: missing both 'instruction' and 'output'")
                                continue
                            samples.append({
                                "instruction": obj.get("instruction", ""),
                                "input": obj.get("input", ""),
                                "output": obj.get("output", ""),
                            })
                        except json.JSONDecodeError as e:
                            errors.append(f"Line {i}: invalid JSON — {e}")

                    if errors:
                        with st.expander(f"Validation Warnings ({len(errors)})", expanded=False):
                            for err in errors[:20]:
                                st.caption(err)
                            if len(errors) > 20:
                                st.caption(f"... and {len(errors) - 20} more")

                    if samples:
                        st.success(f"Parsed **{len(samples)}** valid samples from {len(lines)} lines.")

                        if st.button("[UPLOAD TO MODELFORGE]", type="primary", use_container_width=True,
                                     key="upload_ds_confirm", disabled=not upload_name.strip()):
                            profile = st.session_state.get("user_profile", {})
                            user_id = profile.get("id")
                            if not user_id:
                                st.error("You must be logged in to upload datasets.")
                            else:
                                dataset_id = str(uuid.uuid4())
                                storage_path = f"{user_id}/{dataset_id}.jsonl"
                                ds = Dataset(
                                    id=dataset_id,
                                    name=upload_name.strip(),
                                    mode=0,
                                    mode_name="Uploaded Dataset",
                                    storage_path=storage_path,
                                    samples=samples,
                                    created_at=datetime.now(timezone.utc),
                                )
                                ds.save()
                                st.session_state.datasets[dataset_id] = ds
                                st.session_state.show_upload_modal = False
                                st.session_state._datasets_loaded = False
                                st.success(f"Uploaded **{ds.name}** with {ds.sample_count} samples!")
                                st.rerun()
                    else:
                        st.error("No valid samples found. Check the file format.")

                except Exception as e:
                    st.error(f"Error reading file: {e}")

            with st.expander("Expected Format"):
                st.code('{"instruction": "Explain photosynthesis", "input": "", "output": "Photosynthesis is..."}\n{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}', language="json")

            st.markdown("---")

    col_list, col_viewer = st.columns([1, 2])

    with col_list:
        st.markdown("### LOCAL_DATASETS")
        if not st.session_state.datasets:
            st.info("[INFO] No datasets found. Generate some data first!")
        else:
            for dataset_id, dataset in st.session_state.datasets.items():
                is_active = st.session_state.active_dataset == dataset_id
                st.markdown(f"""
                <div class="industrial-plate" style="margin-bottom: 0.5rem; {'border: 2px solid #ff6b35;' if is_active else ''}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: white;">{dataset.name}</strong>
                            <div style="color: #8c6b5d; font-size: 0.75rem;">{dataset.sample_count} samples - Mode {dataset.mode}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                col_view, col_add = st.columns(2)
                with col_view:
                    if st.button("[VIEW]", key=f"view_{dataset_id}", use_container_width=True):
                        st.session_state.active_dataset = dataset_id
                        st.rerun()
                with col_add:
                    if st.button("[ADD]", key=f"add_{dataset_id}", use_container_width=True):
                        create_pipeline(f"Add to {dataset.name}", dataset_id=dataset_id)
                        st.rerun()

    with col_viewer:
        if st.session_state.active_dataset and st.session_state.active_dataset in st.session_state.datasets:
            dataset = st.session_state.datasets[st.session_state.active_dataset]
            st.markdown(f"### DATASET: {dataset.name}")
            st.markdown(f"**{dataset.sample_count} samples** | Mode: {dataset.mode_name}")
            st.markdown(f"Storage: `{dataset.storage_path}`")

            if dataset.samples:
                st.markdown("---")
                samples_per_page = 10
                total_pages = (len(dataset.samples) + samples_per_page - 1) // samples_per_page
                if 'sample_page' not in st.session_state:
                    st.session_state.sample_page = 0
                col_prev, col_page, col_next = st.columns([1, 2, 1])
                with col_prev:
                    if st.button("[PREV]", disabled=st.session_state.sample_page == 0):
                        st.session_state.sample_page -= 1
                        st.rerun()
                with col_page:
                    st.markdown(f"<div style='text-align: center; color: #d6c0b6;'>Page {st.session_state.sample_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
                with col_next:
                    if st.button("[NEXT]", disabled=st.session_state.sample_page >= total_pages - 1):
                        st.session_state.sample_page += 1
                        st.rerun()

                start_idx = st.session_state.sample_page * samples_per_page
                end_idx = min(start_idx + samples_per_page, len(dataset.samples))
                for i, sample in enumerate(dataset.samples[start_idx:end_idx], start=start_idx + 1):
                    with st.expander(f"Sample {i}: {sample.get('instruction', 'No instruction')[:60]}..."):
                        st.markdown("**Instruction:**")
                        st.text(sample.get('instruction', ''))
                        if sample.get('input'):
                            st.markdown("**Input:**")
                            st.text(sample.get('input', ''))
                        st.markdown("**Output:**")
                        st.text(sample.get('output', ''))

                st.markdown("---")
                st.markdown("### EXPORT_OPTIONS")
                col_dl, col_mem, col_del = st.columns(3)
                with col_dl:
                    jsonl_content = "\n".join([json.dumps(s, ensure_ascii=False) for s in dataset.samples])
                    st.download_button("[DOWNLOAD_JSONL]", data=jsonl_content,
                        file_name=f"{dataset.name.replace(' ', '_')}.jsonl",
                        mime="application/json", use_container_width=True)
                with col_mem:
                    memory_key = f"show_memory_{st.session_state.active_dataset}"
                    if st.button("[VIEW_MEMORY]", use_container_width=True):
                        st.session_state[memory_key] = not st.session_state.get(memory_key, False)
                        st.rerun()
                with col_del:
                    if st.button("[DELETE_DATASET]", use_container_width=True):
                        try:
                            dataset.delete()
                            del st.session_state.datasets[st.session_state.active_dataset]
                            st.session_state.active_dataset = None
                            st.session_state._datasets_loaded = False
                            st.success("[SUCCESS] Dataset deleted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"[ERROR] Error deleting: {e}")

                memory_key = f"show_memory_{st.session_state.active_dataset}"
                if st.session_state.get(memory_key, False):
                    render_memory_visualization(dataset)
        else:
            st.info("[INFO] Select a dataset to view its contents")

    if st.session_state.get('show_create_modal', False):
        render_create_pipeline_modal()


# ==================== COLAB NOTEBOOK GENERATOR ====================

# Base models list (mirrors UNSLOTH_MODELS from finetuning_manager for offline use)
COLAB_BASE_MODELS = [
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Qwen3-0.6B-bnb-4bit",
    "unsloth/Qwen3-1.7B-bnb-4bit",
    "unsloth/Qwen3-4B-bnb-4bit",
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-2b-it-bnb-4bit",
    "unsloth/Phi-4-bnb-4bit",
    "unsloth/SmolLM2-360M-Instruct-bnb-4bit",
    "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit",
    "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    "unsloth/Qwen3-8B-bnb-4bit",
    "unsloth/Qwen3-14B-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
    "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
]

SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1laHp6Ymhpa2VobGdlc294Z3FoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzEyOTI2MDIsImV4cCI6MjA4Njg2ODYwMn0.QMFn8Nt3bqWMw82Q7vqxQJPWDOuVxo6jqaqkFMAgqRM"
SUPABASE_PUBLIC_URL = "https://mehzzbhikehlgesoxgqh.supabase.co"


def _get_github_token() -> Optional[str]:
    """Get GitHub token from Streamlit secrets (for Gist creation)."""
    try:
        return st.secrets.get("GITHUB_TOKEN", None)
    except Exception:
        return None


def upload_notebook_to_gist(notebook_json: str, filename: str) -> Optional[str]:
    """Upload a notebook to GitHub Gist and return the Colab URL.

    Returns the Colab URL on success, or None on failure.
    """
    token = _get_github_token()
    if not token:
        return None

    payload = {
        "description": f"ModelForge Training Notebook — {filename}",
        "public": False,
        "files": {
            filename: {
                "content": notebook_json,
            }
        },
    }

    try:
        resp = requests.post(
            "https://api.github.com/gists",
            json=payload,
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=30,
        )
        if resp.status_code == 201:
            data = resp.json()
            gist_id = data["id"]
            owner = data["owner"]["login"]
            colab_url = f"https://colab.research.google.com/gist/{owner}/{gist_id}/{filename}"
            return colab_url
        else:
            print(f"[WARN] Gist creation failed: {resp.status_code} {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"[WARN] Gist upload error: {e}")
        return None


def _nb_cell(cell_type, source, **kwargs):
    """Helper to create a notebook cell dict."""
    cell = {
        "cell_type": cell_type,
        "metadata": kwargs.get("metadata", {}),
        "source": source if isinstance(source, list) else source.split("\n"),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def generate_colab_notebook(config: dict) -> str:
    """Generate a complete .ipynb JSON string for Google Colab training.

    config keys:
        model_name, dataset_storage_path, dataset_name, num_train_epochs,
        per_device_train_batch_size, learning_rate, lora_r, lora_alpha,
        max_seq_length, load_in_4bit, custom_model_name
    """
    model_name = config["model_name"]
    dataset_path = config["dataset_storage_path"]
    dataset_name = config.get("dataset_name", "dataset")
    epochs = config.get("num_train_epochs", 3)
    batch_size = config.get("per_device_train_batch_size", 2)
    lr = config.get("learning_rate", 2e-4)
    lora_r = config.get("lora_r", 32)
    lora_alpha = config.get("lora_alpha", 32)
    max_seq = config.get("max_seq_length", 1024)
    load_4bit = config.get("load_in_4bit", True)
    custom_name = config.get("custom_model_name", "").strip()
    output_name = custom_name if custom_name else model_name.split("/")[-1].replace("-bnb-4bit", "") + "-finetuned"

    cells = []

    # ---- Title ----
    cells.append(_nb_cell("markdown", [
        "# ModelForge — Fine-Tuning Notebook\n",
        "\n",
        f"**Base Model:** `{model_name}`  \n",
        f"**Dataset:** `{dataset_name}`  \n",
        f"**Output:** `{output_name}`  \n",
        "\n",
        "> Generated by [ModelForge](https://github.com/modelforge). ",
        "Run all cells top-to-bottom. A T4 GPU (free tier) is sufficient for most 1-4B models.\n",
    ]))

    # ---- Cell 1: Install deps ----
    cells.append(_nb_cell("markdown", [
        "## 1 — Install Dependencies\n",
        "This installs Unsloth (fast LoRA training) and the Supabase client for dataset download.",
    ]))
    cells.append(_nb_cell("code", [
        "%%capture\n",
        "!pip install unsloth supabase\n",
        "# Fix — ensure latest transformers + trl\n",
        "!pip install --upgrade transformers trl datasets\n",
    ]))

    # ---- Cell 2: Mount Google Drive ----
    cells.append(_nb_cell("markdown", [
        "## 2 — Mount Google Drive\n",
        "The trained model will be saved to your Drive so it persists after runtime disconnects.",
    ]))
    cells.append(_nb_cell("code", [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "import os\n",
        f"SAVE_DIR = '/content/drive/MyDrive/ModelForge/models/{output_name}'\n",
        "os.makedirs(SAVE_DIR, exist_ok=True)\n",
        "print(f'Model will be saved to: {SAVE_DIR}')\n",  # uses f-string inside generated code
    ]))

    # ---- Cell 3: Download dataset from Supabase ----
    cells.append(_nb_cell("markdown", [
        "## 3 — Download Training Dataset\n",
        f"Downloads `{dataset_name}` from ModelForge cloud storage.",
    ]))
    cells.append(_nb_cell("code", [
        "from supabase import create_client\n",
        "import json\n",
        "\n",
        f'SUPABASE_URL = "{SUPABASE_PUBLIC_URL}"\n',
        f'SUPABASE_KEY = "{SUPABASE_ANON_KEY}"\n',
        f'DATASET_PATH = "{dataset_path}"\n',
        "\n",
        "sb = create_client(SUPABASE_URL, SUPABASE_KEY)\n",
        "file_bytes = sb.storage.from_('datasets').download(DATASET_PATH)\n",
        "\n",
        "# Parse JSONL\n",
        "raw_text = file_bytes.decode('utf-8')\n",
        "samples = [json.loads(line) for line in raw_text.strip().splitlines() if line.strip()]\n",
        "print(f'Loaded {len(samples)} training samples')\n",
        "print('Sample:', json.dumps(samples[0], indent=2)[:500])\n",
    ]))

    # ---- Cell 4: Format dataset ----
    cells.append(_nb_cell("markdown", [
        "## 4 — Format Dataset for Training\n",
        "Converts Alpaca-format samples into the chat template the model expects.",
    ]))
    cells.append(_nb_cell("code", [
        "from datasets import Dataset\n",
        "\n",
        "alpaca_prompt = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n",
        "\n",
        "### Instruction:\n",
        "{instruction}\n",
        "\n",
        "### Input:\n",
        "{input}\n",
        "\n",
        "### Response:\n",
        "{output}'''\n",
        "\n",
        "def format_alpaca(sample):\n",
        "    return {'text': alpaca_prompt.format(\n",
        "        instruction=sample.get('instruction', ''),\n",
        "        input=sample.get('input', ''),\n",
        "        output=sample.get('output', ''),\n",
        "    )}\n",
        "\n",
        "dataset = Dataset.from_list(samples).map(format_alpaca)\n",
        "print(f'Dataset ready: {len(dataset)} samples')\n",
        "print(dataset[0]['text'][:400])\n",
    ]))

    # ---- Cell 5: Load model ----
    cells.append(_nb_cell("markdown", [
        "## 5 — Load Base Model with Unsloth\n",
        f"Loading `{model_name}` with {'4-bit quantization (QLoRA)' if load_4bit else 'full precision'}.",
    ]))
    cells.append(_nb_cell("code", [
        "from unsloth import FastLanguageModel\n",
        "\n",
        "model, tokenizer = FastLanguageModel.from_pretrained(\n",
        f'    model_name="{model_name}",\n',
        f"    max_seq_length={max_seq},\n",
        f"    load_in_4bit={load_4bit},\n",
        ")\n",
        "print('Model loaded successfully!')\n",
    ]))

    # ---- Cell 6: Apply LoRA ----
    cells.append(_nb_cell("markdown", [
        "## 6 — Apply LoRA Adapters\n",
        f"LoRA rank={lora_r}, alpha={lora_alpha}. These low-rank adapters enable efficient fine-tuning.",
    ]))
    cells.append(_nb_cell("code", [
        "model = FastLanguageModel.get_peft_model(\n",
        "    model,\n",
        f"    r={lora_r},\n",
        f"    lora_alpha={lora_alpha},\n",
        "    lora_dropout=0,\n",
        "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',\n",
        "                    'gate_proj', 'up_proj', 'down_proj'],\n",
        "    bias='none',\n",
        "    use_gradient_checkpointing='unsloth',\n",
        "    random_state=3407,\n",
        ")\n",
        "model.print_trainable_parameters()\n",
    ]))

    # ---- Cell 7: Train ----
    cells.append(_nb_cell("markdown", [
        "## 7 — Train\n",
        f"Training for **{epochs} epoch(s)**, batch size {batch_size}, learning rate {lr:.0e}.",
    ]))
    cells.append(_nb_cell("code", [
        "from trl import SFTTrainer\n",
        "from transformers import TrainingArguments\n",
        "\n",
        "trainer = SFTTrainer(\n",
        "    model=model,\n",
        "    tokenizer=tokenizer,\n",
        "    train_dataset=dataset,\n",
        "    dataset_text_field='text',\n",
        f"    max_seq_length={max_seq},\n",
        "    dataset_num_proc=2,\n",
        "    packing=False,\n",
        "    args=TrainingArguments(\n",
        f"        per_device_train_batch_size={batch_size},\n",
        "        gradient_accumulation_steps=4,\n",
        "        warmup_steps=5,\n",
        f"        num_train_epochs={epochs},\n",
        f"        learning_rate={lr},\n",
        "        fp16=True,\n",
        "        logging_steps=1,\n",
        "        optim='adamw_8bit',\n",
        "        weight_decay=0.01,\n",
        "        lr_scheduler_type='linear',\n",
        "        seed=3407,\n",
        "        output_dir='outputs',\n",
        "        report_to='none',\n",
        "    ),\n",
        ")\n",
        "\n",
        "print('Starting training...')\n",
        "stats = trainer.train()\n",
        "print(f'Training complete! Loss: {stats.training_loss:.4f}')\n",
    ]))

    # ---- Cell 8: Save to Drive ----
    cells.append(_nb_cell("markdown", [
        "## 8 — Save Model to Google Drive\n",
        "Saves the trained LoRA adapters to your Google Drive for later use.",
    ]))
    cells.append(_nb_cell("code", [
        f"# Save to Google Drive\n",
        "model.save_pretrained(SAVE_DIR)\n",
        "tokenizer.save_pretrained(SAVE_DIR)\n",
        "print(f'Model saved to {SAVE_DIR}')\n",
        "\n",
        "# Also save a local copy\n",
        f"model.save_pretrained('{output_name}')\n",
        f"tokenizer.save_pretrained('{output_name}')\n",
        f"print('Also saved locally to ./{output_name}')\n",
    ]))

    # ---- Cell 9: Optional HF push ----
    cells.append(_nb_cell("markdown", [
        "## 9 — (Optional) Push to Hugging Face Hub\n",
        "Uncomment and fill in your HF username to push the model to Hugging Face.",
    ]))
    cells.append(_nb_cell("code", [
        "# Uncomment below to push to Hugging Face Hub:\n",
        "#\n",
        "# HF_USERNAME = 'your-username'\n",
        f"# model.push_to_hub(f'{{HF_USERNAME}}/{output_name}', token='hf_...')\n",
        f"# tokenizer.push_to_hub(f'{{HF_USERNAME}}/{output_name}', token='hf_...')\n",
        "# print('Pushed to Hugging Face Hub!')\n",
    ]))

    # ---- Cell 10: Inference ----
    cells.append(_nb_cell("markdown", [
        "## 10 — Test Inference\n",
        "Run a quick test to verify the fine-tuned model works.",
    ]))
    cells.append(_nb_cell("code", [
        "# Switch to inference mode (2x faster)\n",
        "FastLanguageModel.for_inference(model)\n",
        "\n",
        "# Test prompt — edit the instruction to try your own\n",
        "test_instruction = 'Summarize the key concepts of machine learning in 3 bullet points.'\n",
        "\n",
        "prompt = alpaca_prompt.format(\n",
        "    instruction=test_instruction,\n",
        "    input='',\n",
        "    output='',  # leave blank for generation\n",
        ")\n",
        "\n",
        "inputs = tokenizer([prompt], return_tensors='pt').to('cuda')\n",
        "outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, use_cache=True)\n",
        "response = tokenizer.batch_decode(outputs)[0]\n",
        "\n",
        "# Extract just the response part\n",
        "if '### Response:' in response:\n",
        "    response = response.split('### Response:')[-1].strip()\n",
        "    # Remove EOS tokens\n",
        "    for tok in ['</s>', '<|endoftext|>', '<|end|>', tokenizer.eos_token or '']:\n",
        "        response = response.replace(tok, '').strip()\n",
        "\n",
        "print('=' * 60)\n",
        "print('INSTRUCTION:', test_instruction)\n",
        "print('=' * 60)\n",
        "print('RESPONSE:', response)\n",
        "print('=' * 60)\n",
    ]))

    # ---- Cell 11: Interactive inference loop ----
    cells.append(_nb_cell("markdown", [
        "## 11 — Interactive Chat\n",
        "Run this cell to enter a loop where you can test multiple prompts.",
    ]))
    cells.append(_nb_cell("code", [
        "# Interactive inference loop\n",
        "print('Enter your prompts below (type \"quit\" to stop):\\n')\n",
        "\n",
        "while True:\n",
        "    user_input = input('Instruction: ')\n",
        "    if user_input.lower().strip() in ['quit', 'exit', 'q']:\n",
        "        break\n",
        "\n",
        "    prompt = alpaca_prompt.format(\n",
        "        instruction=user_input,\n",
        "        input='',\n",
        "        output='',\n",
        "    )\n",
        "    inputs = tokenizer([prompt], return_tensors='pt').to('cuda')\n",
        "    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, use_cache=True)\n",
        "    response = tokenizer.batch_decode(outputs)[0]\n",
        "    if '### Response:' in response:\n",
        "        response = response.split('### Response:')[-1].strip()\n",
        "        for tok in ['</s>', '<|endoftext|>', '<|end|>', tokenizer.eos_token or '']:\n",
        "            response = response.replace(tok, '').strip()\n",
        "    print(f'\\nResponse: {response}\\n')\n",
    ]))

    # ---- Cell 12: Load from Drive (bonus) ----
    cells.append(_nb_cell("markdown", [
        "## 12 — (Bonus) Load Model from Drive Later\n",
        "Use this cell in a new session to reload your fine-tuned model from Google Drive.",
    ]))
    cells.append(_nb_cell("code", [
        "# Run this in a NEW session to load your saved model:\n",
        "#\n",
        "# from unsloth import FastLanguageModel\n",
        "# model, tokenizer = FastLanguageModel.from_pretrained(SAVE_DIR)\n",
        "# FastLanguageModel.for_inference(model)\n",
        "# print('Model loaded from Drive!')\n",
    ]))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }

    return json.dumps(notebook, indent=2, ensure_ascii=False)


# ==================== VIEW: FINE-TUNING ====================

def render_finetuning():
    """Render fine-tuning view with two training paths: Docker Backend or Google Colab."""
    load_existing_datasets()

    st.markdown("""
    <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em;">
        MODEL FINE-TUNING
    </h1>
    <p style="color: #8c6b5d; font-size: 0.8rem;">
        Train custom models using Unsloth — choose your GPU path
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    tab_docker, tab_colab = st.tabs(["Docker Backend (Local GPU)", "Google Colab (Free Cloud GPU)"])

    # ==================== TAB 1: DOCKER BACKEND ====================
    with tab_docker:
        _render_docker_training()

    # ==================== TAB 2: GOOGLE COLAB ====================
    with tab_colab:
        _render_colab_training()


def _render_docker_training():
    """Docker backend training path — sends training config to remote GPU backend via HTTP."""
    profile = st.session_state.get("user_profile", {})
    user_id = profile.get("id")

    # Auto-discover backend from Supabase
    if user_id and not st.session_state.backend_url:
        discovered_url = check_backend_connection(user_id)
        if discovered_url:
            st.session_state.backend_url = discovered_url

    backend_url = st.session_state.backend_url

    if not backend_url:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 4px solid #eab308;">
            <div style="color: #eab308; font-weight: bold;">[NO GPU BACKEND DETECTED]</div>
            <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                Start the ModelForge backend on a machine with an NVIDIA GPU.
                It will auto-register and appear here.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("How to Start the Backend", expanded=True):
            st.markdown(f"""
            Run this single command on any machine with Docker and an NVIDIA GPU:
            ```bash
            docker run --gpus all -e USER_ID={user_id or '<your-user-id>'} modelforge/backend:latest
            ```
            The backend will auto-register via Cloudflare tunnel and appear here within 30 seconds.
            """)
            if user_id:
                st.code(f"Your User ID: {user_id}", language="text")

            # Manual override
            st.markdown("---")
            st.markdown("**Or paste a URL manually:**")
            url = st.text_input("Backend URL:", placeholder="https://xxxx.trycloudflare.com", key="backend_url_input")
            if st.button("Connect Manually", key="connect_backend"):
                if url:
                    try:
                        r = requests.get(f"{url.rstrip('/')}/health", timeout=10)
                        if r.status_code == 200:
                            st.session_state.backend_url = url.rstrip('/')
                            st.success("Connected!")
                            st.rerun()
                        else:
                            st.error(f"Backend responded with status {r.status_code}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")

            if st.button("Refresh", key="refresh_backend"):
                st.rerun()
        return

    # Connected state
    st.markdown(f"""
    <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
        <span style="color: #22c55e;">[CONNECTED]</span> <strong>GPU Backend Active</strong>
        <div style="color: #8c6b5d; font-size: 0.75rem;">{backend_url}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Disconnect", key="disconnect_backend"):
        st.session_state.backend_url = ""
        st.rerun()

    st.markdown("---")

    # Training configuration
    col_config, col_output = st.columns([1, 1])

    with col_config:
        st.markdown("### TRAINING_CONFIGURATION")

        st.markdown("#### BASE_MODEL")

        # Fetch base model list from the backend
        base_models = []
        try:
            r = requests.get(f"{backend_url}/base-models", timeout=10)
            if r.status_code == 200:
                base_models = r.json().get("base_models", [])
        except Exception:
            pass

        if base_models:
            selected_model = st.selectbox("Select Base Model", options=base_models, index=0,
                help="Choose a base model to fine-tune.", key="docker_base_model")
        else:
            st.warning("Could not fetch model list from backend. Enter a model name manually.")
            selected_model = st.text_input("Base Model Name",
                value="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
                placeholder="unsloth/model-name-bnb-4bit", key="docker_base_model_input")

        st.markdown("---")

        st.markdown("#### TRAINING_DATASET")
        available_datasets = []
        for ds_id, ds in st.session_state.datasets.items():
            available_datasets.append({
                "name": ds.name,
                "storage_path": ds.storage_path,
                "samples": ds.sample_count,
                "id": ds.id,
            })

        if not available_datasets:
            st.warning("No datasets found. Generate some training data first!")
            selected_dataset = None
        else:
            dataset_options = [f"{d['name']} ({d['samples']} samples)" for d in available_datasets]
            selected_idx = st.selectbox("Select Dataset", options=range(len(dataset_options)),
                format_func=lambda x: dataset_options[x], key="docker_dataset")
            selected_dataset = available_datasets[selected_idx] if dataset_options else None

        st.markdown("---")
        st.markdown("#### TRAINING_PARAMETERS")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            num_epochs = st.slider("Epochs", 1, 10, 3, key="docker_epochs")
            batch_size = st.selectbox("Batch Size", [1, 2, 4, 8], index=1, key="docker_batch")
            max_seq_length = st.selectbox("Max Seq Length", [512, 1024, 2048, 4096], index=1, key="docker_seq")
        with col_p2:
            learning_rate = st.select_slider("Learning Rate",
                options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4], value=2e-4,
                format_func=lambda x: f"{x:.0e}", key="docker_lr")
            lora_r = st.selectbox("LoRA Rank (r)", [8, 16, 32, 64], index=2, key="docker_lora_r")
            lora_alpha = st.selectbox("LoRA Alpha", [16, 32, 64, 128], index=1, key="docker_lora_a")

        st.markdown("---")
        st.markdown("#### QUANTIZATION")
        use_qlora = st.checkbox("Use QLoRA (4-bit quantization)", value=True,
            help="QLoRA uses less VRAM. Recommended for 8GB GPUs.", key="docker_qlora")

        st.markdown("---")
        st.markdown("#### OUTPUT_MODEL_NAME")
        custom_model_name = st.text_input("Model Name (optional)", value="",
            placeholder="e.g., my-custom-model", key="docker_model_name")

        st.markdown("---")
        st.markdown("#### TRAINING_ACTIONS")

        can_train = selected_dataset is not None
        is_training = st.session_state.get('remote_training_active', False)

        if st.button("[START_TRAINING]" if not is_training else "[TRAINING...]",
                     type="primary", use_container_width=True,
                     disabled=not can_train or is_training):
            config = {
                "model_name": selected_model,
                "max_seq_length": max_seq_length,
                "load_in_4bit": use_qlora,
                "use_qlora": use_qlora,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "num_train_epochs": num_epochs,
                "per_device_train_batch_size": batch_size,
                "learning_rate": learning_rate,
                "custom_model_name": custom_model_name,
                "dataset_storage_path": selected_dataset['storage_path'],
            }
            try:
                r = requests.post(f"{backend_url}/train", json=config, timeout=30)
                if r.status_code == 200:
                    st.session_state.remote_training_active = True
                    st.success("Training started on remote GPU!")
                    st.rerun()
                else:
                    st.error(f"Failed to start training: {r.text}")
            except Exception as e:
                st.error(f"Error communicating with backend: {e}")

    with col_output:
        st.markdown("### TRAINING_STATUS")

        if is_training:
            try:
                r = requests.get(f"{backend_url}/train/status", timeout=30)
                if r.status_code == 200:
                    status = r.json()
                    if status.get("status") == "training":
                        st.markdown(f"""
                        <div class="industrial-plate" style="border-left: 3px solid #f96124;">
                            <div style="color: #f96124; font-weight: bold;">[PROCESSING] Training in Progress</div>
                            <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                                <strong>Step:</strong> {status.get('current_step', 'N/A')} / {status.get('total_steps', 'N/A')}<br>
                                <strong>Loss:</strong> {status.get('loss', 'N/A')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if status.get("log"):
                            st.markdown("#### TRAINING_LOG")
                            st.code(status["log"][-3000:], language="text")

                        time.sleep(5)
                        st.rerun()

                    elif status.get("status") == "complete":
                        st.session_state.remote_training_active = False
                        st.markdown(f"""
                        <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
                            <div style="color: #22c55e; font-weight: bold;">[SUCCESS] Training Complete!</div>
                            <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                                <strong>Output:</strong> {status.get('model_path', 'N/A')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    elif status.get("status") == "error":
                        st.session_state.remote_training_active = False
                        st.error(f"Training failed: {status.get('error', 'Unknown error')}")

                    else:
                        st.info(f"Status: {status.get('status', 'unknown')}")
            except Exception:
                # Tunnel/GPU busy — show a non-alarming message and retry
                st.markdown("""
                <div class="industrial-plate" style="border-left: 3px solid #eab308;">
                    <div style="color: #eab308; font-weight: bold;">[TRAINING] GPU Busy — Waiting for Status...</div>
                    <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                        The backend is under heavy load. This is normal during training.
                        Status will refresh automatically.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(8)
                st.rerun()
        else:
            st.markdown("""
            <div class="industrial-plate" style="opacity: 0.7;">
                <div style="color: #8c6b5d; text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">GPU</div>
                    <div>Configure and start training</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### [QUICK_START]")
        st.markdown("""
        1. **Connect** — Start Docker backend on a GPU machine
        2. **Configure** — Select model & dataset
        3. **Train** — Click Start Training
        4. **Done** — Model saved on backend
        """)

    # Trained models from backend
    st.markdown("---")
    st.markdown("### TRAINED_MODELS")
    try:
        r = requests.get(f"{backend_url}/models", timeout=10)
        if r.status_code == 200:
            models = r.json().get("models", [])
            if models:
                for model_info in models:
                    st.markdown(f"""
                    <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
                        <div style="color: white; font-weight: bold;">MODEL: {model_info.get('name', 'Unknown')}</div>
                        <div style="color: #8c6b5d; font-size: 0.75rem;">Path: {model_info.get('path', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                if st.button("[GO_TO_INFERENCE]", type="primary", use_container_width=True):
                    st.session_state.view = 'inference'
                    st.rerun()
            else:
                st.info("[INFO] No trained models yet.")
    except:
        st.info("[INFO] Could not fetch models from backend.")


def _render_colab_training():
    """Google Colab training path — generates a notebook with all training code pre-filled."""
    st.markdown("""
    <div class="industrial-plate" style="border-left: 4px solid #f96124;">
        <div style="color: #f96124; font-weight: bold;">GOOGLE COLAB — FREE GPU TRAINING</div>
        <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
            No local GPU or Docker needed. Configure your training below, download the notebook,
            and open it in Google Colab. Training runs on Google's free T4 GPU.
            The model is saved to your Google Drive.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col_config, col_info = st.columns([1, 1])

    with col_config:
        st.markdown("### TRAINING_CONFIGURATION")

        st.markdown("#### BASE_MODEL")
        selected_model = st.selectbox("Select Base Model", options=COLAB_BASE_MODELS, index=0,
            help="Choose a base model to fine-tune on Colab.", key="colab_base_model")

        st.markdown("---")

        st.markdown("#### TRAINING_DATASET")
        available_datasets = []
        for ds_id, ds in st.session_state.datasets.items():
            available_datasets.append({
                "name": ds.name,
                "storage_path": ds.storage_path,
                "samples": ds.sample_count,
                "id": ds.id,
            })

        if not available_datasets:
            st.warning("No datasets found. Generate some training data first!")
            selected_dataset = None
        else:
            dataset_options = [f"{d['name']} ({d['samples']} samples)" for d in available_datasets]
            selected_idx = st.selectbox("Select Dataset", options=range(len(dataset_options)),
                format_func=lambda x: dataset_options[x], key="colab_dataset")
            selected_dataset = available_datasets[selected_idx] if dataset_options else None

        st.markdown("---")
        st.markdown("#### TRAINING_PARAMETERS")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            num_epochs = st.slider("Epochs", 1, 10, 3, key="colab_epochs")
            batch_size = st.selectbox("Batch Size", [1, 2, 4, 8], index=1, key="colab_batch")
            max_seq_length = st.selectbox("Max Seq Length", [512, 1024, 2048, 4096], index=1, key="colab_seq")
        with col_p2:
            learning_rate = st.select_slider("Learning Rate",
                options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4], value=2e-4,
                format_func=lambda x: f"{x:.0e}", key="colab_lr")
            lora_r = st.selectbox("LoRA Rank (r)", [8, 16, 32, 64], index=2, key="colab_lora_r")
            lora_alpha = st.selectbox("LoRA Alpha", [16, 32, 64, 128], index=1, key="colab_lora_a")

        st.markdown("---")
        st.markdown("#### QUANTIZATION")
        use_qlora = st.checkbox("Use QLoRA (4-bit quantization)", value=True,
            help="QLoRA uses less VRAM. Recommended for Colab T4 GPU.", key="colab_qlora")

        st.markdown("---")
        st.markdown("#### OUTPUT_MODEL_NAME")
        custom_model_name = st.text_input("Model Name (optional)", value="",
            placeholder="e.g., my-custom-model", key="colab_model_name")

        st.markdown("---")
        st.markdown("#### GENERATE_NOTEBOOK")

        can_generate = selected_dataset is not None
        has_github_token = _get_github_token() is not None

        if has_github_token:
            # Primary: Open in Colab directly via GitHub Gist
            if st.button("[OPEN IN GOOGLE COLAB]", type="primary", use_container_width=True,
                         disabled=not can_generate, key="colab_open"):
                config = {
                    "model_name": selected_model,
                    "max_seq_length": max_seq_length,
                    "load_in_4bit": use_qlora,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "num_train_epochs": num_epochs,
                    "per_device_train_batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "custom_model_name": custom_model_name,
                    "dataset_storage_path": selected_dataset['storage_path'],
                    "dataset_name": selected_dataset['name'],
                }
                notebook_json = generate_colab_notebook(config)
                nb_name = (
                    custom_model_name.strip() or selected_model.split("/")[-1].replace("-bnb-4bit", "")
                ) + "_training.ipynb"

                colab_url = upload_notebook_to_gist(notebook_json, nb_name)
                if colab_url:
                    st.session_state['colab_url'] = colab_url
                    st.session_state['colab_notebook_json'] = notebook_json
                    st.session_state['colab_notebook_name'] = nb_name
                else:
                    st.error("Failed to create Gist. Check your GITHUB_TOKEN in secrets.")

            # Show Colab link if gist was created
            if 'colab_url' in st.session_state:
                colab_url = st.session_state['colab_url']
                st.markdown(f"""
                <a href="{colab_url}" target="_blank" style="
                    display: block; text-align: center; padding: 0.6rem;
                    background: #f96124; color: white; font-weight: bold;
                    border-radius: 4px; text-decoration: none; margin-top: 0.5rem;
                ">OPEN NOTEBOOK IN COLAB &rarr;</a>
                """, unsafe_allow_html=True)
                # Auto-open via JS
                components.html(
                    f'<script>window.open("{colab_url}", "_blank");</script>',
                    height=0,
                )
                st.success("Notebook created and opened in Colab!")

        else:
            # Fallback: Generate + download (no GitHub token configured)
            if st.button("[GENERATE COLAB NOTEBOOK]", type="primary", use_container_width=True,
                         disabled=not can_generate, key="colab_generate"):
                config = {
                    "model_name": selected_model,
                    "max_seq_length": max_seq_length,
                    "load_in_4bit": use_qlora,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "num_train_epochs": num_epochs,
                    "per_device_train_batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "custom_model_name": custom_model_name,
                    "dataset_storage_path": selected_dataset['storage_path'],
                    "dataset_name": selected_dataset['name'],
                }
                notebook_json = generate_colab_notebook(config)
                st.session_state['colab_notebook_json'] = notebook_json
                st.session_state['colab_notebook_name'] = (
                    custom_model_name.strip() or selected_model.split("/")[-1].replace("-bnb-4bit", "")
                ) + "_training.ipynb"

            st.caption("Add `GITHUB_TOKEN` to secrets to enable direct Open in Colab.")

        # Always show download button if notebook was generated
        if 'colab_notebook_json' in st.session_state:
            nb_name = st.session_state.get('colab_notebook_name', 'modelforge_training.ipynb')
            st.download_button(
                label="[SAVE .ipynb FILE]",
                data=st.session_state['colab_notebook_json'],
                file_name=nb_name,
                mime="application/x-ipynb+json",
                use_container_width=True,
                key="colab_download",
            )

    with col_info:
        st.markdown("### HOW IT WORKS")
        st.markdown("""
        <div class="industrial-plate" style="opacity: 0.9;">
            <div style="color: #f96124; font-weight: bold; margin-bottom: 0.5rem;">STEP-BY-STEP</div>
            <div style="color: #8c6b5d; font-size: 0.85rem; line-height: 1.8;">
                <strong>1.</strong> Configure training parameters on the left<br>
                <strong>2.</strong> Click <em>Open in Google Colab</em><br>
                <strong>3.</strong> Colab opens automatically with your notebook<br>
                <strong>4.</strong> Runtime → Change runtime type → <strong>T4 GPU</strong><br>
                <strong>5.</strong> Run All cells (Ctrl+F9)<br>
                <strong>6.</strong> Model saves to Google Drive automatically
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### WHAT'S IN THE NOTEBOOK")
        st.markdown("""
        The generated notebook includes **12 cells**:

        1. **Install** — Unsloth, Supabase client, transformers
        2. **Mount Drive** — For persistent model storage
        3. **Download Dataset** — From ModelForge cloud storage
        4. **Format Dataset** — Alpaca prompt template
        5. **Load Model** — Base model with Unsloth
        6. **Apply LoRA** — Low-rank adapter configuration
        7. **Train** — Full training loop with live loss
        8. **Save to Drive** — Persistent model storage
        9. **Push to HF** — Optional Hugging Face Hub upload
        10. **Test Inference** — Single prompt test
        11. **Interactive Chat** — Multi-turn testing loop
        12. **Reload Model** — Load from Drive in new sessions
        """)

        st.markdown("---")
        st.markdown("### COLAB LIMITS")
        st.markdown("""
        <div class="industrial-plate">
            <div style="color: #8c6b5d; font-size: 0.85rem; line-height: 1.7;">
                <strong>Free Tier:</strong> T4 GPU (16GB VRAM), ~4h sessions<br>
                <strong>Best for:</strong> 1B-4B parameter models with QLoRA<br>
                <strong>Colab Pro:</strong> A100 GPU, longer sessions, more RAM<br>
                <strong>Tip:</strong> Models ≤3B fit comfortably on free T4
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==================== VIEW: INFERENCE (REMOTE BACKEND) ====================

def render_inference():
    """Render inference testing page - sends requests to remote GPU backend"""
    st.markdown("""
    <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em;">
        MODEL INFERENCE
    </h1>
    <p style="color: #8c6b5d; font-size: 0.8rem;">Test and compare models via remote GPU backend</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Auto-discover backend from Supabase
    profile = st.session_state.get("user_profile", {})
    user_id = profile.get("id")
    if user_id and not st.session_state.backend_url:
        discovered_url = check_backend_connection(user_id)
        if discovered_url:
            st.session_state.backend_url = discovered_url

    backend_url = st.session_state.backend_url

    if not backend_url:
        st.warning("No GPU backend detected. Start the backend or go to Fine-Tuning page for instructions.")
        return

    st.markdown(f"""
    <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
        <span style="color: #22c55e;">[CONNECTED]</span> <strong>GPU Backend</strong>
        <div style="color: #8c6b5d; font-size: 0.75rem;">{backend_url}</div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch available models from backend
    try:
        r = requests.get(f"{backend_url}/models", timeout=10)
        trained_models = r.json().get("models", []) if r.status_code == 200 else []
    except:
        trained_models = []

    if not trained_models:
        st.warning("[WARN] No trained models found on backend. Train a model first!")
        return

    # Mode tabs
    mode_tab = st.radio("Testing Mode", ["[SINGLE_MODEL]", "[ARENA_MODE]"], horizontal=True, key="inference_mode")
    st.markdown("---")

    if mode_tab == "[SINGLE_MODEL]":
        _render_single_inference(backend_url, trained_models)
    else:
        _render_arena_mode(backend_url, trained_models)


def _render_single_inference(backend_url, trained_models):
    """Single model inference via remote backend"""
    col_model, col_test = st.columns([1, 2])

    with col_model:
        st.markdown("### SELECT_MODEL")
        model_names = [m.get("name", "Unknown") for m in trained_models]
        for i, model in enumerate(trained_models):
            is_selected = st.session_state.get('selected_test_model') == model.get("path", "")
            border_color = "#f96124" if is_selected else "#3d3d3d"
            st.markdown(f"""
            <div class="industrial-plate" style="border-left: 3px solid {border_color};">
                <div style="color: white; font-weight: bold;">MODEL: {model.get('name', 'Unknown')}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select", key=f"select_model_inf_{i}", use_container_width=True):
                st.session_state.selected_test_model = model.get("path", "")
                st.session_state.selected_test_model_name = model.get("name", "Unknown")
                st.session_state.inference_result = None
                st.rerun()

    with col_test:
        st.markdown("### TEST_INFERENCE")
        selected_model = st.session_state.get('selected_test_model')
        if not selected_model:
            st.info("[INFO] Select a model from the left to begin testing")
            return

        st.markdown(f"""
        <div class="industrial-plate" style="border-left: 3px solid #f96124;">
            <div style="color: #f96124; font-size: 0.85rem;">SELECTED_MODEL:</div>
            <div style="color: white; font-weight: bold; font-size: 1.1rem;">{st.session_state.get('selected_test_model_name', '')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ENTER_PROMPT")

        test_prompt = st.text_area("Prompt:", value=st.session_state.get('test_prompt', "Hello! Can you introduce yourself?"),
            height=120, key="inference_prompt_input")

        is_inferencing = st.session_state.get('inference_running', False)

        col_run, col_clear = st.columns([2, 1])
        with col_run:
            if st.button("[RUN_INFERENCE]" if not is_inferencing else "[PROCESSING]",
                         type="primary", use_container_width=True,
                         disabled=is_inferencing or not test_prompt.strip()):
                st.session_state.test_prompt = test_prompt
                st.session_state.inference_running = True
                st.rerun()
        with col_clear:
            if st.button("[CLEAR]", use_container_width=True, key="clear_inf"):
                st.session_state.inference_result = None
                st.rerun()

        if st.session_state.get('inference_running'):
            with st.spinner("[PROCESSING] Running inference on remote GPU..."):
                try:
                    r = requests.post(f"{backend_url}/inference", json={
                        "model_path": selected_model,
                        "prompt": st.session_state.get('test_prompt', test_prompt)
                    }, timeout=120)
                    st.session_state.inference_running = False
                    if r.status_code == 200:
                        result = r.json()
                        st.session_state.inference_result = {
                            "success": True, "output": result.get("response", "No response"),
                            "prompt": st.session_state.get('test_prompt', test_prompt)
                        }
                    else:
                        st.session_state.inference_result = {
                            "success": False, "output": r.text,
                            "prompt": st.session_state.get('test_prompt', test_prompt)
                        }
                except Exception as e:
                    st.session_state.inference_running = False
                    st.session_state.inference_result = {
                        "success": False, "output": str(e),
                        "prompt": st.session_state.get('test_prompt', test_prompt)
                    }
                st.rerun()

        if st.session_state.get('inference_result') is not None:
            st.markdown("---")
            result = st.session_state.inference_result
            if result.get('success'):
                st.markdown("#### MODEL_RESPONSE")
                st.markdown(f"""
                <div class="industrial-plate" style="border-left: 3px solid #22c55e; padding: 1rem;">
                    <div style="color: white; white-space: pre-wrap; font-family: monospace;">{result.get('output', '')}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"[FAILED] {result.get('output', 'Unknown error')}")


def _render_arena_mode(backend_url, trained_models):
    """Arena mode for comparing fine-tuned vs base models via remote backend"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h2 style="color: #f96124; font-size: 1.8rem; font-weight: 900; letter-spacing: 0.15em;">MODEL ARENA (BATTLE)</h2>
        <p style="color: #8c6b5d; font-size: 0.85rem;">Compare your fine-tuned model against a base model from Backboard</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ARENA_CONFIGURATION")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="industrial-plate" style="border-left: 3px solid #f96124;"><div style="color: #f96124; font-weight: bold; font-size: 0.9rem;">CHALLENGER (FT)</div></div>', unsafe_allow_html=True)
        model_names = [m.get("name", "Unknown") for m in trained_models]
        selected_finetuned_idx = st.selectbox("Select Fine-tuned Model", range(len(model_names)),
            format_func=lambda i: model_names[i], key="arena_finetuned_model")
        finetuned_model = trained_models[selected_finetuned_idx]

    with col_right:
        st.markdown('<div class="industrial-plate" style="border-left: 3px solid #22c55e;"><div style="color: #22c55e; font-weight: bold; font-size: 0.9rem;">DEFENDER (Base Model)</div></div>', unsafe_allow_html=True)
        api_key = st.session_state.get('api_key')
        if not api_key:
            st.warning("Set Backboard API key to use base models")
            base_model = None
        else:
            base_model = st.text_input("Model Name",
                value=st.session_state.get('arena_base_model_input', 'openai/gpt-4o'),
                key="arena_base_model_input", placeholder="openai/gpt-4o")
            if not base_model or '/' not in base_model:
                st.warning("Enter model as `provider/model-name`")
                base_model = None

    st.markdown("---")

    # Judge
    st.markdown("### Judge Configuration (Optional)")
    col_judge, col_criteria = st.columns([1, 2])
    with col_judge:
        enable_judge = st.checkbox("Enable Judge Model", key="arena_enable_judge")
        judge_model = None
        if enable_judge:
            judge_model = st.text_input("Judge Model",
                value=st.session_state.get('arena_judge_model_input', 'anthropic/claude-3-5-sonnet-20241022'),
                key="arena_judge_model_input")
            if not judge_model or '/' not in judge_model:
                st.warning("Enter model as `provider/model-name`")
                judge_model = None
    with col_criteria:
        if enable_judge:
            judge_criteria = st.text_area("Judging Criteria",
                value="Evaluate both responses based on:\n1. Accuracy\n2. Relevance\n3. Clarity\n4. Completeness\n5. Helpfulness\n\nProvide a brief analysis and declare a winner.",
                height=180, key="arena_judge_criteria")

    st.markdown("---")
    st.markdown("### ARENA_PROMPT")
    arena_prompt = st.text_area("Enter the prompt to test both models:", height=120,
        key="arena_prompt_input", placeholder="Enter your prompt here...")

    col_run, col_clear = st.columns([3, 1])
    is_running = st.session_state.get('arena_running', False)
    with col_run:
        run_disabled = is_running or not arena_prompt.strip() or not base_model
        if st.button("[START_BATTLE]" if not is_running else "[BATTLE_IN_PROGRESS]",
                     type="primary", use_container_width=True, disabled=run_disabled):
            st.session_state.arena_running = True
            st.session_state.arena_results = None
            st.session_state.arena_prompt = arena_prompt
            st.rerun()
    with col_clear:
        if st.button("[CLEAR]", use_container_width=True, key="clear_arena"):
            st.session_state.arena_results = None
            st.session_state.judge_result = None
            st.rerun()

    # Execute arena
    if st.session_state.get('arena_running'):
        st.markdown("---")
        st.markdown("### Battle in Progress...")
        prompt_to_use = st.session_state.get('arena_prompt', arena_prompt)
        results = {}
        progress = st.empty()

        # Fine-tuned model via backend
        progress.markdown("Running **Challenger** (Fine-tuned model)...")
        try:
            r = requests.post(f"{backend_url}/inference", json={
                "model_path": finetuned_model.get("path", ""),
                "prompt": prompt_to_use
            }, timeout=120)
            if r.status_code == 200:
                results['finetuned'] = {'success': True, 'output': r.json().get("response", ""),
                                         'model_name': finetuned_model.get("name", "Unknown")}
            else:
                results['finetuned'] = {'success': False, 'output': r.text,
                                         'model_name': finetuned_model.get("name", "Unknown")}
        except Exception as e:
            results['finetuned'] = {'success': False, 'output': str(e),
                                     'model_name': finetuned_model.get("name", "Unknown")}

        # Base model via Backboard
        progress.markdown("Running **Defender** (Base model via Backboard)...")
        try:
            backboard_mgr = BackboardManager(api_key=st.session_state.api_key)
            model_parts = base_model.split("/")
            llm_provider = model_parts[0] if len(model_parts) > 1 else "openai"
            model_name_str = model_parts[1] if len(model_parts) > 1 else base_model

            async def run_base():
                assistant = await backboard_mgr.client.create_assistant(
                    name="Arena Base Model",
                    description="You are a helpful AI assistant.")
                thread = await backboard_mgr.client.create_thread(assistant.assistant_id)
                response = await backboard_mgr.client.add_message(
                    thread_id=thread.thread_id, content=prompt_to_use,
                    llm_provider=llm_provider, model_name=model_name_str, stream=False)
                return response.content if hasattr(response, 'content') else str(response)

            base_output = asyncio.run(run_base())
            results['base'] = {'success': True, 'output': base_output, 'model_name': base_model}
        except Exception as e:
            results['base'] = {'success': False, 'output': str(e), 'model_name': base_model}

        # Judge
        if enable_judge and judge_model and results.get('finetuned', {}).get('success') and results.get('base', {}).get('success'):
            progress.markdown("Running **Judge**...")
            try:
                judge_parts = judge_model.split("/")
                j_provider = judge_parts[0] if len(judge_parts) > 1 else "openai"
                j_model = judge_parts[1] if len(judge_parts) > 1 else judge_model
                judge_prompt_text = f"""You are an impartial judge evaluating two AI model responses.

**PROMPT:** {prompt_to_use}

**RESPONSE A (Model: {results['finetuned']['model_name']}):**
{results['finetuned']['output']}

**RESPONSE B (Model: {results['base']['model_name']}):**
{results['base']['output']}

**CRITERIA:** {st.session_state.get('arena_judge_criteria', 'Evaluate overall quality')}

Provide analysis of each and declare: "WINNER: A" or "WINNER: B" or "TIE"
"""
                async def run_judge():
                    ja = await backboard_mgr.client.create_assistant(name="Arena Judge",
                        description="Impartial AI judge.")
                    jt = await backboard_mgr.client.create_thread(ja.assistant_id)
                    jr = await backboard_mgr.client.add_message(thread_id=jt.thread_id,
                        content=judge_prompt_text, llm_provider=j_provider, model_name=j_model, stream=False)
                    return jr.content if hasattr(jr, 'content') else str(jr)

                verdict = asyncio.run(run_judge())
                st.session_state.judge_result = {'success': True, 'verdict': verdict, 'judge_model': judge_model}
            except Exception as e:
                st.session_state.judge_result = {'success': False, 'verdict': str(e), 'judge_model': judge_model}

        progress.empty()
        st.session_state.arena_running = False
        st.session_state.arena_results = results
        st.rerun()

    # Display results
    if st.session_state.get('arena_results'):
        st.markdown("---")
        st.markdown("### ARENA_RESULTS")
        results = st.session_state.arena_results

        col_a, col_vs, col_b = st.columns([5, 1, 5])
        with col_a:
            ft = results.get('finetuned', {})
            c = "#f96124" if ft.get('success') else "#ef4444"
            st.markdown(f"""
            <div class="industrial-plate" style="border: 2px solid {c}; min-height: 300px;">
                <div style="color: {c}; font-weight: bold; font-size: 1rem; text-align: center;">CHALLENGER (FT)</div>
                <div style="color: #8c6b5d; font-size: 0.75rem; text-align: center; margin-bottom: 1rem;">{ft.get('model_name', '')}</div>
                <div style="color: white; white-space: pre-wrap; font-family: monospace; font-size: 0.85rem; max-height: 400px; overflow-y: auto;">{ft.get('output', 'No output')[:2000]}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_vs:
            st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 300px;"><span style="font-size: 2rem; color: #f96124; font-weight: 900;">VS</span></div>', unsafe_allow_html=True)
        with col_b:
            base = results.get('base', {})
            c2 = "#22c55e" if base.get('success') else "#ef4444"
            st.markdown(f"""
            <div class="industrial-plate" style="border: 2px solid {c2}; min-height: 300px;">
                <div style="color: {c2}; font-weight: bold; font-size: 1rem; text-align: center;">DEFENDER (BASE)</div>
                <div style="color: #8c6b5d; font-size: 0.75rem; text-align: center; margin-bottom: 1rem;">{base.get('model_name', '')}</div>
                <div style="color: white; white-space: pre-wrap; font-family: monospace; font-size: 0.85rem; max-height: 400px; overflow-y: auto;">{base.get('output', 'No output')[:2000]}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### YOUR_VOTE")
        col_va, col_tie, col_vb = st.columns(3)
        with col_va:
            if st.button("[CHALLENGER_WINS]", use_container_width=True):
                st.success("You voted for the Challenger!")
        with col_tie:
            if st.button("[TIE]", use_container_width=True):
                st.info("You declared a tie!")
        with col_vb:
            if st.button("[DEFENDER_WINS]", use_container_width=True):
                st.success("You voted for the Defender!")

        if st.session_state.get('judge_result'):
            st.markdown("---")
            st.markdown("### JUDGE_VERDICT")
            jr = st.session_state.judge_result
            if jr.get('success'):
                verdict = jr.get('verdict', '')
                w_color = "#f96124"
                if "WINNER: A" in verdict: w_color = "#f96124"
                elif "WINNER: B" in verdict: w_color = "#22c55e"
                else: w_color = "#f59e0b"
                st.markdown(f"""
                <div class="industrial-plate" style="border: 2px solid {w_color};">
                    <div style="color: #8c6b5d; font-size: 0.75rem; text-align: center; margin-bottom: 1rem;">Judged by: {jr.get('judge_model', '')}</div>
                    <div style="color: white; white-space: pre-wrap; font-size: 0.9rem;">{verdict}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Judge error: {jr.get('verdict', '')}")


# ==================== VIEW: HUGGINGFACE DEPLOY ====================

def render_hf_deploy():
    """Render the HuggingFace deployment page - pushes models/datasets via backend"""
    load_existing_datasets()

    st.markdown("""
    <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em;">
        HUGGINGFACE DEPLOY
    </h1>
    <p style="color: #8c6b5d; font-size: 0.8rem;">Upload your fine-tuned models and datasets to HuggingFace Hub via GPU Backend</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Check backend connection
    profile = st.session_state.get("user_profile", {})
    user_id = profile.get("id")

    if user_id and not st.session_state.backend_url:
        discovered_url = check_backend_connection(user_id)
        if discovered_url:
            st.session_state.backend_url = discovered_url

    backend_url = st.session_state.backend_url

    if not backend_url:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 4px solid #eab308;">
            <div style="color: #eab308; font-weight: bold;">[NO GPU BACKEND DETECTED]</div>
            <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                Connect your GPU backend first. Models are stored on the backend and
                pushed to HuggingFace from there.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("[GO_TO_TRAINING]", use_container_width=True):
            st.session_state.view = 'fine_tuning'
            st.rerun()
        return

    st.markdown(f"""
    <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
        <span style="color: #22c55e;">[CONNECTED]</span> <strong>GPU Backend Active</strong>
        <div style="color: #8c6b5d; font-size: 0.75rem;">{backend_url}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Read credentials from session state (configured in Settings page)
    hf_token = st.session_state.get('hf_token', '')
    hf_username = st.session_state.get('hf_username', '')

    if not hf_token or not hf_username:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 4px solid #eab308;">
            <div style="color: #eab308; font-weight: bold;">[HUGGINGFACE CREDENTIALS REQUIRED]</div>
            <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                Configure your HuggingFace token and username in the <strong>Settings</strong> page before uploading.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("[GO TO SETTINGS]", use_container_width=True, key="hf_goto_settings"):
            st.session_state.view = 'settings'
            st.rerun()
        return
    else:
        st.markdown(f"""
        <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
            <span style="color: #22c55e;">[AUTHENTICATED]</span>
            HuggingFace user: <strong style="color: white;">{hf_username}</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_models, col_datasets = st.columns(2)

    # ---- Model Upload ----
    with col_models:
        st.markdown("### UPLOAD MODEL")

        # Fetch models from backend
        backend_models = []
        try:
            r = requests.get(f"{backend_url}/models", timeout=15)
            if r.status_code == 200:
                backend_models = r.json().get("models", [])
        except Exception:
            pass

        if backend_models:
            model_names = [m["name"] for m in backend_models]
            selected_model = st.selectbox("Select Trained Model", model_names, key="hf_model_select")

            # Show model info
            model_info = next((m for m in backend_models if m["name"] == selected_model), {})
            st.caption(f"Path: {model_info.get('path', 'N/A')} | LoRA: {'Yes' if model_info.get('has_adapter') else 'No'}")

            repo_name = st.text_input("Repository Name", value=selected_model or "",
                placeholder="my-awesome-model", key="hf_repo_name_model")
            private_repo = st.checkbox("Private Repository", value=False, key="hf_private_model")

            can_push = hf_token and hf_username and selected_model
            if st.button("[PUSH MODEL TO HF]", type="primary", use_container_width=True,
                         disabled=not can_push, key="push_model_btn"):
                progress_bar = st.progress(0, text="Preparing model upload...")
                status_text = st.empty()
                try:
                    progress_bar.progress(10, text="Sending model to HuggingFace...")
                    status_text.caption("This may take several minutes for large models.")

                    # Use a thread to poll progress while waiting
                    import threading, time as _time
                    upload_done = threading.Event()
                    upload_result = [None]

                    def _do_upload():
                        try:
                            upload_result[0] = requests.post(f"{backend_url}/push-to-hf", json={
                                "hf_token": hf_token,
                                "hf_username": hf_username,
                                "repo_name": repo_name or selected_model,
                                "model_name": selected_model,
                                "private": private_repo,
                            }, timeout=600)
                        except Exception as ex:
                            upload_result[0] = ex
                        finally:
                            upload_done.set()

                    t = threading.Thread(target=_do_upload, daemon=True)
                    t.start()

                    # Animate progress bar while waiting
                    pct = 10
                    while not upload_done.is_set():
                        _time.sleep(2)
                        if pct < 90:
                            pct += 2
                        progress_bar.progress(pct, text=f"Uploading model to HuggingFace... {pct}%")

                    t.join()
                    r = upload_result[0]

                    if isinstance(r, Exception):
                        progress_bar.progress(100, text="Upload failed")
                        if "Timeout" in type(r).__name__ or "timeout" in str(r).lower():
                            st.warning("Request timed out — the upload may still be in progress. Large models take a while.")
                        else:
                            st.error(f"Error: {r}")
                    elif r.status_code == 200:
                        result = r.json()
                        if result.get("status") == "success":
                            progress_bar.progress(100, text="Upload complete!")
                            st.success("Model uploaded successfully!")
                            url = result.get("url", "")
                            st.markdown(f"""
                            <div style="background: #0a2810; border: 1px solid #22c55e; padding: 1rem; border-radius: 4px;">
                                <a href="{url}" target="_blank" style="color: #22c55e; font-weight: bold;">{url}</a>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            progress_bar.progress(100, text="Upload failed")
                            st.error(f"Push failed: {result.get('error', 'Unknown error')}")
                    else:
                        progress_bar.progress(100, text="Upload failed")
                        st.error(f"Backend error: {r.status_code} — {r.text[:200]}")
                except Exception as e:
                    progress_bar.progress(100, text="Upload failed")
                    st.error(f"Error: {e}")
                finally:
                    status_text.empty()
        else:
            st.info("No trained models found on backend. Train a model first!")
            if st.button("[GO_TO_TRAINING]", use_container_width=True, key="hf_goto_train"):
                st.session_state.view = 'fine_tuning'
                st.rerun()

    # ---- Dataset Upload ----
    with col_datasets:
        st.markdown("### UPLOAD DATASET")

        available_datasets = []
        for ds_id, ds in st.session_state.datasets.items():
            available_datasets.append({
                "name": ds.name,
                "storage_path": ds.storage_path,
                "samples": ds.sample_count,
                "id": ds.id,
            })

        if available_datasets:
            dataset_options = [f"{d['name']} ({d['samples']} samples)" for d in available_datasets]
            selected_idx = st.selectbox("Select Dataset", options=range(len(dataset_options)),
                format_func=lambda x: dataset_options[x], key="hf_dataset_select")
            selected_dataset = available_datasets[selected_idx] if dataset_options else None

            if selected_dataset:
                ds_filename = selected_dataset["name"].replace(" ", "_") + ".jsonl"
                dataset_repo_name = st.text_input("Repository Name",
                    value=selected_dataset["name"].replace(" ", "-").lower(),
                    placeholder="my-awesome-dataset", key="hf_repo_name_dataset")
                private_ds_repo = st.checkbox("Private Repository", value=False, key="hf_private_dataset")

                can_push_ds = hf_token and hf_username and selected_dataset
                if st.button("[PUSH DATASET TO HF]", type="primary", use_container_width=True,
                             disabled=not can_push_ds, key="push_dataset_btn"):
                    progress_bar_ds = st.progress(0, text="Preparing dataset upload...")
                    status_text_ds = st.empty()
                    try:
                        progress_bar_ds.progress(10, text="Sending dataset to HuggingFace...")
                        status_text_ds.caption("Downloading from storage and uploading...")

                        import threading, time as _time
                        upload_done_ds = threading.Event()
                        upload_result_ds = [None]

                        def _do_ds_upload():
                            try:
                                upload_result_ds[0] = requests.post(f"{backend_url}/push-dataset-to-hf", json={
                                    "hf_token": hf_token,
                                    "hf_username": hf_username,
                                    "repo_name": dataset_repo_name,
                                    "dataset_storage_path": selected_dataset["storage_path"],
                                    "dataset_filename": ds_filename,
                                    "private": private_ds_repo,
                                }, timeout=300)
                            except Exception as ex:
                                upload_result_ds[0] = ex
                            finally:
                                upload_done_ds.set()

                        t_ds = threading.Thread(target=_do_ds_upload, daemon=True)
                        t_ds.start()

                        pct_ds = 10
                        while not upload_done_ds.is_set():
                            _time.sleep(1.5)
                            if pct_ds < 90:
                                pct_ds += 3
                            progress_bar_ds.progress(pct_ds, text=f"Uploading dataset... {pct_ds}%")

                        t_ds.join()
                        r = upload_result_ds[0]

                        if isinstance(r, Exception):
                            progress_bar_ds.progress(100, text="Upload failed")
                            if "Timeout" in type(r).__name__ or "timeout" in str(r).lower():
                                st.warning("Request timed out — the upload may still be in progress.")
                            else:
                                st.error(f"Error: {r}")
                        elif r.status_code == 200:
                            result = r.json()
                            if result.get("status") == "success":
                                progress_bar_ds.progress(100, text="Upload complete!")
                                st.success("Dataset uploaded successfully!")
                                url = result.get("url", "")
                                st.markdown(f"""
                                <div style="background: #0a2810; border: 1px solid #22c55e; padding: 1rem; border-radius: 4px;">
                                    <a href="{url}" target="_blank" style="color: #22c55e; font-weight: bold;">{url}</a>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                progress_bar_ds.progress(100, text="Upload failed")
                                st.error(f"Push failed: {result.get('error', 'Unknown error')}")
                        else:
                            progress_bar_ds.progress(100, text="Upload failed")
                            st.error(f"Backend error: {r.status_code} — {r.text[:200]}")
                    except Exception as e:
                        progress_bar_ds.progress(100, text="Upload failed")
                        st.error(f"Error: {e}")
                    finally:
                        status_text_ds.empty()
        else:
            st.info("No datasets found. Generate some data first!")
            if st.button("[GO_TO_DASHBOARD]", use_container_width=True, key="hf_goto_dash"):
                st.session_state.view = 'dashboard'
                st.rerun()

    st.markdown("---")
    st.markdown("### TIPS")
    st.markdown("""
    <div style="background: #111; border: 1px solid #333; padding: 1rem; font-size: 0.85rem; border-radius: 4px;">
        <ul style="margin: 0; padding-left: 1.5rem; color: #888;">
            <li><strong>Model uploads</strong> include all adapter files (LoRA weights, config, tokenizer)</li>
            <li><strong>Dataset uploads</strong> push JSONL files from Supabase Storage</li>
            <li>Use <strong>private repositories</strong> for proprietary data or models</li>
            <li>Load models: <code>model = AutoModel.from_pretrained("username/repo")</code></li>
            <li>Load datasets: <code>dataset = load_dataset("username/repo")</code></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ==================== VIEW: SETTINGS ====================

def render_settings():
    """Render settings page with API key management, backend config, and logout."""
    st.markdown("""
    <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em;">
        SETTINGS
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- API Keys Section ---
    st.markdown("### API KEYS")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="industrial-plate">
            <div style="color: #f96124; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.75rem;">
                BACKBOARD API KEY
            </div>
        </div>
        """, unsafe_allow_html=True)
        current_key = st.session_state.get("api_key", "")
        if current_key:
            masked = current_key[:6] + "..." + current_key[-4:] if len(current_key) > 10 else "****"
            st.success(f"Configured: {masked}")
        else:
            st.warning("Not configured — data generation will not work.")

        new_key = st.text_input(
            "Enter Backboard API Key",
            type="password",
            placeholder="sk-...",
            key="settings_api_key",
            label_visibility="collapsed",
        )
        if st.button("SAVE API KEY", type="primary", key="save_api_key"):
            if new_key.strip():
                st.session_state.api_key = new_key.strip()
                st.session_state.validated_models = {}  # Clear model cache
                save_profile_field("backboard_api_key", new_key.strip())
                st.success("API Key saved!")
                st.rerun()
            else:
                st.error("Please enter a valid key.")

        if current_key and st.button("CLEAR API KEY", key="clear_api_key"):
            st.session_state.api_key = None
            st.session_state.validated_models = {}
            save_profile_field("backboard_api_key", None)
            st.rerun()

    with col2:
        st.markdown("""
        <div class="industrial-plate">
            <div style="color: #f96124; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.75rem;">
                HUGGINGFACE TOKEN
            </div>
        </div>
        """, unsafe_allow_html=True)
        current_hf = st.session_state.get("hf_token", "")
        if current_hf:
            st.success("HuggingFace token configured")
        else:
            st.info("Optional — needed for model/dataset uploads.")

        new_hf = st.text_input(
            "Enter HuggingFace Token",
            type="password",
            placeholder="hf_...",
            key="settings_hf_token",
            label_visibility="collapsed",
        )
        if st.button("SAVE HF TOKEN", type="primary", key="save_hf_token"):
            if new_hf.strip():
                st.session_state.hf_token = new_hf.strip()
                save_profile_field("hf_token", new_hf.strip())
                st.success("HuggingFace token saved!")
                st.rerun()
            else:
                st.error("Please enter a valid token.")

    st.markdown("---")

    # --- HuggingFace Username ---
    st.markdown("### HUGGINGFACE USERNAME")
    st.markdown("""
    <div class="industrial-plate">
        <div style="color: #f96124; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.75rem;">
            HUGGINGFACE USERNAME
        </div>
        <div style="color: #8c6b5d; font-size: 0.8rem;">Your HuggingFace username for model/dataset uploads.</div>
    </div>
    """, unsafe_allow_html=True)
    current_hf_user = st.session_state.get("hf_username", "")
    if current_hf_user:
        st.success(f"Username: {current_hf_user}")
    else:
        st.info("Optional — needed for HuggingFace uploads.")

    new_hf_user = st.text_input(
        "Enter HuggingFace Username",
        placeholder="your-username",
        key="settings_hf_username",
        label_visibility="collapsed",
    )
    if st.button("SAVE HF USERNAME", type="primary", key="save_hf_username"):
        if new_hf_user.strip():
            st.session_state.hf_username = new_hf_user.strip()
            save_profile_field("hf_username", new_hf_user.strip())
            st.success("HuggingFace username saved!")
            st.rerun()
        else:
            st.error("Please enter a valid username.")

    st.markdown("---")

    # --- Backend Connection ---
    st.markdown("### GPU BACKEND")
    st.markdown("""
    <div class="industrial-plate">
        <div style="color: #f96124; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.5rem;">
            REMOTE GPU BACKEND URL
        </div>
    </div>
    """, unsafe_allow_html=True)
    current_backend = st.session_state.get("backend_url", "")
    if current_backend:
        st.success(f"Connected: {current_backend}")
    else:
        st.info("Not connected — fine-tuning and inference require a GPU backend.")

    new_backend = st.text_input(
        "Backend URL",
        value=current_backend,
        placeholder="https://your-ngrok-url.ngrok-free.app",
        key="settings_backend_url",
        label_visibility="collapsed",
    )
    if st.button("SAVE BACKEND URL", type="primary", key="save_backend_url"):
        st.session_state.backend_url = new_backend.strip()
        # Persist to RAM cache so it survives browser refresh
        email = st.session_state.get("user_profile", {}).get("email", "")
        if email:
            _save_to_ram(email, "backend_url", new_backend.strip())
        st.success("Backend URL saved!")
        st.rerun()

    # Show user ID for backend setup
    profile = st.session_state.get("user_profile", {})
    if profile.get("id"):
        st.markdown(f"""
        <div style="margin-top: 0.5rem; color: #8c6b5d; font-size: 0.75rem;">
            Your User ID (for backend Docker setup): <code>{profile['id']}</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Account / Logout ---
    st.markdown("### ACCOUNT")
    user_display = "Operator"
    try:
        user_display = st.user.email
    except Exception:
        pass

    st.markdown(f"""
    <div class="industrial-plate">
        <div style="color: #ccc; font-size: 0.9rem;">Signed in as: <strong style="color: white;">{user_display}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    col_logout, _ = st.columns([1, 3])
    with col_logout:
        if st.button("LOGOUT", type="primary", use_container_width=True, key="settings_logout"):
            st.session_state.user_logged_in = False
            st.session_state.user_profile = {}
            st.session_state.datasets = {}
            st.session_state._datasets_loaded = False
            st.session_state.api_key = None
            st.session_state.backend_url = ""
            st.session_state.hf_token = ""
            try:
                st.logout()
            except Exception:
                st.rerun()

    st.markdown("---")

    # --- Danger Zone ---
    st.markdown("### DANGER ZONE")
    col_reset, _ = st.columns([1, 3])
    with col_reset:
        if st.button("RESET ALL SETTINGS", key="reset_all"):
            # Clear Supabase profile fields
            save_profile_field("backboard_api_key", None)
            save_profile_field("hf_token", None)
            save_profile_field("hf_username", None)
            st.session_state.api_key = None
            st.session_state.backend_url = ""
            st.session_state.hf_token = ""
            st.session_state.hf_username = ""
            st.session_state.validated_models = {}
            st.rerun()


# ==================== TOP NAVIGATION BAR ====================

def render_top_nav():
    """Render a persistent top navigation bar with buttons."""
    current_view = st.session_state.view

    nav_items = [
        ("DASHBOARD", "dashboard"),
        ("DATA", "data_viewer"),
        ("TRAIN", "fine_tuning"),
        ("INFERENCE", "inference"),
        ("DEPLOY", "hf_deploy"),
        ("SETTINGS", "settings"),
    ]

    cols = st.columns([3] + [1] * len(nav_items))

    # Left: user info + backend status
    with cols[0]:
        user_display = "Operator"
        try:
            user_display = st.user.email
        except Exception:
            pass
        backend_status = "CONNECTED" if st.session_state.backend_url else "NO GPU"
        backend_color = "#22c55e" if st.session_state.backend_url else "#555"
        api_status = "API OK" if st.session_state.api_key else "NO API KEY"
        api_color = "#22c55e" if st.session_state.api_key else "#f96124"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.75rem;padding-top:0.25rem;">'
            f'<span style="color:#e0e0e0;font-size:0.85rem;font-family:Rajdhani,sans-serif;font-weight:600;letter-spacing:0.05em;">{user_display}</span>'
            f'<span style="color:{api_color};font-size:0.65rem;font-family:JetBrains Mono,monospace;border:1px solid {api_color};padding:0.15rem 0.5rem;">{api_status}</span>'
            f'<span style="color:{backend_color};font-size:0.65rem;font-family:JetBrains Mono,monospace;border:1px solid {backend_color};padding:0.15rem 0.5rem;">GPU: {backend_status}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Nav buttons
    for i, (label, view_key) in enumerate(nav_items):
        with cols[i + 1]:
            btn_type = "primary" if current_view == view_key else "secondary"
            if st.button(label, key=f"topnav_{view_key}", type=btn_type, use_container_width=True):
                st.session_state.view = view_key
                st.rerun()

    st.markdown('<div style="background:#222;height:1px;margin-bottom:1rem;"></div>', unsafe_allow_html=True)


# ==================== MAIN APP ====================

def main():
    st.set_page_config(page_title="ModelForge", page_icon="", layout="wide", initial_sidebar_state="collapsed")

    load_forge_theme()
    init_session_state()

    # --- AUTHENTICATION (Google OAuth) ---
    if not st.session_state.get("user_logged_in"):
        # Check if user is already authenticated via st.login
        try:
            user_email = st.user.email
            if user_email:
                # Upsert profile in Supabase
                ensure_user_profile(user_email)
                st.session_state.user_logged_in = True
        except:
            pass

    if not st.session_state.get("user_logged_in"):
        _, col_center, _ = st.columns([1, 1.5, 1])
        with col_center:
            st.write("")
            st.write("")
            lc1, lc2, lc3 = st.columns([3, 1, 3])
            with lc2:
                try:
                    st.image("logo.png", use_container_width=True)
                except:
                    pass
            st.markdown("""
            <style>
                .mf-signin-title {
                    color: #f96124 !important;
                    -webkit-text-fill-color: #f96124;
                    font-size: 3.5rem;
                    font-weight: 900;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    line-height: 1;
                    text-shadow:
                        0 0 12px rgba(249, 97, 36, 0.65),
                        0 0 32px rgba(249, 97, 36, 0.52),
                        0 0 62px rgba(249, 97, 36, 0.38),
                        0 0 96px rgba(249, 97, 36, 0.26),
                        0 0 128px rgba(249, 97, 36, 0.16);
                    animation: mfTitleGlow 2.6s ease-in-out infinite, mfTitleFlicker 3.8s linear infinite;
                }

                @keyframes mfTitleGlow {
                    0%, 100% {
                        text-shadow:
                            0 0 12px rgba(249, 97, 36, 0.62),
                            0 0 30px rgba(249, 97, 36, 0.48),
                            0 0 56px rgba(249, 97, 36, 0.34),
                            0 0 88px rgba(249, 97, 36, 0.22),
                            0 0 116px rgba(249, 97, 36, 0.13);
                    }
                    50% {
                        text-shadow:
                            0 0 14px rgba(249, 97, 36, 0.95),
                            0 0 40px rgba(249, 97, 36, 0.72),
                            0 0 74px rgba(249, 97, 36, 0.48),
                            0 0 112px rgba(249, 97, 36, 0.32),
                            0 0 150px rgba(249, 97, 36, 0.2);
                    }
                }

                @keyframes mfTitleFlicker {
                    0%, 18%, 22%, 56%, 58%, 100% {
                        text-shadow:
                            0 0 12px rgba(249, 97, 36, 0.65),
                            0 0 32px rgba(249, 97, 36, 0.52),
                            0 0 62px rgba(249, 97, 36, 0.38),
                            0 0 96px rgba(249, 97, 36, 0.26),
                            0 0 128px rgba(249, 97, 36, 0.16);
                    }
                    20% {
                        text-shadow:
                            0 0 12px rgba(249, 97, 36, 0.6),
                            0 0 28px rgba(249, 97, 36, 0.44),
                            0 0 52px rgba(249, 97, 36, 0.29),
                            0 0 82px rgba(249, 97, 36, 0.19),
                            0 0 108px rgba(249, 97, 36, 0.12);
                    }
                    57% {
                        text-shadow:
                            0 0 12px rgba(249, 97, 36, 0.62),
                            0 0 30px rgba(249, 97, 36, 0.46),
                            0 0 56px rgba(249, 97, 36, 0.32),
                            0 0 86px rgba(249, 97, 36, 0.2),
                            0 0 112px rgba(249, 97, 36, 0.13);
                    }
                }
            </style>
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 class="mf-signin-title">
                    MODEL<br>FORGE
                </h1>
                <div style="background: linear-gradient(90deg, transparent, #f96124, transparent); height: 2px; width: 100%; margin: 1.5rem auto; opacity: 0.5;"></div>
                <p style="color: #8c6b5d; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.3em;">
                    // AUTHENTICATION_PROTOCOL //
                </p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Sign in with Google", type="primary", use_container_width=True):
                st.login("google")

            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #666; font-size: 0.75rem;">
                Or enter credentials manually:
            </div>
            """, unsafe_allow_html=True)

            with st.form("manual_login", border=True):
                st.markdown("### [MANUAL_ACCESS]")
                name = st.text_input("OPERATOR_ID", placeholder="Enter designation...")
                api_key = st.text_input("SECURITY_KEY", type="password", placeholder="sk-...")
                submitted = st.form_submit_button("[INITIALIZE_SYSTEM]", type="primary", use_container_width=True)
                if submitted:
                    if name and api_key:
                        email = f"{name.lower().replace(' ', '_')}@manual.local"
                        ensure_user_profile(email, display_name=name)
                        st.session_state.user_logged_in = True
                        st.session_state.api_key = api_key
                        save_profile_field("backboard_api_key", api_key)
                        st.rerun()
                    elif name:
                        email = f"{name.lower().replace(' ', '_')}@manual.local"
                        ensure_user_profile(email, display_name=name)
                        st.session_state.user_logged_in = True
                        st.rerun()
                    else:
                        st.error("[ACCESS_DENIED] Enter at least an Operator ID")
        st.stop()

    # Auto-discover backend on each page load
    profile = st.session_state.get("user_profile", {})
    if profile.get("id") and not st.session_state.backend_url:
        discovered = check_backend_connection(profile["id"])
        if discovered:
            st.session_state.backend_url = discovered
            email = profile.get("email", "")
            if email:
                _save_to_ram(email, "backend_url", discovered)

    # --- TOP NAVIGATION BAR ---
    render_top_nav()

    # --- ROUTE TO VIEW ---
    if st.session_state.view == 'dashboard':
        render_dashboard()
    elif st.session_state.view == 'data_generation':
        render_data_generation()
    elif st.session_state.view == 'data_viewer':
        render_data_viewer()
    elif st.session_state.view == 'fine_tuning':
        render_finetuning()
    elif st.session_state.view == 'inference':
        render_inference()
    elif st.session_state.view == 'hf_deploy':
        render_hf_deploy()
    elif st.session_state.view == 'settings':
        render_settings()


if __name__ == "__main__":
    main()
