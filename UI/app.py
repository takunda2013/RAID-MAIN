from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from textwrap import dedent, wrap
from typing import Iterable

import streamlit as st
import streamlit.components.v1 as components

from auth import hash_password
from auth import login as auth_login
from auth import register as auth_register
from db import (
    connect,
    count_users,
    create_document,
    create_user,
    delete_documents_for_user,
    get_document_for_user,
    get_user_by_id,
    initialize,
    list_documents_for_user,
    list_users,
    mark_document_processed,
    row_to_document,
    row_to_user,
    set_user_active,
    set_user_password_hash,
    set_user_role,
)

APP_VERSION = "v1.0"
MIN_WORDS = 100
APP_DIR = Path(__file__).resolve().parent
DOCUMENT_STORAGE_DIR = APP_DIR / "data" / "documents"
REPORT_STORAGE_DIR = APP_DIR / "data" / "reports"


@dataclass(frozen=True)
class DemoResult:
    name: str
    label: str
    confidence: int
    ai_probability: int
    summary: str
    explanation: str
    tokens: tuple[tuple[int, str, str, str], ...]
    metrics: tuple[tuple[str, str, str], ...]
    annotated_words: tuple[tuple[str, str], ...]


DEMO_RESULTS: dict[str, DemoResult] = {
    "Human-written essay": DemoResult(
        name="Human-written essay",
        label="Human-written",
        confidence=92,
        ai_probability=8,
        summary="This submission has been classified as likely human-written with high confidence.",
        explanation=(
            "The essay shows uneven sentence rhythm, locally specific phrasing, and varied transitions. "
            "Those signals are more consistent with human authorship than generated prose."
        ),
        tokens=(
            (1, "therefore", "+0.31", "Human"),
            (2, "although", "+0.28", "Human"),
            (3, "Macbeth", "+0.25", "Human"),
            (4, "hesitates", "+0.22", "Human"),
            (5, "however", "+0.19", "Human"),
        ),
        metrics=(
            ("Mean surprisal", "7.82", "Higher variation"),
            ("Semantic cluster", "Human region", "Stable"),
            ("Paraphrase similarity", "Low risk", "No strong match"),
            ("Classifier margin", "0.84", "High"),
        ),
        annotated_words=(
            ("John's", "human"),
            ("essay", "neutral"),
            ("on", "neutral"),
            ("Macbeth", "human"),
            ("argues", "human"),
            ("that", "neutral"),
            ("ambition", "human"),
            ("slowly", "human"),
            ("corrupts", "human"),
            ("moral", "human"),
            ("judgement", "human"),
            ("through", "neutral"),
            ("fear", "human"),
            ("and", "neutral"),
            ("pressure.", "human"),
        ),
    ),
    "AI-generated essay": DemoResult(
        name="AI-generated essay",
        label="AI-generated",
        confidence=95,
        ai_probability=95,
        summary="This submission has been classified as likely AI-generated with high confidence.",
        explanation=(
            "The text has unusually smooth transitions, compressed argument structure, and repeated "
            "high-level phrasing. The semantic embedding sits close to the AI-generated cluster."
        ),
        tokens=(
            (1, "Furthermore", "+0.38", "AI"),
            (2, "ultimately", "+0.34", "AI"),
            (3, "underscores", "+0.31", "AI"),
            (4, "complexity", "+0.29", "AI"),
            (5, "societal", "+0.25", "AI"),
        ),
        metrics=(
            ("Mean surprisal", "4.11", "Low variation"),
            ("Semantic cluster", "AI region", "Strong"),
            ("Paraphrase similarity", "Medium risk", "Template-like"),
            ("Classifier margin", "0.89", "High"),
        ),
        annotated_words=(
            ("Furthermore,", "ai"),
            ("the", "neutral"),
            ("essay", "neutral"),
            ("ultimately", "ai"),
            ("underscores", "ai"),
            ("the", "neutral"),
            ("complexity", "ai"),
            ("of", "neutral"),
            ("human", "neutral"),
            ("ambition", "neutral"),
            ("within", "neutral"),
            ("a", "neutral"),
            ("societal", "ai"),
            ("framework.", "ai"),
        ),
    ),
    "Adversarially paraphrased AI": DemoResult(
        name="Adversarially paraphrased AI",
        label="Adversarially paraphrased AI",
        confidence=98,
        ai_probability=91,
        summary=(
            "This submission has been classified as likely AI-authored after paraphrase-style "
            "rewriting."
        ),
        explanation=(
            "Surface wording looks less uniform, but semantic features remain close to generated "
            "examples while statistical features show paraphrase smoothing."
        ),
        tokens=(
            (1, "therefore", "+0.31", "AI/paraphrase"),
            (2, "pession", "+0.30", "AI/paraphrase"),
            (3, "fooseed", "+0.28", "AI/paraphrase"),
            (4, "rewerd", "+0.25", "AI/paraphrase"),
            (5, "status", "+0.23", "AI/paraphrase"),
        ),
        metrics=(
            ("Mean surprisal", "5.07", "Moderate"),
            ("Semantic cluster", "AI/paraphrase region", "Very strong"),
            ("Paraphrase similarity", "High risk", "Detected"),
            ("Classifier margin", "0.93", "Very high"),
        ),
        annotated_words=(
            ("The", "neutral"),
            ("original", "neutral"),
            ("essay", "neutral"),
            ("therefore", "ai"),
            ("uses", "neutral"),
            ("green", "human"),
            ("local", "human"),
            ("phrasing", "human"),
            ("but", "neutral"),
            ("retains", "ai"),
            ("red", "ai"),
            ("generated", "ai"),
            ("structure.", "ai"),
        ),
    ),
}


def configure_page() -> None:
    st.set_page_config(
        page_title="AI Essay Detector",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Editorial-suite redesign inspired by `main-ui.jpeg` while preserving the
    # existing Streamlit controls, authentication, reports, uploads, and results.
    vars_css = """
    :root {
        --bg: #f3f0e6;
        --bg-soft: #ebe6d8;
        --grid-line: rgba(103, 72, 49, 0.08);
        --panel: #fffdf8;
        --panel-strong: #ffffff;
        --panel-line: #ead7c7;
        --ink: #3f2b20;
        --ink-strong: #5b351f;
        --muted: #9b8d83;
        --muted-strong: #746254;
        --cream: #fff7e8;
        --green: #1f6f4a;
        --green-bg: rgba(31, 111, 74, 0.24);
        --red: #bf513f;
        --red-bg: rgba(191, 81, 63, 0.14);
        --amber: #d98738;
        --amber-bg: rgba(217, 135, 56, 0.16);
        --cyan: #b76d33;
        --brown: #5a341f;
        --brown-soft: #7a4b31;
        --shadow: 0 28px 80px rgba(91, 53, 31, 0.12);
        --radius-lg: 28px;
        --radius-md: 18px;
    }
    """
    css = (
        "<style>\n"
        + vars_css
        + """
        /* Hide Streamlit chrome (top bar + deploy/menu) */
        header[data-testid="stHeader"] { display: none; }
        [data-testid="stToolbar"] { display: none; }
        [data-testid="stDecoration"] { display: none; }
        #MainMenu { display: none; }
        footer { display: none; }
        /* Keep sidebar always present (disable collapse affordances) */
        div[data-testid="stSidebarCollapseButton"] { display: none; }
        div[data-testid="collapsedControl"] { display: none; }
        section[data-testid="stSidebar"] { min-width: 320px; max-width: 320px; }

        .stApp {
            background:
                radial-gradient(900px 420px at 18% 6%, rgba(17, 122, 115, 0.08), transparent 55%),
                radial-gradient(820px 420px at 92% 10%, rgba(176, 107, 0, 0.08), transparent 55%),
                linear-gradient(90deg, rgba(18, 22, 32, 0.05) 1px, transparent 1px),
                linear-gradient(rgba(18, 22, 32, 0.04) 1px, transparent 1px),
                var(--bg);
            background-size: auto, auto, 36px 36px, 36px 36px, auto;
            color: var(--ink);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, var(--bg-soft) 100%);
            border-right: 1px solid var(--panel-line);
            box-shadow: 14px 0 40px rgba(9, 12, 18, 0.06);
            transform: none !important;
            visibility: visible !important;
            min-width: 320px !important;
            max-width: 320px !important;
        }
        div[data-testid="stSidebar"] {
            transform: none !important;
            visibility: visible !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li {
            color: var(--ink);
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
            color: var(--muted-strong);
        }
        .main .block-container {
            padding-top: 1.4rem;
            max-width: 1160px;
        }
        h1, h2, h3, label {
            color: var(--ink) !important;
            letter-spacing: 0;
        }
        /* Ensure text inside containers uses readable ink */
        .box, .annotation, .data-table, .stat-card, .sidebar-brand { color: var(--ink); }
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span { color: var(--ink); }
        p, li, span, div {
            letter-spacing: 0;
        }
        .app-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border: 1px solid var(--panel-line);
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,255,255,0.88));
            box-shadow: var(--shadow);
            margin-bottom: 1.1rem;
            padding: 1.25rem 1.35rem;
            border-radius: 18px;
        }
        .app-title h1 {
            font-size: clamp(1.75rem, 4vw, 2.8rem);
            line-height: 1.05;
            margin: 0;
            font-weight: 850;
        }
        .title-copy {
            min-width: 0;
        }
        .eyebrow {
            color: var(--cyan);
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.38rem;
            text-transform: uppercase;
        }
        .subtitle {
            color: var(--muted);
            margin-top: 0.45rem;
            max-width: 720px;
            font-size: 0.98rem;
        }
        .version {
            background: rgba(17, 122, 115, 0.10);
            border: 1px solid rgba(17, 122, 115, 0.22);
            border-radius: 999px;
            color: var(--cyan);
            font-weight: 800;
            padding: 0.45rem 0.75rem;
            white-space: nowrap;
        }
        .section-title {
            color: var(--muted-strong);
            font-weight: 850;
            text-transform: uppercase;
            margin: 1rem 0 0.55rem;
            font-size: 0.84rem;
            letter-spacing: 0.08em;
        }
        .box {
            border: 1px solid var(--panel-line);
            background: rgba(255,255,255,0.94);
            box-shadow: 0 16px 40px rgba(9, 12, 18, 0.08);
            padding: 1rem;
            border-radius: 18px;
            margin-bottom: 1rem;
        }
        /* Make Streamlit `st.container(border=True)` match our panel styling */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--panel-line) !important;
            background: rgba(255,255,255,0.94) !important;
            box-shadow: 0 16px 40px rgba(9, 12, 18, 0.08) !important;
            border-radius: 18px !important;
            padding: 1rem !important;
            margin-bottom: 1rem !important;
        }
        .panel-box {
            border: 1px solid rgba(17, 122, 115, 0.18);
            background: rgba(255, 255, 255, 0.96);
            box-shadow: 0 8px 24px rgba(9, 12, 18, 0.06);
            padding: 1rem;
            border-radius: 18px;
            margin-bottom: 1rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(span.create-user-panel-marker),
        div[data-testid="stVerticalBlock"]:has(span.create-user-panel-marker) {
            border: 1px solid rgba(17, 122, 115, 0.18) !important;
            background: rgba(255, 255, 255, 0.96) !important;
            box-shadow: 0 8px 24px rgba(9, 12, 18, 0.06) !important;
            padding: 1rem !important;
            border-radius: 18px !important;
            margin-bottom: 1rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(span.admin-select-user-panel-marker),
        div[data-testid="stVerticalBlock"]:has(span.admin-select-user-panel-marker),
        [data-testid="stVerticalBlockBorderWrapper"] div:has(span.admin-select-user-panel-marker) {
            border: 1px solid rgba(17, 122, 115, 0.18) !important;
            background: rgba(255, 255, 255, 0.96) !important;
            box-shadow: 0 8px 24px rgba(9, 12, 18, 0.06) !important;
            padding: 1rem !important;
            border-radius: 18px !important;
            margin-bottom: 1rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(span.auth-panel-marker),
        div[data-testid="stVerticalBlock"]:has(span.auth-panel-marker) {
            border: 1px solid rgba(17, 122, 115, 0.18) !important;
            background: rgba(255, 255, 255, 0.96) !important;
            box-shadow: 0 8px 24px rgba(9, 12, 18, 0.06) !important;
            padding: 1rem !important;
            border-radius: 18px !important;
            margin-bottom: 1rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(span.educator-input-panel-marker),
        div[data-testid="stVerticalBlock"]:has(span.educator-input-panel-marker) {
            border: 1px solid rgba(17, 122, 115, 0.18) !important;
            background: rgba(255, 255, 255, 0.96) !important;
            box-shadow: 0 8px 24px rgba(9, 12, 18, 0.06) !important;
            padding: 1rem !important;
            border-radius: 18px !important;
            margin-bottom: 1rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(span.educator-results-panel-marker),
        div[data-testid="stVerticalBlock"]:has(span.educator-results-panel-marker) {
            border: 1px solid rgba(17, 122, 115, 0.18) !important;
            background: rgba(255, 255, 255, 0.96) !important;
            box-shadow: 0 8px 24px rgba(9, 12, 18, 0.06) !important;
            padding: 1rem !important;
            border-radius: 18px !important;
            margin-bottom: 1rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(span.educator-explanation-panel-marker),
        div[data-testid="stVerticalBlock"]:has(span.educator-explanation-panel-marker) {
            border: 1px solid rgba(17, 122, 115, 0.18) !important;
            background: rgba(255, 255, 255, 0.96) !important;
            box-shadow: 0 8px 24px rgba(9, 12, 18, 0.06) !important;
            padding: 1rem !important;
            border-radius: 18px !important;
            margin-bottom: 1rem !important;
        }
        /* Auth panel — target Streamlit's own block wrapper so no empty div appears */
        [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stTabs"]) {
            max-width: 380px;
            margin: 1rem auto 1.5rem;
            padding: 2rem 1.8rem 1.8rem;
            background: #ffffff;
            border: 1px solid rgba(17, 122, 115, 0.20) !important;
            border-radius: 28px !important;
            box-shadow:
                0 4px 6px rgba(15, 20, 35, 0.04),
                0 12px 28px rgba(15, 20, 35, 0.10),
                0 40px 80px rgba(15, 20, 35, 0.14);
        }
        /* Tab strip */
        [data-testid="stVerticalBlockBorderWrapper"] .stTabs [data-baseweb="tab-list"] {
            background: rgba(17,122,115,0.06);
            border-radius: 12px;
            padding: 3px;
            gap: 0;
        }
        [data-testid="stVerticalBlockBorderWrapper"] .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            font-weight: 700;
            font-size: 0.88rem;
            color: var(--muted);
            flex: 1;
            justify-content: center;
            background: rgba(18,22,32,0.08);
        }
        [data-testid="stVerticalBlockBorderWrapper"] .stTabs [data-baseweb="tab"]:hover {
            background: rgba(18,22,32,0.15);
        }
        [data-testid="stVerticalBlockBorderWrapper"] .stTabs [aria-selected="true"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            box-shadow: 0 2px 8px rgba(15,20,35,0.10);
        }
        /* Auth buttons — inactive: visible charcoal */
        [data-testid="stVerticalBlockBorderWrapper"] .stButton > button {
            width: 100% !important;
            min-height: 3rem !important;
            border-radius: 12px !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            background: #3a3f4a !important;
            color: #ffffff !important;
            border: none !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] .stButton > button:hover {
            background: #2b2f38 !important;
            color: #ffffff !important;
        }
        /* Active / primary button — teal-amber gradient */
        [data-testid="stVerticalBlockBorderWrapper"] .stButton > button[kind="primary"],
        [data-testid="stVerticalBlockBorderWrapper"] .stButton > button[kind="primary"]:not(:hover) {
            background: linear-gradient(135deg, #11867a, #b06b00) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 6px 20px rgba(17,122,115,0.28) !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #13957f, #c07800) !important;
            box-shadow: 0 8px 26px rgba(17,122,115,0.36) !important;
            color: #ffffff !important;
            border: none !important;
        }
        /* Inputs */
        [data-testid="stVerticalBlockBorderWrapper"] .stTextInput input {
            border-radius: 10px !important;
            border: 1.5px solid rgba(18,22,32,0.14) !important;
            font-size: 0.92rem !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] .stTextInput input:focus {
            border-color: rgba(17,122,115,0.50) !important;
            box-shadow: 0 0 0 3px rgba(17,122,115,0.10) !important;
        }
        .result-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border-bottom: 1px solid var(--panel-line);
            padding-bottom: 0.85rem;
            margin-bottom: 0.85rem;
        }
        .classification {
            font-size: 1rem;
            font-weight: 850;
            color: var(--ink);
        }
        .confidence {
            text-align: right;
            font-weight: 900;
            font-size: 2.35rem;
            line-height: 1;
            color: var(--amber);
        }
        .confidence span {
            display: block;
            font-size: 0.72rem;
            font-weight: 800;
            margin-top: 0.15rem;
            color: var(--muted);
            text-transform: uppercase;
        }
        .summary {
            color: var(--muted-strong);
            font-weight: 700;
            margin: 0.45rem 0 0;
        }
        .meter {
            height: 12px;
            background: rgba(18, 22, 32, 0.07);
            border: 1px solid rgba(18, 22, 32, 0.12);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 0.85rem;
        }
        .meter > div {
            height: 100%;
            background: linear-gradient(90deg, var(--green), var(--amber), var(--red));
            box-shadow: 0 0 22px rgba(176, 107, 0, 0.18);
        }
        .small-note {
            color: var(--muted);
            font-size: 0.83rem;
            font-weight: 650;
            margin-top: 0.25rem;
        }
        .annotation {
            border: 1px solid var(--panel-line);
            border-radius: 14px;
            padding: 1rem;
            background: rgba(246, 247, 251, 0.95);
            line-height: 1.8;
            font-weight: 650;
        }
        .mark {
            border-radius: 7px;
            padding: 0.08rem 0.25rem;
            border: 1px solid rgba(18, 22, 32, 0.12);
            white-space: nowrap;
        }
        .mark-human {
            color: var(--green);
            background: var(--green-bg);
        }
        .mark-ai {
            color: var(--red);
            background: var(--red-bg);
        }
        .mark-neutral {
            background: rgba(18, 22, 32, 0.06);
            color: var(--muted-strong);
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-bottom: 1rem;
        }
        .stat-card {
            border: 1px solid var(--panel-line);
            background: rgba(255,255,255,0.94);
            border-radius: 14px;
            padding: 0.9rem;
            min-height: 96px;
        }
        .stat-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .stat-value {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 900;
            line-height: 1.15;
            margin-top: 0.35rem;
        }
        .stat-caption {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            border: 1px solid var(--panel-line);
            border-radius: 14px;
            margin: 0.5rem 0 1rem;
        }
        .data-table th {
            background: rgba(18, 22, 32, 0.06);
            color: var(--muted-strong);
            font-size: 0.76rem;
            text-align: left;
            text-transform: uppercase;
            padding: 0.72rem;
        }
        .data-table td {
            border-top: 1px solid rgba(18, 22, 32, 0.08);
            color: var(--ink);
            padding: 0.72rem;
        }
        .insight {
            border: 1px solid rgba(17, 122, 115, 0.22);
            background: rgba(17, 122, 115, 0.08);
            border-radius: 14px;
            color: var(--muted-strong);
            padding: 0.9rem 1rem;
            margin-top: 0.75rem;
        }
        .sidebar-brand {
            border: 1px solid var(--panel-line);
            border-radius: 16px;
            padding: 1rem;
            background: rgba(255,255,255,0.78);
            margin-bottom: 1rem;
        }
        .sidebar-brand strong {
            color: var(--ink);
            display: block;
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }
        .sidebar-brand span {
            color: var(--muted);
            font-size: 0.85rem;
        }
        .sidebar-pill {
            display: inline-block;
            border: 1px solid rgba(17, 122, 115, 0.20);
            color: var(--cyan);
            background: rgba(17, 122, 115, 0.08);
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            margin: 0.18rem 0.18rem 0.18rem 0;
            padding: 0.32rem 0.55rem;
        }
        .stButton > button,
        .stDownloadButton > button {
            border: 1px solid var(--panel-line);
            border-radius: 11px;
            font-weight: 800;
            background: rgba(255,255,255,0.92);
            color: var(--ink);
            min-height: 2.85rem;
        }
        /* Prevent download button from capturing clicks across the page
           by keeping it positioned but with a low stacking context. */
        .stDownloadButton,
        .stDownloadButton > a,
        .stDownloadButton > button {
            position: relative;
            z-index: 1;
            pointer-events: auto;
        }
        /* Ensure normal buttons sit above download anchors so clicks hit them first */
        .stButton > button {
            position: relative;
            z-index: 3;
        }
        /* Disabled/inactive buttons should keep black text on light theme */
        .stButton > button:disabled,
        .stDownloadButton > button:disabled {
            color: #121620 !important;
            background: rgba(255,255,255,0.92) !important;
            border-color: rgba(20, 24, 31, 0.14) !important;
            opacity: 0.78;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: rgba(17, 122, 115, 0.38);
            color: var(--ink);
            background: rgba(17, 122, 115, 0.08);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, rgba(17, 122, 115, 0.92), rgba(176, 107, 0, 0.86));
            border-color: transparent;
            color: #ffffff;
            box-shadow: 0 14px 34px rgba(17, 122, 115, 0.16);
        }
        .stButton > button[kind="primary"]:hover {
            color: #ffffff;
            filter: brightness(1.05);
        }
        textarea, input, select, div[data-baseweb="select"] {
            border-radius: 12px !important;
            color: var(--ink) !important;
        }
        textarea, input, select {
            background: rgba(255,255,255,0.98) !important;
            border: 1px solid var(--panel-line) !important;
        }
        /* Ensure native option text is dark as well */
        select, select * , option {
            color: #121620 !important;
            -webkit-text-fill-color: #121620 !important;
            background-color: transparent !important;
        }
        /* Force all input/select text to black (some themes override with white) */
        textarea, input, select {
            color: #121620 !important;
            -webkit-text-fill-color: #121620 !important;
            caret-color: #121620 !important;
        }
        textarea:disabled, input:disabled, select:disabled {
            color: #121620 !important;
            -webkit-text-fill-color: #121620 !important;
            opacity: 0.85;
        }
        div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.98);
            border-color: var(--panel-line);
        }
        /* Force the currently-selected value + caret to render black */
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg,
        div[data-baseweb="select"] input {
            color: #121620 !important;
            fill: #121620 !important;
            -webkit-text-fill-color: #121620 !important;
        }
        /* BaseWeb select renders selected value inside role=combobox; force black there too */
        div[data-baseweb="select"] div[role="combobox"],
        div[data-baseweb="select"] div[role="combobox"] * {
            color: #121620 !important;
            -webkit-text-fill-color: #121620 !important;
            fill: #121620 !important;
        }
        /* Selectbox dropdown options */
        div[data-baseweb="select"] [data-baseweb="menu"] div,
        div[data-baseweb="select"] [role="option"],
        div[data-baseweb="select"] [role="option"] * {
            color: #121620 !important;
            -webkit-text-fill-color: #121620 !important;
        }
        /* Force selectbox value text and all nested elements to black */
        [data-testid="stSelectbox"] span,
        [data-testid="stSelectbox"] div,
        [data-testid="stSelectbox"] [role="combobox"] * {
            color: #121620 !important;
            -webkit-text-fill-color: #121620 !important;
        }
        div[data-baseweb="select"] [data-baseweb="menu"] div:hover {
            background: rgba(17,122,115,0.08) !important;
        }
        textarea::placeholder, input::placeholder { color: rgba(18, 22, 32, 0.45) !important; }

        /* Tab labels (Login / Register) should be black */
        button[role="tab"] p,
        button[role="tab"] span,
        button[role="tab"] {
            color: #121620 !important;
        }

        /* Radio labels (Educator Page / User Management) should be black */
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] label span,
        div[data-testid="stRadio"] p,
        div[data-testid="stRadio"] span {
            color: #121620 !important;
        }
        [data-testid="stFileUploader"] section {
            background: rgba(0, 0, 0, 0.16);
            border: 1px dashed rgba(122, 215, 209, 0.34);
            border-radius: 14px;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span {
            color: var(--muted) !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 14px;
            background: rgba(242, 189, 87, 0.12);
            color: var(--ink);
        }
        hr {
            border-color: var(--panel-line);
        }
        /* Editorial Engine visual refresh */
        .stApp {
            background:
                linear-gradient(90deg, var(--grid-line) 1px, transparent 1px),
                linear-gradient(var(--grid-line) 1px, transparent 1px),
                radial-gradient(900px 460px at 18% 6%, rgba(217, 135, 56, 0.10), transparent 58%),
                radial-gradient(900px 460px at 88% 8%, rgba(91, 53, 31, 0.08), transparent 58%),
                var(--bg) !important;
            background-size: 64px 64px, 64px 64px, auto, auto, auto !important;
        }
        .main .block-container {
            max-width: 1220px;
            padding: 3rem 3rem 4rem;
        }
        section[data-testid="stSidebar"] {
            background: rgba(255, 253, 248, 0.82) !important;
            backdrop-filter: blur(14px);
            border-right: 1px solid var(--panel-line) !important;
            box-shadow: 18px 0 60px rgba(91, 53, 31, 0.08) !important;
        }
        .sidebar-brand {
            background: transparent !important;
            border: 0 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            padding: 0.35rem 0 1.25rem !important;
            border-bottom: 1px solid var(--panel-line) !important;
        }
        .sidebar-logo-row {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            color: var(--ink-strong);
            font-weight: 950;
            letter-spacing: -0.04em;
            text-transform: uppercase;
        }
        .sidebar-logo-mark {
            display: grid;
            place-items: center;
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 0.55rem;
            color: var(--amber);
            background: var(--cream);
            border: 1px solid var(--panel-line);
        }
        .sidebar-logo-row em {
            color: var(--amber);
            font-style: normal;
        }
        .sidebar-brand > span {
                    color: var(--muted-strong) !important;
                    display: block;
                    font-size: 0.78rem;
                    font-weight: 750;
                    letter-spacing: 0.08em;
                    margin-top: 0.55rem;
                    text-transform: uppercase;
                }
        .sidebar-pill {
            border-color: rgba(217, 135, 56, 0.28) !important;
            color: var(--brown) !important;
            background: rgba(217, 135, 56, 0.12) !important;
        }
        .editorial-topbar {
            align-items: center;
            display: flex;
            justify-content: space-between;
            margin-bottom: 3rem;
        }
        .brand-lockup {
            align-items: center;
            display: flex;
            gap: 0.65rem;
            color: var(--ink-strong);
            font-size: 1.05rem;
            font-weight: 950;
            letter-spacing: -0.045em;
            text-transform: uppercase;
        }
        .brand-lockup em {
            color: var(--amber);
            font-style: normal;
        }
        .brand-icon {
            align-items: center;
            background: var(--cream);
            border: 1px solid var(--panel-line);
            border-radius: 0.55rem;
            color: var(--amber);
            display: inline-flex;
            height: 2rem;
            justify-content: center;
            width: 2rem;
        }
        .suite-label {
            color: var(--muted-strong);
            font-size: 0.73rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .app-title {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            display: grid !important;
            grid-template-columns: minmax(0, 1.55fr) minmax(320px, 1fr);
            gap: 3.5rem;
            margin: 0 0 2rem !important;
            padding: 0 !important;
        }
        .hero-panel {
            background: var(--brown);
            border: 1px solid rgba(255, 247, 232, 0.16);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow);
            color: #fff7e8;
            min-height: 255px;
            padding: clamp(1.6rem, 3vw, 2.4rem);
        }
        .hero-panel .eyebrow {
            color: rgba(255, 247, 232, 0.72) !important;
            font-size: 0.75rem;
            letter-spacing: 0.22em;
            margin-bottom: 1.1rem;
        }
        .hero-panel h1 {
            color: #fff7e8 !important;
            font-size: clamp(2.4rem, 6vw, 4.3rem) !important;
            letter-spacing: -0.06em;
            line-height: 0.98 !important;
            max-width: 760px;
        }
        .hero-panel .subtitle {
            color: rgba(255, 247, 232, 0.76) !important;
            font-size: 1.08rem;
            font-weight: 650;
            line-height: 1.55;
            margin-top: 1.45rem;
            max-width: 860px;
        }
        .status-panel {
            align-items: center;
            background: rgba(255, 253, 248, 0.9);
            border: 2px solid var(--panel-line);
            border-radius: var(--radius-lg);
            box-shadow: 0 18px 52px rgba(91, 53, 31, 0.07);
            display: flex;
            justify-content: center;
            min-height: 220px;
            padding: 2rem;
            text-align: center;
        }
        .status-icon {
            color: rgba(217, 135, 56, 0.34);
            font-size: 2rem;
            margin-bottom: 0.75rem;
        }
        .status-kicker {
            color: var(--muted-strong);
            font-size: 0.82rem;
            font-weight: 950;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .status-copy {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.8rem;
        }
        .status-score {
            color: var(--brown);
            font-size: 3.4rem;
            font-weight: 950;
            letter-spacing: -0.08em;
            line-height: 1;
            margin: 0.8rem 0 0.25rem;
        }
        .box,
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 253, 248, 0.92) !important;
            border: 2px solid var(--panel-line) !important;
            border-radius: var(--radius-md) !important;
            box-shadow: 0 18px 50px rgba(91, 53, 31, 0.07) !important;
        }
        .section-title {
            color: var(--ink-strong) !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.04em !important;
            margin-top: 1.35rem !important;
        }
        textarea,
        input,
        select,
        div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.96) !important;
            border: 1.5px solid var(--panel-line) !important;
            border-radius: 14px !important;
            color: var(--ink) !important;
        }
        [data-testid="stFileUploader"] section {
            background: rgba(255, 247, 232, 0.72) !important;
            border: 1.5px dashed rgba(217, 135, 56, 0.36) !important;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-color: var(--panel-line) !important;
            border-radius: 12px !important;
            color: var(--ink-strong) !important;
            font-weight: 900 !important;
        }
        .stButton > button[kind="primary"] {
            background: var(--brown) !important;
            box-shadow: 0 14px 30px rgba(91, 53, 31, 0.18) !important;
            color: #fff7e8 !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--brown-soft) !important;
        }
        .result-head {
            border-bottom-color: var(--panel-line) !important;
        }
        .confidence,
        .stat-value {
            color: var(--brown) !important;
        }
        .meter {
            background: rgba(91, 53, 31, 0.08) !important;
            border-color: var(--panel-line) !important;
        }
        .meter > div {
            background: linear-gradient(90deg, var(--green), var(--amber), var(--red)) !important;
        }
        .annotation,
        .stat-card,
        .data-table {
            background: rgba(255, 247, 232, 0.52) !important;
            border-color: var(--panel-line) !important;
        }
        .insight {
            background: rgba(217, 135, 56, 0.12) !important;
            border-color: rgba(217, 135, 56, 0.28) !important;
        }
        @media (max-width: 980px) {
            .app-title {
                grid-template-columns: 1fr;
                gap: 1rem;
            }
            .editorial-topbar {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.75rem;
                margin-bottom: 1.5rem;
            }
            .main .block-container {
                padding: 1.5rem 1rem 3rem;
            }
        }
        @media (max-width: 760px) {
            .app-title {
                align-items: flex-start;
                flex-direction: column;
            }
            .result-head {
                align-items: flex-start;
                flex-direction: column;
            }
            .confidence {
                text-align: left;
            }
            .stat-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """
    )
    st.markdown(css, unsafe_allow_html=True)


def get_conn():
    conn = st.session_state.get("_db_conn")
    if conn is None:
        conn = connect()
        initialize(conn)
        st.session_state["_db_conn"] = conn
    return conn


def ensure_bootstrap_admin(conn) -> None:
    if count_users(conn) > 0:
        return

    create_user(
        conn,
        email="admin@local.com",
        display_name="Super Admin",
        password_hash=hash_password("admin1234"),
        role="super_admin",
        actor_user_id=None,
    )


def auth_current_user(conn):
    user_id = st.session_state.get("auth_user_id")
    if not user_id:
        return None
    row = get_user_by_id(conn, int(user_id))
    return row_to_user(row) if row else None


def auth_logout() -> None:
    st.session_state["auth_user_id"] = None
    st.session_state["auth_role"] = None
    st.session_state["auth_email"] = None
    st.session_state["auth_name"] = None
    st.session_state["result"] = None
    st.session_state["source_text"] = ""
    st.session_state["file_extract_failed"] = False
    st.session_state["report_ready"] = False


def inject_autocomplete_hints() -> None:
    # Streamlit doesn't expose autocomplete attributes; patch them client-side
    # so browser autofill works and warnings go away.
    components.html(
        """
        <script>
        (function () {
          function norm(s) { return (s || "").toLowerCase().trim(); }
          function ensureIdName(el, idx) {
            if (!el.id) el.id = "st_field_" + idx + "_" + Math.random().toString(16).slice(2);
            if (!el.name) el.name = el.id;
          }
          function linkLabel(el) {
            // Streamlit renders labels near inputs, but audits sometimes fail to detect association.
            // Try to find a nearby <label> element and attach a matching "for".
            const container = el.closest('[data-testid="stTextInput"],[data-testid="stTextArea"],[data-testid="stSelectbox"],[data-testid="stNumberInput"],[data-testid="stDateInput"],[data-testid="stTimeInput"],[data-testid="stFileUploader"],[data-testid="stForm"]') || el.parentElement;
            if (!container) return;
            const label = container.querySelector("label");
            if (label && !label.getAttribute("for")) label.setAttribute("for", el.id);
          }
          function setAuto(el, value) {
            const current = el.getAttribute("autocomplete");
            if (!current || current === "") el.setAttribute("autocomplete", value);
          }
          function inferAutocomplete(el) {
            const aria = norm(el.getAttribute("aria-label"));
            const ph = norm(el.getAttribute("placeholder"));
            const type = norm(el.getAttribute("type"));
            const hint = aria + " " + ph;

            if (hint.includes("email")) return "email";
            if (type === "password" || hint.includes("password")) {
              if (hint.includes("temporary") || hint.includes("set new") || hint.includes("new password")) return "new-password";
              // login password
              return "current-password";
            }
            if (hint.includes("display name") || hint === "name") return "name";
            // avoid empty autocomplete attribute warnings on other fields
            return "off";
          }

          const doc = window.parent.document;
          const fields = Array.from(doc.querySelectorAll("input, textarea"));
          fields.forEach((el, idx) => {
            if (!el) return;
            ensureIdName(el, idx);
            linkLabel(el);
            setAuto(el, inferAutocomplete(el));
          });
        })();
        </script>
        """,
        height=0,
    )


def render_auth_gate(conn) -> None:
    inject_autocomplete_hints()
    with st.container(border=True):
        st.markdown('<span class="auth-panel-marker"></span>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Account</div>', unsafe_allow_html=True)
        st.caption("You must be logged in to access the educator page.")

        tabs = st.tabs(["Login", "Register"])

        with tabs[0]:
            st.text_input("Email", key="login_email", placeholder="you@school.edu")
            st.text_input("Password", key="login_password", type="password", placeholder="Your password")

            def _do_login() -> None:
                res = auth_login(
                    conn,
                    email=st.session_state.get("login_email", ""),
                    password=st.session_state.get("login_password", ""),
                )
                if not res.ok or not res.user:
                    st.session_state["auth_error"] = res.error or "Login failed."
                    return
                st.session_state["auth_user_id"] = res.user.id
                st.session_state["auth_role"] = res.user.role
                st.session_state["auth_email"] = res.user.email
                st.session_state["auth_name"] = res.user.display_name
                st.session_state["auth_error"] = None

            st.button("Login", type="primary", use_container_width=True, key="btn_login", on_click=_do_login)
            if st.session_state.get("auth_error"):
                st.error(st.session_state["auth_error"])

            st.caption("First-run default super admin: admin@local.com / admin1234")

        with tabs[1]:
            st.text_input("Display name", key="reg_name", placeholder="Jane Doe")
            st.text_input("Email", key="reg_email", placeholder="you@school.edu")
            st.text_input("Password", key="reg_password", type="password", placeholder="At least 8 characters")

            def _do_register() -> None:
                res = auth_register(
                    conn,
                    email=st.session_state.get("reg_email", ""),
                    display_name=st.session_state.get("reg_name", ""),
                    password=st.session_state.get("reg_password", ""),
                    role="educator",
                    actor_user_id=None,
                )
                if not res.ok:
                    st.session_state["reg_error"] = res.error or "Registration failed."
                    return
                st.session_state["reg_error"] = None
                st.success("Account created. You can log in now.")

            st.button("Register", type="primary", use_container_width=True, key="btn_register", on_click=_do_register)
            if st.session_state.get("reg_error"):
                st.error(st.session_state["reg_error"])


def render_super_admin_portal(conn, current_user) -> None:
    inject_autocomplete_hints()
    st.markdown('<div class="section-title">Super Admin Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Manage users, roles, and access.</div>', unsafe_allow_html=True)

    st.session_state.setdefault("admin_show_create_user", False)
    col_left, _ = st.columns([1, 3])
    with col_left:
        def _toggle_create_form():
            st.session_state["admin_show_create_user"] = not st.session_state["admin_show_create_user"]

        btn_label = "Hide Create Form" if st.session_state["admin_show_create_user"] else "Create User"
        st.button(btn_label, type="secondary", use_container_width=False, key="btn_toggle_create_user", on_click=_toggle_create_form)

    if st.session_state.get("admin_show_create_user", False):
        with st.container(border=True):
            st.markdown('<span class="create-user-panel-marker"></span>', unsafe_allow_html=True)
            st.text_input("Display name", key="create_name", placeholder="New user name")
            st.text_input("Email", key="create_email", placeholder="new.user@school.edu")
            st.selectbox("Role", options=["educator", "super_admin"], key="create_role", index=0)
            st.text_input(
                "Temporary password",
                key="create_password",
                type="password",
                placeholder="At least 8 characters",
            )

            def _create_user() -> None:
                res = auth_register(
                    conn,
                    email=st.session_state.get("create_email", ""),
                    display_name=st.session_state.get("create_name", ""),
                    password=st.session_state.get("create_password", ""),
                    role=st.session_state.get("create_role", "educator"),
                    actor_user_id=current_user.id if current_user else None,
                )
                if not res.ok:
                    st.session_state["create_user_msg"] = res.error or "Failed to create user."
                    st.session_state["create_user_ok"] = False
                    return
                st.session_state["create_user_msg"] = "User created."
                st.session_state["create_user_ok"] = True

            st.button("Submit New User", type="primary", use_container_width=True, key="btn_create_user", on_click=_create_user)
            if st.session_state.get("create_user_msg"):
                if st.session_state.get("create_user_ok"):
                    st.success(st.session_state["create_user_msg"])
                else:
                    st.error(st.session_state["create_user_msg"])

    rows = list_users(conn)
    users = [row_to_user(r) for r in rows]
    if not users:
        st.info("No users found.")
        return

    user_map = {u.email: u for u in users}
    st.session_state["admin_user_map"] = {
        email: {
            "id": u.id,
            "email": u.email,
            "display_name": u.display_name,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at,
            "last_login_at": u.last_login_at,
        }
        for email, u in user_map.items()
    }

    def _label(u) -> str:
        status = "Active" if u.is_active else "Disabled"
        return f"{u.display_name} ({u.email}) [{u.role}, {status}]"

    def _on_admin_user_change() -> None:
        selected = st.session_state.get("admin_selected_email")
        info = (st.session_state.get("admin_user_map") or {}).get(selected)
        if not info:
            return
        st.session_state["admin_role"] = info["role"]
        st.session_state["admin_active"] = bool(info["is_active"])
        st.session_state["admin_new_password"] = ""
        st.session_state["admin_msg"] = None

    with st.container(border=True):
        st.markdown('<span class="admin-select-user-panel-marker"></span>', unsafe_allow_html=True)
        selected_email = st.selectbox(
            "Select user",
            options=[u.email for u in users],
            format_func=lambda e: _label(next(u for u in users if u.email == e)),
            key="admin_selected_email",
            on_change=_on_admin_user_change,
        )
        selected_email = str(selected_email)
        selected_user = user_map[selected_email]

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.text_input("User id", value=str(selected_user.id), disabled=True)
            st.text_input("Email", value=selected_user.email, disabled=True)
        with c2:
            st.text_input("Display name", value=selected_user.display_name, disabled=True)
            new_role = str(
                st.selectbox(
                    "Role",
                    options=["educator", "super_admin"],
                    index=0 if selected_user.role == "educator" else 1,
                    key="admin_role",
                )
            )
        with c3:
            is_active = st.checkbox("Active", value=selected_user.is_active, key="admin_active")
            st.caption(f"Created: {selected_user.created_at}")
            st.caption(f"Last login: {selected_user.last_login_at or '—'}")

        new_password = st.text_input(
            "Set new password (optional)",
            type="password",
            key="admin_new_password",
            placeholder="Leave blank to keep unchanged",
        )

        def _apply_user_changes() -> None:
            target_id = selected_user.id
            actor_id = current_user.id if current_user else None

            if current_user and current_user.id == target_id and not is_active:
                st.session_state["admin_msg"] = "You cannot disable your own account."
                return

            if new_role != selected_user.role:
                set_user_role(conn, user_id=target_id, role=new_role, actor_user_id=actor_id)
            if is_active != selected_user.is_active:
                set_user_active(conn, user_id=target_id, is_active=is_active, actor_user_id=actor_id)
            if new_password.strip():
                set_user_password_hash(
                    conn,
                    user_id=target_id,
                    password_hash=hash_password(new_password.strip()),
                    actor_user_id=actor_id,
                )

            st.session_state["admin_msg"] = "Changes saved."

        st.button("Save Changes", type="primary", use_container_width=True, key="admin_save", on_click=_apply_user_changes)
        if st.session_state.get("admin_msg"):
            st.info(st.session_state["admin_msg"])


def count_words(text: str) -> int:
    return len([word for word in text.strip().split() if word])


def extract_uploaded_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]
    data = uploaded_file.getvalue()

    if suffix == "txt":
        return data.decode("utf-8", errors="ignore")

    if suffix == "pdf":
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]

            reader = PdfReader(BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception:
            return ""

    if suffix == "docx":
        try:
            from docx import Document  # type: ignore[import-not-found]

            document = Document(BytesIO(data))
            return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
        except Exception:
            return ""

    return ""


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "document"


def user_document_dir(user_id: int) -> Path:
    path = DOCUMENT_STORAGE_DIR / f"user_{int(user_id)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_report_dir(user_id: int) -> Path:
    path = REPORT_STORAGE_DIR / f"user_{int(user_id)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_file(path: str | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def remove_file_if_exists(path: str | None) -> None:
    if not path:
        return
    try:
        file_path = Path(path)
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
    except Exception:
        return


def demo_result_for_document(document_id: int, preferred_scenario: str) -> DemoResult:
    scenarios = list(DEMO_RESULTS.keys())
    if preferred_scenario in DEMO_RESULTS:
        index = (document_id - 1) % len(scenarios)
        # Keep the selected scenario for the first item, then rotate the rest so
        # bulk scans visibly process different demo outcomes one at a time.
        if index == 0:
            return DEMO_RESULTS[preferred_scenario]
        return DEMO_RESULTS[scenarios[index]]
    return DEMO_RESULTS[scenarios[(document_id - 1) % len(scenarios)]]


def default_demo_text(name: str) -> str:
    if name == "AI-generated essay":
        return (
            "Furthermore, the essay ultimately underscores the complexity of ambition by "
            "presenting Macbeth as a figure whose moral decline reflects broader social and "
            "psychological pressures."
        )
    if name == "Adversarially paraphrased AI":
        return (
            "The original essay text is reproduced with therefore-like phrasing and small "
            "word substitutions, yet it keeps the same generated structure beneath the surface."
        )
    return (
        "John's essay on Macbeth argues that ambition slowly corrupts moral judgement through "
        "fear, pressure, and the choices characters make when they know something is wrong."
    )


def escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_demo_pdf(result: DemoResult, source_text: str, title: str = "AI Essay Detector Demo Report") -> bytes:
    lines = [
        title,
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Classification: {result.label}",
        f"Confidence: {result.confidence}%",
        f"AI probability: {result.ai_probability}%",
        "",
        "Summary:",
        result.summary,
        "",
        "Explanation:",
        result.explanation,
        "",
        "Submitted text preview:",
        source_text[:800] or "No text supplied.",
    ]

    content_lines: list[str] = ["BT", "/F1 12 Tf", "50 760 Td", "14 TL"]
    first_line = True
    for line in lines:
        wrapped = wrap(line, width=82) or [""]
        for wrapped_line in wrapped:
            text = escape_pdf_text(wrapped_line)
            if first_line:
                content_lines.append(f"({text}) Tj")
                first_line = False
            else:
                content_lines.append(f"T* ({text}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def store_uploaded_document(conn, *, user_id: int, uploaded_file, demo_scenario: str) -> int | None:
    extracted_text = extract_uploaded_text(uploaded_file)
    if not extracted_text:
        return None

    storage_dir = user_document_dir(user_id)
    original_name = getattr(uploaded_file, "name", "document.txt")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    text_filename = f"{timestamp}_{safe_filename(original_name)}.txt"
    text_path = storage_dir / text_filename
    text_path.write_text(extracted_text, encoding="utf-8")

    return create_document(
        conn,
        user_id=user_id,
        original_filename=original_name,
        stored_path=str(text_path),
        demo_scenario=demo_scenario,
        word_count=count_words(extracted_text),
    )


def store_pasted_document(conn, *, user_id: int, text: str, demo_scenario: str) -> int | None:
    if not text.strip():
        return None

    storage_dir = user_document_dir(user_id)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    text_path = storage_dir / f"{timestamp}_pasted_essay.txt"
    text_path.write_text(text.strip(), encoding="utf-8")

    return create_document(
        conn,
        user_id=user_id,
        original_filename=f"Pasted essay {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
        stored_path=str(text_path),
        demo_scenario=demo_scenario,
        word_count=count_words(text),
    )


def process_uploaded_documents(conn, *, user_id: int) -> int:
    pending_rows = list_documents_for_user(conn, user_id=user_id, status="uploaded")
    processed_count = 0
    for row in pending_rows:
        document = row_to_document(row)
        source_text = read_text_file(document.stored_path)
        result = demo_result_for_document(document.id, document.demo_scenario)
        report_path = user_report_dir(user_id) / f"report_document_{document.id}.pdf"
        report_path.write_bytes(
            make_demo_pdf(
                result,
                source_text,
                title=f"AI Detector Demo Report - {document.original_filename}",
            )
        )
        mark_document_processed(
            conn,
            user_id=user_id,
            document_id=document.id,
            report_path=str(report_path),
            label=result.label,
            confidence=result.confidence,
            ai_probability=result.ai_probability,
        )
        processed_count += 1
    return processed_count


def delete_document_records(conn, *, user_id: int, document_ids: Iterable[int]) -> int:
    rows = delete_documents_for_user(conn, user_id=user_id, document_ids=document_ids)
    for row in rows:
        document = row_to_document(row)
        remove_file_if_exists(document.stored_path)
        remove_file_if_exists(document.report_path)
    return len(rows)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-logo-row">
                    <span class="sidebar-logo-mark">📄</span>
                    <span>Originality<em>Engine</em></span>
                </div>
                <span>Static detector demo console.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        user_name = st.session_state.get("auth_name")
        user_role = st.session_state.get("auth_role")
        if user_name:
            st.markdown(f"**Signed in:** {html.escape(str(user_name))}")
            if user_role:
                st.markdown(
                    f"<span class='sidebar-pill'>{html.escape(str(user_role))}</span>",
                    unsafe_allow_html=True,
                )
            st.button("Logout", use_container_width=True, key="btn_logout", on_click=auth_logout)
        else:
            st.markdown("<span class='sidebar-pill'>Not signed in</span>", unsafe_allow_html=True)

        st.markdown("### Detection Modes")
        st.markdown(
            """
            <span class="sidebar-pill">Human</span>
            <span class="sidebar-pill">AI</span>
            <span class="sidebar-pill">Paraphrased AI</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Brief Guide")
        st.markdown(
            """
            - Upload a PDF, DOCX, or TXT essay.
            - Or paste text directly into the review area.
            - Run the demo analysis to preview the final workflow.
            """
        )
        st.divider()
        st.markdown("### Reports")
        result = st.session_state.get("result")
        source_text = st.session_state.get("source_text", "")
        st.session_state.setdefault("report_ready", False)

        def _prepare_report() -> None:
            st.session_state["report_ready"] = True

        if not result:
            st.button(
                "Download Report (.pdf)",
                disabled=True,
                use_container_width=True,
                key="download_report_disabled",
            )
        else:
            if not st.session_state.get("report_ready"):
                st.button(
                    "Prepare Report",
                    use_container_width=True,
                    key="btn_prepare_report",
                    on_click=_prepare_report,
                )
                st.caption("Generate a report for the current result, then download.")
            else:
                st.download_button(
                    "Download Report (.pdf)",
                    data=make_demo_pdf(result, source_text),
                    file_name="ai_detector_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_report",
                )


def render_title() -> None:
    result = st.session_state.get("result")
    if result:
        status_markup = dedent(
            f"""
            <div>
                <div class="status-icon">📜</div>
                <div class="status-kicker">Analysis complete</div>
                <div class="status-score">{result.confidence}%</div>
                <div class="status-copy">{html.escape(result.label)} · {result.ai_probability}% AI probability</div>
            </div>
            """
        ).strip()
    else:
        status_markup = dedent(
            """
            <div>
                <div class="status-icon">📜</div>
                <div class="status-kicker">Awaiting manuscript</div>
                <div class="status-copy">Submit content to generate an authorship integrity report.</div>
            </div>
            """
        ).strip()

    st.markdown(
        dedent(
            f"""
            <div class="editorial-topbar">
                <div class="brand-lockup">
                    <span class="brand-icon">📄</span>
                    <span>Originality<em>Engine</em></span>
                </div>
                <div class="suite-label">Editorial Integrity Suite · {APP_VERSION}</div>
            </div>
            <div class="app-title">
                <div class="hero-panel">
                    <div class="eyebrow">The Academic Standard</div>
                    <h1>Safeguarding the Integrity of Written Work</h1>
                    <div class="subtitle">
                        Verify manuscript authenticity with the same detector workflow: paste text, upload files,
                        run the demo analysis, review explanations, and export a PDF report.
                    </div>
                </div>
                <div class="status-panel">
                    {status_markup} 
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_input(conn, current_user) -> tuple[str, str]:
    st.markdown('<div class="section-title">Bulk Scan Intake</div>', unsafe_allow_html=True)
    st.markdown('<span class="educator-input-panel-marker"></span>', unsafe_allow_html=True)
    with st.container(border=False):
        st.markdown('<div class="box">', unsafe_allow_html=True)

        def _ensure_defaults() -> None:
            st.session_state.setdefault("demo_scenario", list(DEMO_RESULTS.keys())[0])
            st.session_state.setdefault("essay_text", default_demo_text(st.session_state["demo_scenario"]))
            st.session_state.setdefault("result", None)
            st.session_state.setdefault("source_text", "")
            st.session_state.setdefault("file_extract_failed", False)
            st.session_state.setdefault("batch_scan_message", "")

        def _on_demo_change() -> None:
            scenario = st.session_state.get("demo_scenario", list(DEMO_RESULTS.keys())[0])
            st.session_state["essay_text"] = default_demo_text(scenario)
            st.session_state["result"] = None

        def _on_analyze() -> None:
            scenario = str(st.session_state.get("demo_scenario", list(DEMO_RESULTS.keys())[0]))
            uploaded_files = st.session_state.get("upload_files") or []
            added_count = 0
            failed_files: list[str] = []

            for uploaded in uploaded_files:
                document_id = store_uploaded_document(
                    conn,
                    user_id=current_user.id,
                    uploaded_file=uploaded,
                    demo_scenario=scenario,
                )
                if document_id is None:
                    failed_files.append(getattr(uploaded, "name", "uploaded file"))
                else:
                    added_count += 1

            if not uploaded_files:
                document_id = store_pasted_document(
                    conn,
                    user_id=current_user.id,
                    text=str(st.session_state.get("essay_text", "")),
                    demo_scenario=scenario,
                )
                if document_id is not None:
                    added_count += 1

            processed_count = process_uploaded_documents(conn, user_id=current_user.id)
            st.session_state["result"] = DEMO_RESULTS[scenario]
            st.session_state["source_text"] = st.session_state.get("essay_text", "")
            st.session_state["report_ready"] = False
            st.session_state["file_extract_failed"] = bool(failed_files)
            st.session_state["batch_scan_message"] = (
                f"Added {added_count} document(s) and processed {processed_count} document(s) one at a time."
            )
            if failed_files:
                st.session_state["batch_scan_message"] += " Could not extract: " + ", ".join(failed_files)

        def _on_clear() -> None:
            scenario = st.session_state.get("demo_scenario", list(DEMO_RESULTS.keys())[0])
            st.session_state["essay_text"] = default_demo_text(scenario)
            st.session_state["result"] = None
            st.session_state["source_text"] = ""
            st.session_state["file_extract_failed"] = False
            st.session_state["report_ready"] = False
            st.session_state["batch_scan_message"] = ""

        _ensure_defaults()

        demo_name = str(
            st.selectbox(
                "Demo scenario",
                options=list(DEMO_RESULTS.keys()),
                key="demo_scenario",
                on_change=_on_demo_change,
                help="Static demo output to show while the model integration is pending.",
            )
        )

        uploaded_files = st.file_uploader(
            "Upload essay files for bulk scanning",
            type=("pdf", "docx", "txt"),
            accept_multiple_files=True,
            key="upload_files",
            help="Upload many documents; the demo scanner stores them and processes them sequentially.",
        )

        if uploaded_files:
            st.markdown(
                f'<div class="small-note">Queued from uploader: {len(uploaded_files)} file(s). Click Analyse Essays to store and scan them.</div>',
                unsafe_allow_html=True,
            )

        if st.session_state.get("file_extract_failed"):
            st.warning("Some files could not be extracted. Supported formats are PDF, DOCX, and TXT.")

        essay_text = str(
            st.text_area(
                "Optional pasted essay",
                key="essay_text",
                height=170,
                placeholder="Paste one essay here if you are not uploading files...",
            )
        )
        words = count_words(essay_text)
        st.markdown(f'<div class="small-note">Pasted text word count: {words} words</div>', unsafe_allow_html=True)

        left, right = st.columns([1, 1])
        with left:
            st.button(
                "Analyse Essays",
                type="primary",
                use_container_width=True,
                key="btn_analyze",
                on_click=_on_analyze,
            )
        with right:
            st.button(
                "Clear Text",
                use_container_width=True,
                key="btn_clear",
                on_click=_on_clear,
            )

        if st.session_state.get("batch_scan_message"):
            st.info(st.session_state["batch_scan_message"])

        st.markdown(
            f'<div class="small-note">* Minimum {MIN_WORDS} words recommended per document. Uploaded reports are saved in <code>{html.escape(str(REPORT_STORAGE_DIR))}</code>.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    return demo_name, essay_text


def render_meter(value: int) -> None:
    st.markdown(
        f"""
        <div class="meter" aria-label="AI probability">
            <div style="width: {max(0, min(value, 100))}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(result: DemoResult | None) -> None:
    st.markdown('<div class="section-title">Results Section</div>', unsafe_allow_html=True)
    st.markdown('<span class="educator-results-panel-marker"></span>', unsafe_allow_html=True)
    if result is None:
        st.markdown(
            """
            <div class="box">
                <div class="result-head">
                    <div class="classification">Classification: Waiting for essay</div>
                    <div class="confidence">--<span>Run analysis</span></div>
                </div>
                <p class="summary">Submit text or upload a file, then analyse the essay.</p>
                <div class="stat-grid">
                    <div class="stat-card">
                        <div class="stat-label">Status</div>
                        <div class="stat-value">Ready</div>
                        <div class="stat-caption">Static demo backend</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Inputs</div>
                        <div class="stat-value">PDF / DOCX / TXT</div>
                        <div class="stat-caption">Paste text also supported</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Output</div>
                        <div class="stat-value">2 classes</div>
                        <div class="stat-caption">Human, AI</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-head">
            <div>
                <div class="small-note">Classification</div>
                <div class="classification">{html.escape(result.label)}</div>
            </div>
            <div class="confidence">{result.confidence}%<span>Auditor estimate</span></div>
        </div>
        <p class="summary">{html.escape(result.summary)}</p>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">AI probability</div>
                <div class="stat-value">{result.ai_probability}%</div>
                <div class="stat-caption">Calibrated auditor score</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Displayed score</div>
                <div class="stat-value">{result.confidence}%</div>
                <div class="stat-caption">Probability of the displayed label</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Scenario</div>
                <div class="stat-value">{html.escape(result.name)}</div>
                <div class="stat-caption">Static sample result</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_meter(result.ai_probability)
    # Keep the visual meter; remove the duplicate 'Demo AI probability' note
    st.markdown("</div>", unsafe_allow_html=True)


def render_annotation(words: Iterable[tuple[str, str]]) -> None:
    spans = []
    for word, kind in words:
        safe_word = html.escape(word)
        css_kind = "human" if kind == "human" else "ai" if kind == "ai" else "neutral"
        spans.append(f'<span class="mark mark-{css_kind}">{safe_word}</span>')
    st.markdown(f'<div class="annotation">{" ".join(spans)}</div>', unsafe_allow_html=True)


def render_token_table(tokens: Iterable[tuple[int, str, str, str]]) -> None:
    rows = []
    for rank, token, weight, signal in tokens:
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{html.escape(token)}</td>"
            f"<td>{html.escape(weight)}</td>"
            f"<td>{html.escape(signal)}</td>"
            "</tr>"
        )
    st.markdown(
        """
        <table class="data-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Token</th>
                    <th>Weight</th>
                    <th>Signal</th>
                </tr>
            </thead>
            <tbody>
        """
        + "".join(rows)
        + """
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_metric_table(metrics: Iterable[tuple[str, str, str]]) -> None:
    rows = []
    for signal, value, interpretation in metrics:
        rows.append(
            "<tr>"
            f"<td>{html.escape(signal)}</td>"
            f"<td>{html.escape(value)}</td>"
            f"<td>{html.escape(interpretation)}</td>"
            "</tr>"
        )
    st.markdown(
        """
        <table class="data-table">
            <thead>
                <tr>
                    <th>Signal</th>
                    <th>Demo value</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>
        """
        + "".join(rows)
        + """
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_explanation(result: DemoResult | None) -> None:
    if result is None:
        return

    st.markdown('<div class="section-title">Explanation Section</div>', unsafe_allow_html=True)
    st.markdown('<span class="educator-explanation-panel-marker"></span>', unsafe_allow_html=True)
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown("**Annotated Essay**")
    st.markdown(
        '<div class="small-note">Green highlights lean human. Red highlights lean AI authored.</div>',
        unsafe_allow_html=True,
    )
    render_annotation(result.annotated_words)
    st.write("")
    st.markdown(f'<div class="insight">{html.escape(result.explanation)}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def document_table_rows(documents) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for document in documents:
        rows.append(
            {
                "ID": document.id,
                "File": document.original_filename,
                "Status": document.status.title(),
                "Words": document.word_count,
                "Scenario": document.demo_scenario,
                "Label": document.label or "—",
                "Confidence": f"{document.confidence}%" if document.confidence is not None else "—",
                "AI probability": f"{document.ai_probability}%" if document.ai_probability is not None else "—",
                "Uploaded": document.uploaded_at,
                "Processed": document.processed_at or "—",
            }
        )
    return rows


def render_document_viewer(conn, *, current_user, document_id: int, key_prefix: str) -> None:
    row = get_document_for_user(conn, user_id=current_user.id, document_id=document_id)
    if row is None:
        st.warning("Document not found or you do not have access to it.")
        return

    document = row_to_document(row)
    source_text = read_text_file(document.stored_path)
    with st.expander(f"Viewing: {document.original_filename}", expanded=True):
        meta_left, meta_right = st.columns([1, 1])
        with meta_left:
            st.markdown(f"**Status:** {html.escape(document.status.title())}")
            st.markdown(f"**Words:** {document.word_count}")
            st.markdown(f"**Uploaded:** {html.escape(document.uploaded_at)}")
        with meta_right:
            st.markdown(f"**Label:** {html.escape(document.label or 'Not processed')}")
            st.markdown(
                f"**Confidence:** {document.confidence}%" if document.confidence is not None else "**Confidence:** —"
            )
            st.markdown(f"**Processed:** {html.escape(document.processed_at or '—')}")

        if document.report_path and Path(document.report_path).exists():
            st.download_button(
                "Download Stored Report",
                data=Path(document.report_path).read_bytes(),
                file_name=Path(document.report_path).name,
                mime="application/pdf",
                use_container_width=True,
                key=f"{key_prefix}_download_{document.id}",
            )

        st.text_area(
            "Extracted document text",
            value=source_text,
            height=240,
            disabled=True,
            key=f"{key_prefix}_text_{document.id}",
        )


def render_document_management(conn, current_user) -> None:
    st.markdown('<div class="section-title">Document Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Only documents uploaded by your educator account are shown here.</div>', unsafe_allow_html=True)

    all_docs = [row_to_document(row) for row in list_documents_for_user(conn, user_id=current_user.id)]
    processed_docs = [row_to_document(row) for row in list_documents_for_user(conn, user_id=current_user.id, status="processed")]

    documents_tab, processed_tab = st.tabs(["Documents", "Processed Reports"])

    with documents_tab:
        st.markdown("**Your uploaded documents**")
        if not all_docs:
            st.info("No documents yet. Upload files and click Analyse Essays to store and process them.")
        else:
            st.dataframe(document_table_rows(all_docs), use_container_width=True, hide_index=True)
            document_options = {f"#{doc.id} · {doc.original_filename} · {doc.status.title()}": doc.id for doc in all_docs}
            selected_view = st.selectbox(
                "View document",
                options=["—"] + list(document_options.keys()),
                key="document_view",
            )
            if selected_view != "—":
                render_document_viewer(conn, current_user=current_user, document_id=document_options[str(selected_view)], key_prefix="document")

            selected_delete = st.multiselect(
                "Select documents to delete",
                options=list(document_options.keys()),
                key="document_delete",
            )
            if st.button("Delete Selected Documents", use_container_width=True, key="delete_documents"):
                deleted = delete_document_records(
                    conn,
                    user_id=current_user.id,
                    document_ids=[document_options[label] for label in selected_delete],
                )
                st.success(f"Deleted {deleted} document(s) and any stored report file(s).")
                st.rerun()

    with processed_tab:
        st.markdown("**Processed documents and stored demo reports**")
        if not processed_docs:
            st.info("No processed reports yet. Upload documents and click Analyse Essays to generate demo reports.")
        else:
            st.dataframe(document_table_rows(processed_docs), use_container_width=True, hide_index=True)
            processed_options = {f"#{doc.id} · {doc.original_filename}": doc.id for doc in processed_docs}
            selected_view = st.selectbox(
                "View processed document/report",
                options=["—"] + list(processed_options.keys()),
                key="processed_doc_view",
            )
            if selected_view != "—":
                render_document_viewer(
                    conn,
                    current_user=current_user,
                    document_id=processed_options[str(selected_view)],
                    key_prefix="processed",
                )

            selected_delete = st.multiselect(
                "Select processed documents/reports to delete",
                options=list(processed_options.keys()),
                key="processed_doc_delete",
            )
            if st.button("Delete Selected Processed Documents", use_container_width=True, key="delete_processed_docs"):
                deleted = delete_document_records(
                    conn,
                    user_id=current_user.id,
                    document_ids=[processed_options[label] for label in selected_delete],
                )
                st.success(f"Deleted {deleted} processed document(s) and report file(s).")
                st.rerun()


def main() -> None:
    configure_page()
    conn = get_conn()
    ensure_bootstrap_admin(conn)

    render_sidebar()
    render_title()

    current_user = auth_current_user(conn)
    if current_user is None:
        render_auth_gate(conn)
        return

    if current_user.role == "super_admin":
        view = st.radio(
            "View",
            options=["Educator Page", "User Management"],
            horizontal=True,
            key="admin_view",
            index=1,
        )
        if view == "User Management":
            render_super_admin_portal(conn, current_user)
            return

    content_left, content_right = st.columns([1.35, 1], gap="large")
    with content_left:
        _, _ = render_input(conn, current_user)
    with content_right:
        render_results(st.session_state.get("result"))
    render_document_management(conn, current_user)
    render_explanation(st.session_state.get("result"))


if __name__ == "__main__":
    main()
