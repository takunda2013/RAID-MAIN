"""
Streamlit app for the Robust AI Detector.

This app reuses the existing UI styling/auth/database layer from app.py, but it
connects the Analyse button to Robust-Ai-Detector/Deployment/deployment_pipeline.py
instead of the static demo results.

Run from the repository root:

    streamlit run Robust-Ai-Detector/UI/deploy_app.py
"""

from __future__ import annotations

import html
import os
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Iterable

import streamlit as st

import app as demo_ui
from db import (
    create_document,
    list_documents_for_user,
    mark_document_processed,
    row_to_document,
)

UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
DEPLOYMENT_DIR = PROJECT_ROOT / "Deployment"
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from deployment_pipeline import (  # noqa: E402
    DeploymentConfig,
    analyze_text,
    result_to_ui_tuples,
)

APP_VERSION = "v1.0"
LIVE_MODE_NAME = "Deployment"
LIVE_LIME_SAMPLES = int(os.environ.get("AI_DETECTOR_LIME_SAMPLES", "8"))
LIVE_LIME_FEATURES = int(os.environ.get("AI_DETECTOR_LIME_FEATURES", "12"))


def render_live_processing_overlay(
    placeholder,
    *,
    processed_count: int,
    total_count: int,
    status_text: str = "Running model stack...",
    progress_ratio: float | None = None,
    animate_bar: bool = False,
) -> None:
    safe_total = max(1, int(total_count))
    safe_processed = max(0, min(int(processed_count), safe_total))
    if progress_ratio is None:
        progress_value = safe_processed / safe_total
    else:
        progress_value = max(0.0, min(float(progress_ratio), 1.0))
    progress_percent = int(round(progress_value * 100))
    safe_status = html.escape(str(status_text))
    bar_class = "live-progress-animate" if animate_bar else ""
    placeholder.markdown(
        f"""
        <style>
            .live-progress-overlay {{
                position: fixed;
                inset: 0;
                z-index: 99999;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(13, 18, 26, 0.30);
                backdrop-filter: blur(1.5px);
            }}
            .live-progress-card {{
                width: min(440px, 90vw);
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 0.35);
                background: rgba(255, 255, 255, 0.96);
                box-shadow: 0 22px 48px rgba(10, 14, 20, 0.28);
                padding: 1rem 1.1rem;
                text-align: center;
            }}
            .live-progress-spinner {{
                width: 36px;
                height: 36px;
                margin: 0 auto 0.55rem;
                border: 3px solid rgba(31, 111, 74, 0.18);
                border-top-color: #1f6f4a;
                border-radius: 50%;
                animation: live-progress-spin 0.75s linear infinite;
            }}
            @keyframes live-progress-spin {{
                from {{ transform: rotate(0deg); }}
                to {{ transform: rotate(360deg); }}
            }}
            @keyframes live-progress-shimmer {{
                from {{ background-position: 0% 0; }}
                to {{ background-position: 200% 0; }}
            }}
            .live-progress-title {{
                color: #203224;
                font-size: 1.05rem;
                font-weight: 800;
                margin-bottom: 0.22rem;
            }}
            .live-progress-count {{
                color: #3f2b20;
                font-size: 0.97rem;
                font-weight: 700;
                margin-bottom: 0.62rem;
            }}
            .live-progress-bar {{
                width: 100%;
                height: 11px;
                background: rgba(18, 22, 32, 0.08);
                border: 1px solid rgba(18, 22, 32, 0.12);
                border-radius: 999px;
                overflow: hidden;
            }}
            .live-progress-bar > span {{
                display: block;
                height: 100%;
                width: {progress_percent}%;
                background: linear-gradient(90deg, #1f6f4a, #2f8f66);
                transition: width 180ms ease;
            }}
            .live-progress-bar > span.live-progress-animate {{
                background: linear-gradient(
                    100deg,
                    #1f6f4a 0%,
                    #2f8f66 40%,
                    #55b488 50%,
                    #2f8f66 60%,
                    #1f6f4a 100%
                );
                background-size: 200% 100%;
                animation: live-progress-shimmer 1.05s linear infinite;
            }}
            .live-progress-status {{
                margin-top: 0.58rem;
                color: #5b351f;
                font-size: 0.84rem;
                font-weight: 650;
            }}
        </style>
        <div class="live-progress-overlay" role="status" aria-live="polite">
            <div class="live-progress-card">
                <div class="live-progress-spinner"></div>
                <div class="live-progress-title">Analyzing Essays</div>
                <div class="live-progress-count">{safe_processed} / {safe_total} processed</div>
                <div class="live-progress-bar"><span class="{bar_class}"></span></div>
                <div class="live-progress-status">{safe_status}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _overall_progress_ratio(
    *,
    completed_docs: int,
    total_docs: int,
    current_doc_stage: float = 0.0,
) -> float:
    safe_total = max(1, int(total_docs))
    safe_completed = max(0, min(int(completed_docs), safe_total))
    safe_stage = max(0.0, min(float(current_doc_stage), 1.0))
    return min(1.0, (safe_completed + safe_stage) / safe_total)


def deployment_result_to_ui_result(result: dict) -> demo_ui.DemoResult:
    tuples = result_to_ui_tuples(result)
    auditor_prob_ai = float(result.get("auditor", {}).get("prob_ai", 0.0))
    auditor_percent = int(round(auditor_prob_ai * 100))
    auditor_label = "AI-generated" if auditor_percent >= 50 else "Human-written"
    class_probability_percent = (
        auditor_percent if auditor_label == "AI-generated" else 100 - auditor_percent
    )
    return demo_ui.DemoResult(
        name=LIVE_MODE_NAME,
        label=auditor_label,
        confidence=class_probability_percent,
        ai_probability=auditor_percent,
        summary=str(result["summary"]),
        explanation=str(result["explanation"]),
        tokens=tuples["tokens"],
        metrics=tuples["metrics"],
        annotated_words=tuples["annotated_words"],
    )


def run_live_analysis(
    text: str, *, include_lime: bool = True
) -> tuple[demo_ui.DemoResult, dict]:
    config = DeploymentConfig(
        lime_num_samples=LIVE_LIME_SAMPLES,
        lime_num_features=LIVE_LIME_FEATURES,
    )
    result = analyze_text(
        text,
        config=config,
        include_lime=include_lime,
    )
    return deployment_result_to_ui_result(result), result


def store_uploaded_document_live(conn, *, user_id: int, uploaded_file) -> int | None:
    extracted_text = demo_ui.extract_uploaded_text(uploaded_file)
    if not extracted_text:
        return None

    storage_dir = demo_ui.user_document_dir(user_id)
    original_name = getattr(uploaded_file, "name", "document.txt")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    text_filename = f"{timestamp}_{demo_ui.safe_filename(original_name)}.txt"
    text_path = storage_dir / text_filename
    text_path.write_text(extracted_text, encoding="utf-8")

    return create_document(
        conn,
        user_id=user_id,
        original_filename=original_name,
        stored_path=str(text_path),
        demo_scenario=LIVE_MODE_NAME,
        word_count=demo_ui.count_words(extracted_text),
    )


def store_pasted_document_live(conn, *, user_id: int, text: str) -> int | None:
    if not text.strip():
        return None

    storage_dir = demo_ui.user_document_dir(user_id)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    text_path = storage_dir / f"{timestamp}_pasted_essay.txt"
    text_path.write_text(text.strip(), encoding="utf-8")

    return create_document(
        conn,
        user_id=user_id,
        original_filename=f"Pasted essay {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
        stored_path=str(text_path),
        demo_scenario=LIVE_MODE_NAME,
        word_count=demo_ui.count_words(text),
    )


def process_uploaded_documents_live(conn, *, user_id: int, overlay_placeholder=None) -> int:
    pending_rows = list_documents_for_user(conn, user_id=user_id, status="uploaded")
    total = len(pending_rows)
    processed_count = 0

    overlay = overlay_placeholder or st.empty()
    if total > 0:
        render_live_processing_overlay(
            overlay,
            processed_count=0,
            total_count=total,
            status_text="Preparing first document...",
            progress_ratio=0.02,
            animate_bar=True,
        )

    for idx, row in enumerate(pending_rows):
        document = row_to_document(row)
        render_live_processing_overlay(
            overlay,
            processed_count=processed_count,
            total_count=total,
            status_text=f"Loading text for {idx + 1} of {total}: {document.original_filename}",
            progress_ratio=_overall_progress_ratio(
                completed_docs=processed_count,
                total_docs=total,
                current_doc_stage=0.08,
            ),
            animate_bar=True,
        )

        source_text = demo_ui.read_text_file(document.stored_path)
        if not source_text.strip():
            render_live_processing_overlay(
                overlay,
                processed_count=processed_count,
                total_count=total,
                status_text=f"Skipping empty document {idx + 1} of {total}",
                progress_ratio=_overall_progress_ratio(
                    completed_docs=processed_count,
                    total_docs=total,
                    current_doc_stage=0.98,
                ),
            )
            continue

        render_live_processing_overlay(
            overlay,
            processed_count=processed_count,
            total_count=total,
            status_text=f"Running AI analysis for {idx + 1} of {total}...",
            progress_ratio=_overall_progress_ratio(
                completed_docs=processed_count,
                total_docs=total,
                current_doc_stage=0.32,
            ),
            animate_bar=True,
        )

        ui_result, raw_result = run_live_analysis(source_text, include_lime=True)

        render_live_processing_overlay(
            overlay,
            processed_count=processed_count,
            total_count=total,
            status_text=f"Generating report for {idx + 1} of {total}...",
            progress_ratio=_overall_progress_ratio(
                completed_docs=processed_count,
                total_docs=total,
                current_doc_stage=0.78,
            ),
            animate_bar=True,
        )

        report_path = demo_ui.user_report_dir(user_id) / f"report_document_{document.id}.html"
        report_path.write_bytes(
            demo_ui.make_styled_report_html(
                ui_result,
                source_text,
                title=f"AI Detector Report - {document.original_filename}",
            )
        )

        render_live_processing_overlay(
            overlay,
            processed_count=processed_count,
            total_count=total,
            status_text=f"Saving results for {idx + 1} of {total}...",
            progress_ratio=_overall_progress_ratio(
                completed_docs=processed_count,
                total_docs=total,
                current_doc_stage=0.92,
            ),
            animate_bar=True,
        )

        mark_document_processed(
            conn,
            user_id=user_id,
            document_id=document.id,
            report_path=str(report_path),
            label=ui_result.label,
            confidence=ui_result.confidence,
            ai_probability=ui_result.ai_probability,
        )
        processed_count += 1
        render_live_processing_overlay(
            overlay,
            processed_count=processed_count,
            total_count=total,
            status_text=f"Completed {processed_count} of {total}",
            progress_ratio=_overall_progress_ratio(
                completed_docs=processed_count,
                total_docs=total,
                current_doc_stage=0.0,
            ),
        )

        # Keep the most recent full JSON available for the active session.
        st.session_state["result"] = ui_result
        st.session_state["raw_deployment_result"] = raw_result

    if total > 0:
        overlay.empty()

    return processed_count


def render_live_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-logo-row">
                    <span class="sidebar-logo-mark">&#128196;</span>
                    <span>Originality<em>Engine</em></span>
                </div>
                <span>Detector console.</span>
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
            st.button(
                "Logout",
                use_container_width=True,
                key="btn_logout",
                on_click=demo_ui.auth_logout,
            )
        else:
            st.markdown(
                "<span class='sidebar-pill'>Not signed in</span>",
                unsafe_allow_html=True,
            )

        st.markdown("### Detection Modes")
        st.markdown(
            """
            <span class="sidebar-pill">Human</span>
            <span class="sidebar-pill">AI</span>
            <span class="sidebar-pill">Paraphrased AI -> AI</span>
            """,
            unsafe_allow_html=True,
        )
        # Backend/LIME details removed for deployment UI
        st.divider()
        st.markdown("### Reports")
        result = st.session_state.get("result")
        source_text = st.session_state.get("source_text", "")
        st.session_state.setdefault("report_ready", False)

        def _prepare_report() -> None:
            st.session_state["report_ready"] = True

        if not result:
            st.button(
                "Download Report (.html)",
                disabled=True,
                use_container_width=True,
                key="download_report_disabled",
            )
        elif not st.session_state.get("report_ready"):
            st.button(
                "Prepare Report",
                use_container_width=True,
                key="btn_prepare_report",
                on_click=_prepare_report,
            )
            st.caption("Generate a report for the current result, then download.")
        else:
            st.download_button(
                "Download Styled Report (.html)",
                data=demo_ui.make_styled_report_html(
                    result, source_text, title="AI Essay Detector Report"
                ),
                file_name="ai_detector_report.html",
                mime="text/html",
                use_container_width=True,
                key="download_report",
            )


def render_live_title() -> None:
    result = st.session_state.get("result")
    if result:
        status_markup = dedent(
            f"""
            <div>
                <div class="status-icon">&#128220;</div>
                <div class="status-kicker">Analysis complete</div>
                <div class="status-score">{result.confidence}%</div>
                <div class="status-copy">{html.escape(result.label)} · {result.confidence}% {"AI probability" if result.label == "AI-generated" else "Human probability"}</div>
            </div>
            """
        ).strip()
    else:
        status_markup = dedent(
            """
            <div>
                <div class="status-icon">&#128220;</div>
                <div class="status-kicker">Detector ready</div>
                <div class="status-copy">Submit content to run the deployment model stack.</div>
            </div>
            """
        ).strip()

    st.markdown(
        f"""
        <div class="editorial-topbar">
            <div class="brand-lockup">
                <span class="brand-icon">&#128196;</span>
                <span>Originality<em>Engine</em></span>
            </div>
            <div class="suite-label">Editorial Integrity Suite &middot; {APP_VERSION}</div>
        </div>
        <div class="app-title">
            <div class="hero-panel">
                <div class="eyebrow">Deployment</div>
                <h1 style="color: white;">Safeguarding the Integrity of Written Work</h1>
                <div class="subtitle">
                    Paste text or upload essays. An Interpretable and Robust Detector for Student Essays.
                </div>
            </div>
            <div class="status-panel">
                {status_markup}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_input(conn, current_user) -> tuple[str, str]:
    st.markdown('<div class="section-title">Essay Intake</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="educator-input-panel-marker"></span>', unsafe_allow_html=True
    )
    with st.container(border=False):
        st.markdown('<div class="box">', unsafe_allow_html=True)

        def _ensure_defaults() -> None:
            st.session_state.setdefault("essay_text", "")
            st.session_state.setdefault("result", None)
            st.session_state.setdefault("source_text", "")
            st.session_state.setdefault("file_extract_failed", False)
            st.session_state.setdefault("batch_scan_message", "")
            st.session_state.setdefault("raw_deployment_result", None)

        def _on_analyze() -> None:
            uploaded_files = st.session_state.get("upload_files") or []
            added_count = 0
            failed_files: list[str] = []
            overlay_placeholder = st.empty()
            estimated_total = len(uploaded_files)

            try:
                if estimated_total > 0:
                    render_live_processing_overlay(
                        overlay_placeholder,
                        processed_count=0,
                        total_count=estimated_total,
                        status_text="Preparing uploaded documents...",
                    )

                for uploaded in uploaded_files:
                    document_id = store_uploaded_document_live(
                        conn,
                        user_id=current_user.id,
                        uploaded_file=uploaded,
                    )
                    if document_id is None:
                        failed_files.append(getattr(uploaded, "name", "uploaded file"))
                    else:
                        added_count += 1

                pasted_text = str(st.session_state.get("essay_text", "")).strip()
                if not uploaded_files and pasted_text:
                    render_live_processing_overlay(
                        overlay_placeholder,
                        processed_count=0,
                        total_count=1,
                        status_text="Preparing pasted essay...",
                    )
                    document_id = store_pasted_document_live(
                        conn,
                        user_id=current_user.id,
                        text=pasted_text,
                    )
                    if document_id is not None:
                        added_count += 1

                if added_count == 0:
                    st.session_state["batch_scan_message"] = (
                        "No essay text found to analyse."
                    )
                    return

                # Process documents with centered floating progress.
                processed_count = process_uploaded_documents_live(
                    conn,
                    user_id=current_user.id,
                    overlay_placeholder=overlay_placeholder,
                )

                st.session_state["source_text"] = pasted_text

                st.session_state["report_ready"] = False
                st.session_state["file_extract_failed"] = bool(failed_files)
                st.session_state["batch_scan_message"] = (
                    f"Added {added_count} document(s) and processed {processed_count} document(s)."
                )
                if failed_files:
                    st.session_state["batch_scan_message"] += (
                        " Could not extract: " + ", ".join(failed_files)
                    )
            finally:
                overlay_placeholder.empty()

        def _on_clear() -> None:
            st.session_state["essay_text"] = ""
            st.session_state["result"] = None
            st.session_state["source_text"] = ""
            st.session_state["file_extract_failed"] = False
            st.session_state["report_ready"] = False
            st.session_state["batch_scan_message"] = ""
            st.session_state["raw_deployment_result"] = None

        _ensure_defaults()

        uploaded_files = st.file_uploader(
            "Upload essay files for scanning",
            type=("pdf", "docx", "txt"),
            accept_multiple_files=True,
            key="upload_files",
            help="Upload documents to run the detector. Large batches may take time because LIME audits each essay.",
        )

        if uploaded_files:
            st.markdown(
                f'<div class="small-note">Queued from uploader: {len(uploaded_files)} file(s). Click Analyse Essays to run the model stack.</div>',
                unsafe_allow_html=True,
            )

        if st.session_state.get("file_extract_failed"):
            st.warning(
                "Some files could not be extracted. Supported formats are PDF, DOCX, and TXT."
            )

        essay_text = str(
            st.text_area(
                "Optional pasted essay",
                key="essay_text",
                height=170,
                placeholder="Paste one essay here if you are not uploading files...",
            )
        )
        words = demo_ui.count_words(essay_text)
        st.markdown(
            f'<div class="small-note">Pasted text word count: {words} words</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1])
        with left:
            if st.button(
                "Analyse Essays",
                type="primary",
                use_container_width=True,
                key="btn_analyze",
                disabled=st.session_state.get("processing", False),
            ):
                st.session_state["processing"] = True
                try:
                    _on_analyze()
                finally:
                    st.session_state["processing"] = False
                # Force a fresh rerun so top status cards (rendered earlier in the
                # script) reflect the newest processed result immediately.
                st.rerun()
        with right:
            st.button(
                "Clear Text",
                use_container_width=True,
                key="btn_clear",
                on_click=_on_clear,
            )

        if st.session_state.get("batch_scan_message"):
            st.info(st.session_state["batch_scan_message"])

        # Minimum-words/report-storage note removed for deployment UI
        st.markdown("</div>", unsafe_allow_html=True)

    return LIVE_MODE_NAME, essay_text


def render_live_results(result: demo_ui.DemoResult | None) -> None:
    demo_ui.render_results(result)
    # 'Model details' expander removed from the UI per user request.
    # Raw deployment details remain available in `st.session_state['raw_deployment_result']`
    # for debugging if needed, but they are not shown in the live interface.


def _dominant_highlight_kind(result: demo_ui.DemoResult) -> str:
    label = str(result.label).strip().lower()
    if "human" in label and "ai" not in label:
        return "human"
    if "ai" in label:
        return "ai"
    return "ai" if int(result.ai_probability) >= 50 else "human"


def render_live_annotation(
    words: Iterable[tuple[str, str]], *, dominant_kind: str
) -> None:
    dominant_kind = "human" if dominant_kind == "human" else "ai"
    opposite_kind = "ai" if dominant_kind == "human" else "human"
    spans: list[str] = []
    for word, kind in words:
        safe_word = html.escape(str(word))
        normalized_kind = str(kind).strip().lower()
        css_kind = opposite_kind if normalized_kind == opposite_kind else dominant_kind
        spans.append(f'<span class="mark mark-{css_kind}">{safe_word}</span>')
    st.markdown(f'<div class="annotation">{" ".join(spans)}</div>', unsafe_allow_html=True)


def render_live_explanation(result: demo_ui.DemoResult | None) -> None:
    if result is None:
        return
    dominant_kind = _dominant_highlight_kind(result)
    inverse_copy = (
        "Green highlights mark human exceptions inside AI-classified text."
        if dominant_kind == "ai"
        else "Red highlights mark AI exceptions inside human-classified text."
    )

    st.markdown('<div class="section-title">Explanation Section</div>', unsafe_allow_html=True)
    st.markdown('<span class="educator-explanation-panel-marker"></span>', unsafe_allow_html=True)
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown("**Annotated Essay**")
    st.markdown(
        f'<div class="small-note">{html.escape(inverse_copy)}</div>',
        unsafe_allow_html=True,
    )
    render_live_annotation(result.annotated_words, dominant_kind=dominant_kind)
    st.write("")
    st.markdown(
        f'<div class="insight">{html.escape(result.explanation)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">LIME Token Weights</div>', unsafe_allow_html=True
    )
    st.markdown('<div class="box">', unsafe_allow_html=True)
    demo_ui.render_token_table(result.tokens)
    # demo_ui.render_metric_table(result.metrics)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    demo_ui.configure_page()
    conn = demo_ui.get_conn()
    demo_ui.ensure_bootstrap_admin(conn)

    render_live_sidebar()
    render_live_title()

    current_user = demo_ui.auth_current_user(conn)
    if current_user is None:
        demo_ui.render_auth_gate(conn)
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
            demo_ui.render_super_admin_portal(conn, current_user)
            return

    content_left, content_right = st.columns([1.35, 1], gap="large")
    with content_left:
        _, _ = render_live_input(conn, current_user)
    with content_right:
        render_live_results(st.session_state.get("result"))
    # Render explanation above the document management section
    render_live_explanation(st.session_state.get("result"))
    demo_ui.render_document_management(conn, current_user)


if __name__ == "__main__":
    main()
