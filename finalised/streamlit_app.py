import os
import time
from typing import Any, Dict, List, Tuple

import streamlit as st


st.set_page_config(page_title="PII Masker", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; }
      div[data-testid="stMetric"] { background: #F5F7FB; padding: 12px 14px; border-radius: 12px; }
      .small-muted { color: rgba(17, 24, 39, 0.72); font-size: 0.92rem; }
      .card { background: #FFFFFF; border: 1px solid rgba(17, 24, 39, 0.08); border-radius: 14px; padding: 14px 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## PII Masker")
st.markdown(
    '<div class="small-muted">GLiNER entity detection + deterministic regex backstops. Paste text, mask, and export results.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _load_gliner(model_name: str):
    # GLiNER is an optional dependency; keep import inside the cached function.
    try:
        from gliner import GLiNER
    except Exception as e:
        raise RuntimeError(
            "Failed to import 'gliner'. If you're deploying on Streamlit Community Cloud, "
            "ensure you have a repo-root requirements.txt (e.g. '-r finalised/requirements.streamlit.txt') "
            "and pin a supported Python in runtime.txt (e.g. python-3.10 or python-3.11)."
        ) from e

    return GLiNER.from_pretrained(model_name)


def _run_masking(
    text: str,
    model_name: str,
    threshold: float,
) -> Tuple[str, Dict[str, str], Dict[str, float], List[Dict[str, Any]]]:
    from pii_masker.masking import mask_with_gliner

    model = _load_gliner(model_name)
    return mask_with_gliner(text=text, model_name_or_obj=model, threshold=threshold)


with st.sidebar:
    st.markdown("### Settings")

    model_name = st.text_input(
        "GLiNER model",
        value=os.getenv("GLINER_MODEL", "urchade/gliner_multi_pii-v1"),
        help="Hugging Face model id (downloaded on first use).",
    )

    threshold = st.slider(
        "Detection threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(os.getenv("GLINER_THRESHOLD", "0.7")),
        step=0.05,
        help="Higher = fewer entities masked.",
    )

    st.divider()
    st.markdown("### Output")
    show_mapping = st.checkbox("Include mapping column", value=True)
    show_scores = st.checkbox("Include score column", value=True)
    show_spans = st.checkbox("Show raw spans (debug)", value=False)

    st.divider()
    st.markdown("### Runtime")
    runtime_lines = []
    try:
        import torch

        runtime_lines.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            runtime_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        runtime_lines.append("PyTorch not available (or failed to import).")
    st.caption("\n".join(runtime_lines))


tab_mask, tab_about = st.tabs(["Mask", "About"])

with tab_mask:
    st.markdown(
        """
        <div class="card">
          <b>Safety note:</b> avoid pasting real customer PII into public deployments.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col_left, col_right = st.columns((1, 1))

    default_example = """You are an assistant helping review a customer-reported issue.

Customer details:
Name: Harriet Jane Evans
Date of birth: 14 March 1989
Email: harriet.evans89@gmail.com
Mobile: +44 7911 456 882

Account information:
Sort code: 20-45-67
Account number: 83920145
IBAN: GB29BARC20456783920145
Customer ID: CUST-UK-492817
Transaction ID: TXN-77834192

Issue: Customer claims £2,450 was transferred on 11/01/2026."""

    with col_left:
        st.subheader("Input")
        with st.container(border=True):
            input_text = st.text_area(
                "Text to mask",
                height=320,
                placeholder="Paste text containing PII here...",
                value=st.session_state.get("input_text", ""),
            )

            c1, c2 = st.columns((1, 1))
            with c1:
                run = st.button("Mask", type="primary", use_container_width=True)
            with c2:
                if st.button("Load example", use_container_width=True):
                    st.session_state["input_text"] = default_example
                    st.rerun()

    with col_right:
        st.subheader("Output")
        with st.container(border=True):
            if run:
                if not input_text.strip():
                    st.warning("Please paste some text first.")
                else:
                    started = time.perf_counter()
                    with st.spinner("Running masking pipeline..."):
                        try:
                            masked_text, mapping, scores, spans = _run_masking(
                                text=input_text,
                                model_name=model_name,
                                threshold=threshold,
                            )
                        except Exception as e:
                            st.error("Masking failed.")
                            st.exception(e)
                        else:
                            elapsed_ms = int((time.perf_counter() - started) * 1000)
                            st.text_area("Masked text", value=masked_text, height=320)

                            st.write("")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Tags", len(mapping))
                            m2.metric("Spans", len(spans))
                            m3.metric("Time", f"{elapsed_ms} ms")

                            details_rows = []
                            all_tags = sorted(set(mapping.keys()) | set(scores.keys()))
                            for tag in all_tags:
                                row = {"tag": tag}
                                if show_mapping:
                                    row["original"] = mapping.get(tag, "")
                                if show_scores:
                                    score = scores.get(tag, None)
                                    row["score"] = None if score is None else float(score)
                                details_rows.append(row)

                            with st.expander("Details table", expanded=True):
                                st.dataframe(details_rows, use_container_width=True)
                                if show_mapping:
                                    st.caption("Mapping shows what each placeholder replaced.")

                                c_dl1, c_dl2 = st.columns((1, 1))
                                with c_dl1:
                                    st.download_button(
                                        "Download masked text",
                                        data=masked_text,
                                        file_name="masked.txt",
                                        mime="text/plain",
                                        use_container_width=True,
                                    )
                                with c_dl2:
                                    try:
                                        import json

                                        st.download_button(
                                            "Download details (JSON)",
                                            data=json.dumps(details_rows, indent=2, ensure_ascii=False),
                                            file_name="details.json",
                                            mime="application/json",
                                            use_container_width=True,
                                        )
                                    except Exception:
                                        pass

                            if show_spans:
                                with st.expander("Raw spans (debug)", expanded=False):
                                    st.dataframe(spans, use_container_width=True)
            else:
                st.info("Paste text on the left, then click ‘Mask’.", icon="ℹ️")

with tab_about:
    st.markdown(
        """
        ### What this app does
        - Runs **GLiNER** entity detection over the input text.
        - Maps GLiNER labels to your canonical labels.
        - Adds deterministic regex-based spans to cover common misses.
        - Replaces detected spans with placeholders like `[PERSON_1]`.

        ### Notes
        - On first run, the model downloads and may take time.
        - Free hosting is typically CPU-only; performance depends on text length.
        """
    )
