import os
import time
from typing import Any, Dict, List, Tuple

import streamlit as st


st.set_page_config(
        page_title="PII Masker",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
            :root {
                --bg: #F8FAFC;
                --card: #FFFFFF;
                --border: rgba(15, 23, 42, 0.10);
                --muted: rgba(15, 23, 42, 0.72);
                --shadow: 0 1px 0 rgba(15, 23, 42, 0.04);
            }

            .stApp { background: var(--bg); }
            .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; }

            /* Sidebar */
            section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--border); }
            section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }

            /* Cards */
            .card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 14px 16px;
                box-shadow: var(--shadow);
            }
            .small-muted { color: var(--muted); font-size: 0.92rem; }
            .kpi {
                background: #F1F5F9;
                border: 1px solid rgba(15, 23, 42, 0.06);
                border-radius: 14px;
                padding: 10px 12px;
            }

            /* Metrics: keep Streamlit metric but make it look consistent */
            div[data-testid="stMetric"] {
                background: #F1F5F9;
                border: 1px solid rgba(15, 23, 42, 0.06);
                padding: 12px 14px;
                border-radius: 14px;
            }

            /* Inputs */
            textarea { border-radius: 12px !important; }
            .stDownloadButton > button, .stButton > button {
                border-radius: 12px !important;
            }

            /* Reduce extra top margin on headers */
            h1, h2, h3 { letter-spacing: -0.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
if "example_idx" not in st.session_state:
    st.session_state["example_idx"] = 0

st.title("PII Masker")
st.caption("Detect and mask PII with GLiNER + deterministic regex backstops. Export masked text and details.")

st.markdown(
        """
        <div class="card">
            <b>How to use</b>
            <div class="small-muted">1) Paste text  •  2) Click Mask  •  3) Download masked output</div>
        </div>
        """,
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

    model_options = [
        "nvidia/gliner-PII",
        "urchade/gliner_multi_pii-v1",
    ]

    env_model = os.getenv("GLINER_MODEL", "").strip()
    default_model_index = model_options.index(env_model) if env_model in model_options else 0

    model_name = st.selectbox(
        "Model",
        options=model_options,
        index=default_model_index,
        help="Pick the GLiNER model to use for entity detection.",
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


tab_mask, = st.tabs(["Mask"])

with tab_mask:
    st.write("")
    col_left, col_right = st.columns((1, 1))

    examples = [
        """You are an assistant helping review a customer-reported issue.

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

        ,
        """On 18 February 2025, customer Hannah Louise Mercer (born 11 November 1991) contacted support to report an unfamiliar bank transfer. She stated that she noticed a payment of £1,875 leaving her account at Westbridge Financial Ltd shortly after logging in from her home address at 42 Oakfield Road, Reading, RG1 4PX.

Hannah can be contacted via hannah.mercer91@outlook.com or on her mobile number 07854 662913, although she mentioned that her work phone 0118 496 0821 is only answered during office hours.

The transaction, reference WB-TRX-558201, was processed successfully on 16/02/2025 from account number 29174638 with sort code 40-22-17 and IBAN GB31BARC40221729174638.

System logs show the login originated from IP address 92.184.33.71 at approximately 01:42 AM, which Hannah confirmed was outside her usual activity pattern.

She also confirmed that a debit card ending 4821, expiring 07/26, is still in her possession and that she has not shared her credentials with anyone else.
""",
    ]

    def _load_example() -> None:
        idx = int(st.session_state.get("example_idx", 0))
        st.session_state["input_text"] = examples[idx % len(examples)]
        st.session_state["example_idx"] = (idx + 1) % len(examples)
        st.session_state["last_result"] = None

    def _clear_all() -> None:
        st.session_state["input_text"] = ""
        st.session_state["last_result"] = None

    with col_left:
        st.subheader("Input")
        with st.container(border=True):
            input_text = st.text_area(
                "Text to mask",
                key="input_text",
                height=320,
                placeholder="Paste text containing PII here...",
            )

            c1, c2 = st.columns((1, 1))
            with c1:
                run = st.button("Mask", type="primary", use_container_width=True)
            with c2:
                st.button(
                    "Load example",
                    use_container_width=True,
                    on_click=_load_example,
                )

            st.button(
                "Clear",
                use_container_width=True,
                on_click=_clear_all,
                help="Clears the input and any previous results.",
            )

    with col_right:
        st.subheader("Output")
        with st.container(border=True):
            if run:
                if not input_text.strip():
                    st.warning("Please paste some text first.")
                else:
                    started = time.perf_counter()
                    with st.spinner("Masking text..."):
                        try:
                            masked_text, mapping, scores, spans = _run_masking(
                                text=input_text,
                                model_name=model_name,
                                threshold=threshold,
                            )
                        except Exception as e:
                            st.error("Masking failed.")
                            st.exception(e)
                            st.session_state["last_result"] = None
                        else:
                            elapsed_ms = int((time.perf_counter() - started) * 1000)
                            st.session_state["last_result"] = {
                                "masked_text": masked_text,
                                "mapping": mapping,
                                "scores": scores,
                                "spans": spans,
                                "elapsed_ms": elapsed_ms,
                            }

            last = st.session_state.get("last_result")
            if last is None:
                st.info("Paste text on the left, then click ‘Mask’.", icon="ℹ️")
            else:
                masked_text = last["masked_text"]
                mapping = last["mapping"]
                scores = last["scores"]
                spans = last["spans"]
                elapsed_ms = int(last.get("elapsed_ms", 0))

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

                with st.expander("Details", expanded=True):
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
