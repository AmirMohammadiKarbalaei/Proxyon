import os
import time
from typing import Any, Dict, List, Optional, Tuple

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

            /* Custom chat avatars and bubbles */
            .chat-avatar {
                width: 36px;
                height: 36px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px solid rgba(15, 23, 42, 0.10);
                background: #FFFFFF;
                font-size: 1.1rem;
                flex-shrink: 0;
            }
            .chat-marker { display: none; }
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-marker.user) {
                background: #DBEAFE !important;
                border-color: rgba(59, 130, 246, 0.3) !important;
                border-radius: 14px !important;
            }
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-marker.assistant) {
                background: #F3E8FF !important;
                border-color: rgba(168, 85, 247, 0.3) !important;
                border-radius: 14px !important;
            }
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-marker) p { margin: 0.2rem 0; }
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-marker) ul { margin: 0.3rem 0 0.3rem 1.1rem; }
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-marker) p:first-child { margin-top: 0; }
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-marker) p:last-child { margin-bottom: 0; }

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
if "ui_history" not in st.session_state:
    st.session_state["ui_history"] = []
if "api_history" not in st.session_state:
    st.session_state["api_history"] = []
if "unmask_mapping" not in st.session_state:
    st.session_state["unmask_mapping"] = {}
if "person_alias_map" not in st.session_state:
    st.session_state["person_alias_map"] = {}
if "chat_mapping" not in st.session_state:
    st.session_state["chat_mapping"] = {}
if "processing" not in st.session_state:
    st.session_state["processing"] = False


import google.generativeai as genai

# --- Gemini Client ---
def _get_gemini_client():
    # Assumes GEMINI_API_KEY is set in Streamlit's secrets
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("Please add `GEMINI_API_KEY = '...'` to your Streamlit secrets.")
        st.stop()
    
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    
    model_instruction = """Note: The following message contains intentionally masked information using placeholders 
(e.g. [PERSON_1], [ACCOUNT_ID_1], [TRANSACTION_ID_1]).

Please:
- Treat each placeholder as a consistent anonymous entity.
- Refer back to the same placeholder when discussing actions or events related to it.
- Do not invent names, identities, or additional personal details.
- Use the masked information as provided to respond to the user’s query."""
    
    return genai.GenerativeModel(
        'gemini-2.5-flash',
        system_instruction=model_instruction
    )

def _unmask_text(text: str, mapping: Dict[str, str]) -> str:
    """Replaces placeholder tags with their original values."""
    # Sort by length descending to handle overlapping keys correctly (e.g., PERSON_10 vs PERSON_1)
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    for key in sorted_keys:
        text = text.replace(mapping[key],key)
    return text

def _get_gemini_response(client, history: List[Dict[str, Any]]):
    """Calls Gemini API and returns the text response."""
    try:
        response = client.generate_content(history)
        return response.text
    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        # Return error message to be displayed in both UI and debug views
        return f"Error: API call failed. Details: {e}"

def _invert_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """Inverts a mapping from {original: tag} to {tag: original}."""
    return {v: k for k, v in mapping.items()}

# --- UI ---

st.title("PII Masker & Chat")
st.caption("Mask PII and chat with a Gemini-powered assistant.")

with st.sidebar:
    st.markdown("### Settings")

    model_options = [
        # "nvidia/gliner-PII",
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

tab_mask, tab_chat, tab_debug = st.tabs(["Mask", "Chat", "Debug"])


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

    try:
        return GLiNER.from_pretrained(model_name)
    except Exception as e:
        msg = str(e)
        hint = ""
        # Common in Streamlit Cloud: model is gated/private and local dev has cached/token access.
        if any(k in msg.lower() for k in ["401", "403", "gated", "private", "not authorized", "requires authentication"]):
            hint = (
                "\n\nThis looks like an authentication / gated-model error. "
                "If you're deploying on Streamlit Community Cloud, add a Hugging Face token as a Secret "
                "(Settings → Secrets) and set it as `HF_TOKEN` (or `HUGGINGFACEHUB_API_TOKEN`). "
                "Also ensure you've accepted the model's license/terms on Hugging Face for that account."
            )
        raise RuntimeError(f"Failed to load GLiNER model '{model_name}'.{hint}\n\nOriginal error: {msg}") from e


def _run_masking(
    text: str,
    model_name: str,
    threshold: float,
    person_alias_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, str], Dict[str, float], List[Dict[str, Any]], Dict[str, str]]:
    from pii_masker.masking import mask_with_gliner

    model = _load_gliner(model_name)
    return mask_with_gliner(
        text=text,
        model_name_or_obj=model,
        threshold=threshold,
        person_alias_to_tag=person_alias_map,
    )


with tab_mask:
    st.markdown(
        """
        <div class="card">
            <b>How to use</b>
            <div class="small-muted">1) Paste text  •  2) Click Mask  •  3) Download masked output</div>
        </div>
        <br>
        """,
        unsafe_allow_html=True,
    )
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
                            (
                                masked_text,
                                mapping,
                                scores,
                                spans,
                                updated_alias_map,
                            ) = _run_masking(
                                text=input_text,
                                model_name=model_name,
                                threshold=threshold,
                                person_alias_map=st.session_state.person_alias_map,
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
                            st.session_state.person_alias_map = updated_alias_map

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

with tab_chat:
    def _render_chat_row(role: str, text: str) -> None:
        role = "user" if role == "user" else "assistant"

        if role == "user":
            # User: avatar left, bubble right, aligned to the left side
            col_msg, col_spacer = st.columns([0.8, 0.2], gap="small")
            with col_msg:
                cols = st.columns([0.6, 9.4], gap="small")
                with cols[0]:
                    st.markdown('<div class="chat-avatar">👤</div>', unsafe_allow_html=True)
                with cols[1]:
                    bubble_html = f'''
                    <div style="background: #DBEAFE; border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 14px; padding: 10px 14px; margin-bottom: 12px;">
                    
{text}</div>


                    '''
                    st.markdown(bubble_html, unsafe_allow_html=True)
            with col_spacer:
                st.write("")
        else:
            # Assistant: bubble left, avatar right, aligned to the right side
            col_spacer, col_msg = st.columns([0.2, 0.8], gap="small")
            with col_spacer:
                st.write("")
            with col_msg:
                cols = st.columns([9.4, 0.6], gap="small")
                with cols[0]:
                    bubble_html = f'''
                    <div style="background: #F3E8FF; border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 14px; padding: 10px 14px; margin-bottom: 12px;">
                    {text} </div>

  
                    '''
                    st.markdown(bubble_html, unsafe_allow_html=True)
                with cols[1]:
                    st.markdown('<div class="chat-avatar">🤖</div>', unsafe_allow_html=True)

    # Display unmasked chat history
    for msg in st.session_state.ui_history:
        _render_chat_row(msg.get("role", "assistant"), msg.get("parts", ""))

    # Show processing indicator above chat input
    if st.session_state.get("processing", False):
        st.markdown("**✨ Thinking...**")
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # 1. Add user's unmasked message to UI history
        st.session_state.ui_history.append({"role": "user", "parts": prompt})
        st.session_state["processing"] = True
        st.rerun()

    # Process the last message if we're in processing state
    if st.session_state.get("processing", False) and len(st.session_state.ui_history) > 0:
        last_msg = st.session_state.ui_history[-1]
        if last_msg["role"] == "user":
            # 2. Mask the user's prompt
            (
                masked_prompt,
                mapping,
                _,
                _,
                updated_alias_map,
            ) = _run_masking(
                text=last_msg["parts"],
                model_name=model_name,
                threshold=threshold,
                person_alias_map=st.session_state.person_alias_map,
            )
            st.session_state.person_alias_map = updated_alias_map
            
            # 3. Update the session's unmasking map
            inverted_mapping = _invert_mapping(mapping)
            print("Inverted mapping for this prompt:", inverted_mapping)
            st.session_state.chat_mapping.update(inverted_mapping)

            # 4. Append masked prompt to API history
            st.session_state.api_history.append({"role": "user", "parts": masked_prompt})

            # 5. Call Gemini with the full masked history
            gemini_client = _get_gemini_client()
            masked_response = _get_gemini_response(gemini_client, st.session_state.api_history)

            # 6. Append masked response to API history
            st.session_state.api_history.append({"role": "model", "parts": masked_response})

            # 7. Unmask the response for the UI
            unmasked_response = _unmask_text(masked_response, st.session_state.chat_mapping)

            # 8. Add unmasked response to UI history
            st.session_state.ui_history.append({"role": "assistant", "parts": unmasked_response})
            
            # Clear processing flag and rerun
            st.session_state["processing"] = False
            st.rerun()
            


with tab_debug:
    st.markdown("### Masked API History")
    st.caption("This is the exact data sent to and received from the Gemini API.")
    
    if not st.session_state.api_history:
        st.info("No conversation yet. Start a chat in the 'Chat' tab.")
    else:
        # Display the masked conversation in a chat-like format
        for msg in st.session_state.api_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["parts"])

    st.markdown("---")
    st.markdown("### Unmasking Map")
    st.caption("Current key-value pairs used to unmask responses.")
    if not st.session_state.chat_mapping:
        st.info("No entities have been masked yet.")
    else:
        st.json(st.session_state.chat_mapping)
