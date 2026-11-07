"""app.py - Streamlit GUI (Acceptance criterion 2). Run: streamlit run app.py"""
import os
import tempfile

import streamlit as st

import agents as A
from rag import ingest_pdf

st.set_page_config(page_title="Paper-to-Presentation & Viva Agent", page_icon="🎓", layout="wide")
ss = st.session_state
for k, v in dict(vs=None, msgs=[], last_intent=None, viva=[], paper="").items():
    ss.setdefault(k, v)

# ---------------- Sidebar: model + paper ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("LLM provider", ["Ollama (local)", "Google Gemini (API)"])
    default_model = "llama3.1:8b" if provider.startswith("Ollama") else "gemini-3.8-flash"
    model = st.text_input("Model name", default_model)
    api_key = ""
    if provider.startswith("Google"):
        api_key = st.text_input("Gemini API key", type="password",
                                value=os.getenv("GOOGLE_API_KEY", ""))
    use_reflection = st.checkbox("Reflection (grounding check)", True)
    n_slides = st.slider("Slides", 5, 15, 8)
    n_q = st.slider("Viva questions", 4, 15, 8)
    chat_h = st.slider("Chat window height (px)", 300, 800, 480, 20)

    st.header("📄 Paper")
    pdf = st.file_uploader("Upload research paper (PDF)", type="pdf")
    if pdf and st.button("Process paper", type="primary"):
        with st.spinner("Chunking & embedding into ChromaDB..."):
            data = pdf.getvalue()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(data)
            ss.vs, n = ingest_pdf(f.name, data)
            ss.paper, ss.msgs, ss.last_intent, ss.viva = pdf.name, [], None, []
        st.success(f"Indexed {n} chunks")
    if ss.paper:
        st.caption(f"Active paper: {ss.paper}")


def llm():
    return A.get_llm(provider, model, api_key)


# ---------------- Task handlers (routing targets) ----------------
def do_chat(req, route):
    return A.chain("chat", llm()).invoke({"request": req}), None, []


def do_summary(req, route):
    out, docs = A.summarizer_agent(ss.vs, llm(), use_reflection)
    return out, None, docs


def do_concept(req, route):
    out, docs = A.explain_concept(ss.vs, llm(), route.get("query") or req, use_reflection)
    return out, None, docs


def do_slides(req, route):
    slides, _, docs = A.slide_generator_agent(ss.vs, llm(), n_slides)
    title = ss.paper.replace(".pdf", "")
    return A.slides_to_markdown(slides), A.build_pptx(title, slides), docs


def do_viva(req, route):
    qs, docs = A.viva_agent(ss.vs, llm(), n_q)
    ss.viva = qs
    return A.viva_to_markdown(qs), None, docs


HANDLERS = {"chat": do_chat, "summary": do_summary, "concept": do_concept,
            "slides": do_slides, "viva": do_viva}


def handle(req):
    route = A.route_intent(llm(), req, ss.last_intent)           # 1) detect intent
    intent = route["intent"]
    if intent != "chat" and ss.vs is None:
        return {"role": "assistant", "content": "Please upload and process a paper first."}
    ss.last_intent = intent                                      # retain task in context
    text, pptx, docs = HANDLERS[intent](req, route)              # 2) route (+ retrieval inside)
    trace = (f"**Intent:** `{intent}` | **Retrieval:** "
             f"{'yes' if route.get('needs_retrieval') else 'no (answered directly)'} | "
             f"**Chunks used:** {len(docs)}")
    return {"role": "assistant", "content": text, "pptx": pptx, "trace": trace,
            "sources": [A.format_docs([d]) for d in docs]}


def render(m, idx):
    st.markdown(m["content"])
    if m.get("trace"):
        with st.expander("🔎 Agent trace & retrieved sources"):
            st.markdown(m["trace"])
            for s in m.get("sources", []):
                st.text(s[:500])
    if m.get("pptx"):
        st.download_button("⬇️ Download .pptx", m["pptx"], file_name="slides.pptx",
                           mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                           key=f"dl{idx}")


# ---------------- Main UI ----------------
st.title("🎓 Paper-to-Presentation & Viva Agent")
tab_chat, tab_viva, tab_prompts = st.tabs(["💬 Assistant", "🎤 Viva practice", "🧠 Prompts used"])

with tab_chat:
    c = st.columns(4)
    pending = None
    if c[0].button("📝 Summarize"): pending = "Summarize this paper"
    if c[1].button("📊 Make slides"): pending = "Create a slide deck from this paper"
    if c[2].button("🎤 Viva questions"): pending = "Generate likely viva questions with model answers"
    if c[3].button("💡 Explain a concept"): pending = "Explain the core method of this paper"

    # Scrollable chat window: header, tabs and buttons above stay fixed in place
    chat_box = st.container(height=chat_h, border=True)
    user_text = st.chat_input("Ask: summarize / make slides / explain X / viva practice ...") or pending

    with chat_box:
        for i, m in enumerate(ss.msgs):
            with st.chat_message(m["role"]):
                render(m, i)
        if user_text:
            ss.msgs.append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.markdown(user_text)
            with st.chat_message("assistant"):
                with st.spinner("Routing and thinking..."):
                    try:
                        msg = handle(user_text)
                    except Exception as e:
                        msg = {"role": "assistant", "content": f"⚠️ Error: {e}"}
                render(msg, len(ss.msgs))
            ss.msgs.append(msg)

with tab_viva:
    if not ss.viva:
        st.info("Generate viva questions first (use the 🎤 button in the Assistant tab).")
    else:
        q = st.selectbox("Pick a question", [x["question"] for x in ss.viva])
        ans = st.text_area("Your answer")
        if st.button("Evaluate my answer") and ans.strip():
            with st.spinner("Examiner is grading..."):
                st.markdown(A.evaluate_answer(ss.vs, llm(), q, ans))

with tab_prompts:
    st.caption("System and user prompts used by each agent (criterion 3).")
    for name, (sys_p, usr_p) in A.PROMPTS.items():
        with st.expander(name):
            st.markdown("**System prompt**"); st.code(sys_p, language="text")
            st.markdown("**User prompt**"); st.code(usr_p, language="text")