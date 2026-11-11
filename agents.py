"""agents.py - prompts, LCEL chains, intent router and the four agents.

Criteria covered here:
  3 Prompt engineering  -> PROMPTS dict (system + user prompts; few-shot, CoT, reflection)
  4 LangChain/LCEL      -> chain() = ChatPromptTemplate | LLM | OutputParser
  5 Agentic RAG         -> router decides needs_retrieval; agents retrieve then ground answers
  6 Intent routing      -> route_intent() + dispatch in app.py
"""
import json
import re
import time
from io import BytesIO

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from rag import format_docs, retrieve, retrieve_multi

INTENTS = ["summary", "slides", "concept", "viva", "chat"]

# --------------------------------------------------------------------------
# PROMPTS: every entry is (system_prompt, user_prompt). Shown in the UI tab.
# NOTE: literal JSON braces are doubled {{ }} because these are templates.
# --------------------------------------------------------------------------
PROMPTS = {
    # Few-shot intent classification
    "router": (
        """You are the intent router of a research-paper assistant.
Classify the user's request into exactly ONE intent:
- summary: wants an overview / summary of the paper
- slides: wants a presentation / slide deck / PPT
- concept: wants a specific concept, term, method, equation or result explained
- viva: wants examiner questions, viva practice or a mock interview
- chat: greeting, question about the assistant itself, or anything not needing the paper
If the request is a vague follow-up (e.g. "make it shorter", "more detail"), reuse LAST_INTENT.
Also output needs_retrieval (false only for chat) and a short vector-search "query"
(empty string for summary/slides/viva).
Return ONLY JSON: {{"intent": "...", "needs_retrieval": true, "query": "..."}}

Examples:
User: Give me a quick overview of this paper
{{"intent": "summary", "needs_retrieval": true, "query": ""}}
User: Make a 10 slide deck for my seminar
{{"intent": "slides", "needs_retrieval": true, "query": ""}}
User: What is the attention mechanism they use?
{{"intent": "concept", "needs_retrieval": true, "query": "attention mechanism"}}
User: Ask me tough questions like an examiner
{{"intent": "viva", "needs_retrieval": true, "query": ""}}
User: hi, what can you do?
{{"intent": "chat", "needs_retrieval": false, "query": ""}}""",
        "LAST_INTENT: {last_intent}\nUser request: {request}",
    ),
    "chat": (
        "You are a friendly academic assistant. You can summarize a research paper, build a slide "
        "deck, explain concepts, and run viva practice. Answer briefly and invite the user to upload "
        "a paper or pick a task.",
        "{request}",
    ),
    "summarizer": (
        """You are an expert academic summarizer. Use ONLY the CONTEXT (excerpts of the paper).
Write a concise sectioned summary in markdown with these headings:
## Problem & Motivation
## Proposed Method
## Experiments & Results
## Limitations & Conclusion
Keep each section to 3-5 bullets. Cite evidence as [section, p.N]. If something is not in the
context, write "Not found in retrieved text" - never invent facts.""",
        "CONTEXT:\n{context}\n\nWrite the summary.",
    ),
    "extractor": (
        """You are a research analyst. Using ONLY the CONTEXT, think step by step (first find the
problem, then what is new, then how it works, then the evidence) and then output ONLY JSON:
{{"title_guess": "...", "contributions": ["..."], "methods": ["..."], "results": ["..."],
  "key_terms": [{{"term": "...", "definition": "..."}}]}}
Give 3-5 contributions, 3-5 methods, 3-5 results (include numbers where the paper does) and 4-6 key terms.""",
        "CONTEXT:\n{context}",
    ),
    "slides": (
        """You are a presentation designer. Turn the extracted concepts into a slide deck.
Rules: {n_slides} slides total; flow = Title, Motivation, Contributions, Method (1-2 slides),
Results, Limitations, Conclusion/Future work; max 5 bullets per slide, each under 15 words;
add concise speaker notes grounded in the CONTEXT. Output ONLY JSON:
{{"slides": [{{"title": "...", "bullets": ["..."], "notes": "..."}}]}}""",
        "EXTRACTED CONCEPTS:\n{concepts}\n\nCONTEXT:\n{context}",
    ),
    "concept": (
        """You are a patient professor. Use ONLY the CONTEXT.
Think step by step privately (definition -> role in this paper -> why it matters), then answer with:
**Plain-English definition**, **How this paper uses it**, **Analogy**.
Cite [section, p.N]. If the context does not cover the concept, say so honestly.""",
        "CONTEXT:\n{context}\n\nExplain: {query}",
    ),
    "viva": (
        """You are a strict but fair examiner preparing a viva on the paper. Using ONLY the CONTEXT,
write {n_questions} likely examiner questions covering: motivation, method, experiments, limitations,
and "what if" challenges. For each give a model answer (3-5 sentences, grounded in the paper).
Output ONLY JSON:
{{"questions": [{{"category": "...", "question": "...", "model_answer": "...", "follow_up": "..."}}]}}
Example item: {{"category": "Limitations", "question": "Why did you not evaluate on larger datasets?",
"model_answer": "The paper restricts itself to ... because ...", "follow_up": "How would results change at scale?"}}""",
        "CONTEXT:\n{context}",
    ),
    "evaluator": (
        """You are an examiner grading a student's viva answer against the paper's CONTEXT.
Return markdown: **Score: x/10**, **What was good**, **What was missing**, **Improved answer**.""",
        "CONTEXT:\n{context}\n\nQuestion: {question}\nStudent answer: {answer}",
    ),
    # Reflection: self-check for grounding
    "reflect": (
        """You are a strict fact-checker. Compare the DRAFT with the CONTEXT. Remove or correct any
claim that the CONTEXT does not support. Keep the same format. Do not add new facts.
Return only the corrected draft.""",
        "CONTEXT:\n{context}\n\nDRAFT:\n{draft}",
    ),
}


# --------------------------------------------------------------------------
# LLM + LCEL helpers
# --------------------------------------------------------------------------
def get_llm(provider: str, model: str, api_key: str = "", temperature: float = 0.2):
    if provider == "Ollama (local)":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature,
                                  max_retries=1, timeout=45)


_last_call = [0.0]
MIN_GAP = 4.0  # seconds between LLM calls (protects free-tier requests-per-minute)


def _guarded(llm):
    """Throttle calls and auto-retry on 429 rate limits (waits the delay Google asks for)."""
    def call(prompt_value):
        for attempt in range(3):
            wait_gap = MIN_GAP - (time.time() - _last_call[0])
            if wait_gap > 0:
                time.sleep(wait_gap)
            _last_call[0] = time.time()
            try:
                return llm.invoke(prompt_value)
            except Exception as e:
                t = str(e)
                limited = "429" in t or "RESOURCE_EXHAUSTED" in t
                daily = "PerDay" in t or "limit: 0" in t
                if limited and not daily and attempt < 2:
                    m = re.search(r"retry(?: in|Delay'?:?\s*')\s*([\d.]+)", t)
                    time.sleep(min(float(m.group(1)) + 1 if m else 20, 45))
                    continue
                raise
    return RunnableLambda(call)


def chain(name: str, llm, json_out: bool = False):
    """LCEL: PromptTemplate | (throttled) LLM | OutputParser."""
    system, user = PROMPTS[name]
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
    return prompt | _guarded(llm) | (JsonOutputParser() if json_out else StrOutputParser())


def reflect(llm, draft: str, context: str) -> str:
    try:
        return chain("reflect", llm).invoke({"draft": draft, "context": context})
    except Exception:
        return draft


# --------------------------------------------------------------------------
# Intent router
# --------------------------------------------------------------------------
def _keyword_route(request: str, last_intent):
    r = request.lower()
    rules = [("slides", ["slide", "ppt", "presentation", "deck"]),
             ("viva", ["viva", "examiner", "mock", "interview", "questions"]),
             ("summary", ["summar", "overview", "tl;dr", "abstract"]),
             ("concept", ["explain", "what is", "what does", "meaning", "define"])]
    for intent, words in rules:
        if any(w in r for w in words):
            return {"intent": intent, "needs_retrieval": True, "query": request}
    return {"intent": last_intent or "chat", "needs_retrieval": last_intent not in (None, "chat"),
            "query": request}


def route_intent(llm, request: str, last_intent):
    try:
        res = chain("router", llm, True).invoke(
            {"request": request, "last_intent": last_intent or "none"})
        if res.get("intent") in INTENTS:
            res.setdefault("query", "")
            return res
    except Exception as e:
        # Only fall back to keywords when the model's JSON was malformed.
        # API/network errors (timeout, bad key...) must surface immediately.
        if "Parse" not in type(e).__name__:
            raise
    return _keyword_route(request, last_intent)  # robust fallback for small local models


# --------------------------------------------------------------------------
# The four agents (each retrieves from ChromaDB, then generates)
# --------------------------------------------------------------------------
SUMMARY_QUERIES = ["problem statement and motivation", "proposed method architecture",
                   "experimental setup datasets and results", "limitations and conclusion"]
CONCEPT_QUERIES = ["main contributions of this paper", "method and algorithm details",
                   "quantitative results and comparison with baselines", "key definitions and terminology"]
VIVA_QUERIES = ["motivation and research gap", "method design choices", "experiments and metrics",
                "limitations and future work", "assumptions"]


def summarizer_agent(vs, llm, use_reflection=True):
    docs = retrieve_multi(vs, SUMMARY_QUERIES, k=3)
    ctx = format_docs(docs)
    out = chain("summarizer", llm).invoke({"context": ctx})
    if use_reflection:
        out = reflect(llm, out, ctx)
    return out, docs


def concept_extractor_agent(vs, llm):
    docs = retrieve_multi(vs, CONCEPT_QUERIES, k=3)
    data = chain("extractor", llm, True).invoke({"context": format_docs(docs)})
    return data, docs


def explain_concept(vs, llm, query, use_reflection=True):
    docs = retrieve(vs, query, k=5)
    ctx = format_docs(docs)
    out = chain("concept", llm).invoke({"context": ctx, "query": query})
    if use_reflection:
        out = reflect(llm, out, ctx)
    return out, docs


def slide_generator_agent(vs, llm, n_slides=8):
    concepts, docs = concept_extractor_agent(vs, llm)       # agent -> agent hand-off
    res = chain("slides", llm, True).invoke({
        "n_slides": n_slides, "concepts": json.dumps(concepts)[:4000],
        "context": format_docs(docs)[:5000]})
    slides = res["slides"] if isinstance(res, dict) else res
    return slides, concepts, docs


def viva_agent(vs, llm, n_questions=8):
    docs = retrieve_multi(vs, VIVA_QUERIES, k=3)
    res = chain("viva", llm, True).invoke({"n_questions": n_questions, "context": format_docs(docs)})
    return (res["questions"] if isinstance(res, dict) else res), docs


def evaluate_answer(vs, llm, question, answer):
    docs = retrieve(vs, question, k=4)
    return chain("evaluator", llm).invoke(
        {"context": format_docs(docs), "question": question, "answer": answer})


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def slides_to_markdown(slides) -> str:
    parts = []
    for i, s in enumerate(slides, 1):
        bullets = "\n".join(f"- {b}" for b in s.get("bullets", []))
        parts.append(f"### Slide {i}: {s.get('title','')}\n{bullets}\n\n*Notes: {s.get('notes','')}*")
    return "\n\n".join(parts)


def build_pptx(deck_title: str, slides) -> bytes:
    from pptx import Presentation
    prs = Presentation()
    first = prs.slides.add_slide(prs.slide_layouts[0])
    first.shapes.title.text = deck_title
    first.placeholders[1].text = "Auto-generated by Paper-to-Presentation Agent"
    for s in slides:
        sl = prs.slides.add_slide(prs.slide_layouts[1])
        sl.shapes.title.text = s.get("title", "")
        tf = sl.placeholders[1].text_frame
        tf.clear()
        for j, b in enumerate(s.get("bullets", [])):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.text = b
        sl.notes_slide.notes_text_frame.text = s.get("notes", "")
    buf = BytesIO()
    prs.save(buf)
    return buf.getvalue()


def viva_to_markdown(qs) -> str:
    return "\n\n".join(
        f"**Q{i}. [{q.get('category','')}]** {q.get('question','')}\n\n"
        f"> **Model answer:** {q.get('model_answer','')}\n\n*Follow-up:* {q.get('follow_up','')}"
        for i, q in enumerate(qs, 1))
