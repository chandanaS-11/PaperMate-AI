# 🎓 Paper-to-Presentation & Viva Agent

An AI assistant that turns a research paper (PDF) into a **summary**, a **slide deck (.pptx)**, **concept explanations** and **viva practice**, all grounded in the paper's own text.

🔗 **Live demo:** [https://papermate-aigit-knqjatzckyuccmybnvzepy.streamlit.app/](https://papermate-aigit-knqjatzckyuccmybnvzepy.streamlit.app/)
*(If asked for an API key, get a free one at https://aistudio.google.com/apikey and paste it in the sidebar.)*

## 💡 Idea

Preparing a presentation and anticipating viva questions from a paper is slow. This agentic RAG app stores the paper in a vector database, detects what the user wants, retrieves the relevant sections, and generates a grounded answer with citations.

```
PDF → chunks → embeddings → ChromaDB
Request → Intent Router → Summarizer | Slide Generator | Concept Explainer | Viva Agent
                          (each retrieves from ChromaDB, then answers)
```

**Stack:** Streamlit · Python · Google Gemini API · LangChain (prompt templates + LCEL) · ChromaDB · python-pptx


## 🚀 Run Locally

```bash
git clone PASTE_YOUR_REPO_URL_HERE
cd paper_viva_agent
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env           # Windows: copy .env.example .env
# edit .env and add: GOOGLE_API_KEY=your_key

streamlit run app.py
```

Open http://localhost:8501 (the first run downloads a ~90 MB embedding model).

## 🖱️ Use It

1. Sidebar: enter your Gemini key, click **Test connection**.
2. Upload a PDF, click **Process paper**.
3. Ask: `Summarize this paper` · `Make 8 slides` · `Explain <concept>` · `Give me viva questions`.
4. **Viva practice** tab: answer a question and get graded.
5. **Prompts used** tab: see every system and user prompt.

> Send one request at a time and wait for the answer.

## ✅ Assignment Criteria

| # | Criterion | Where |
|---|---|---|
| 1 | LLM integration | `get_llm()` in `agents.py` |
| 2 | Interactive GUI | `app.py` (Streamlit) |
| 3 | Prompt engineering | `PROMPTS` in `agents.py` |
| 4 | LangChain + LCEL | `chain()` = `prompt \| llm \| parser` |
| 5 | Agentic RAG | Router decides retrieval; agents ground answers |
| 6 | Intent routing | `route_intent()` + handlers in `app.py` |
| 7 | Vector database | ChromaDB in `rag.py` |

## ⚠️ Limitations

Free Gemini tier rate limits · text-only PDF parsing (no figures or tables) · temporary vector store on free hosting.

