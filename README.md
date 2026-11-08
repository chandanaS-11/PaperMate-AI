# Paper-to-Presentation & Viva Agent

## Setup
python -m venv venv && source venv/bin/activate   (Windows: venv\Scripts\activate)
pip install -r requirements.txt

## LLM option A: local (free)
Install Ollama from https://ollama.com, then:  ollama pull llama3.1:8b

## LLM option B: API
Get a Gemini key from https://aistudio.google.com and paste it in the sidebar.

## Run
streamlit run app.py

## Configuring Gemini (key + model)
1. Get a free key: https://aistudio.google.com/apikey
2. Either paste it in the app sidebar, or copy `.env.example` to `.env` and edit it.
3. Choose a model in the sidebar dropdown (or "Custom..." to type any current model name).
   Model names change over time: https://ai.google.dev/gemini-api/docs/models
4. Click "Test connection" to verify before using the assistant.
Do not commit `.env` (add it to .gitignore).
