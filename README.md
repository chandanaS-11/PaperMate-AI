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