# Mini Chatbot (Ollama LLaMA 3.1)
Local-first chatbot demos that talk to an Ollama-hosted `llama3.1` model. The scripts progress from single prompts to context retention, parameter tuning, streaming, prompt templates, and simple memory trimming.

## Requirements
- Python 3.10+
- [Ollama](https://ollama.com/) running at `http://localhost:11434`
- Model pulled locally: `ollama pull llama3.1`

Install Python deps:
```bash
pip install -r requirements.txt
```

## Project layout
- `first_message.py` - single-turn prompt via `/api/generate`.
- `model_context.py` - keeps full chat history with `/api/chat`.
- `parameters.py` - chat with tuned `temperature`, `top_p`, and `max_tokens`.
- `streaming_chat.py` - streams tokens from `/api/chat` and prints them live.
- `memory_trimming.py` - streaming chat plus message-count trimming to keep history small.
- `system_prompts.py` - catalog of prompt templates (JSON-only, code-only, teacher mode, etc.) and a structured-answer default.
- `final/main.py` - consolidated chatbot with streaming, trimming, and skeletons for token-based and semantic memory.

## Quick start
1) Ensure Ollama is running: `ollama serve` (and pull `llama3.1` once).
2) Run any script, e.g.:
```bash
python first_message.py
python model_context.py
python parameters.py
python streaming_chat.py
python memory_trimming.py
python system_prompts.py
python final/main.py
```
3) Type your question when prompted; responses print in the terminal. Streaming variants display tokens as they arrive.

## Customizing the bot
- **Prompting:** Edit `SYSTEM_PROMPT` in `system_prompts.py` or `final/main.py` to change persona or response structure.
- **Sampling params:** Adjust `temperature`, `top_p`, and `max_tokens` in each file's `options` dict.
- **Context management:** Tweak `MAX_HISTORY` in `memory_trimming.py` or `final/main.py`. Token-based trimming and semantic memory skeletons are included in `final/main.py` for future expansion.

## Troubleshooting
- Connection errors usually mean Ollama is not running or reachable at `localhost:11434`.
- Long answers or high temperatures can be slow; lower `max_tokens` to keep responses snappy.
- If you change the model name, update every `model` field in the scripts.
