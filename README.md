# 🤖 Smart Chatbot — RAG + Web Search + Memory

A Streamlit chatbot powered by **Google Gemini**, with:
- 📄 **RAG** (Retrieval-Augmented Generation) — upload your own PDF/TXT files
- 🌐 **Web Search** — live results via SerpAPI as fallback
- 🧠 **Memory** — remembers recent conversation turns
- 🔁 **Smart routing** — auto-picks the best source for every question

---

## How it works

```
User question
      ↓
[1] Search uploaded docs (RAG via FAISS)
      ↓ (not found?)
[2] Ask Gemini AI
      ↓ (not relevant?)
[3] Web Search (SerpAPI → Gemini summarises)
```

---

## 🚀 Deploy on Streamlit Community Cloud

1. **Fork / push this folder to a new GitHub repository**

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**

3. Connect your GitHub repo, set **Main file path** to `app.py`

4. Open **Advanced settings → Secrets** and add:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
SERPAPI_KEY    = "your-serpapi-key"
```

5. Click **Deploy** 🎉

---

## 🔑 Getting API keys

| Key | Where to get |
|-----|-------------|
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) — free tier available |
| `SERPAPI_KEY` | [serpapi.com/manage-api-key](https://serpapi.com/manage-api-key) — 100 free searches/month |

---

