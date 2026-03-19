# Deploying the Journal Recommender on Render

Step-by-step guide to put the Journal Recommender online for free using [Render](https://render.com).

---

## Overview

**Architecture:**
- Render hosts the Python backend (FastAPI) + frontend (`index.html`)
- Pre-generated embeddings are included in the repo (they're only ~15MB)
- Runtime query embedding uses an API provider (no heavy model in memory)
- LLM re-ranking uses your Gemini API key

**Cost:**
- Render free tier: $0/month (spins down after 15 min inactivity → ~30s cold start)
- Render Starter: $7/month (always on, no cold start)
- OpenAI embeddings: ~$0.02 per 1M tokens (essentially free)
- Gemini API: free tier covers moderate usage

---

## Prerequisites

- A [GitHub](https://github.com) account
- A [Render](https://render.com) account (sign up with GitHub)
- Your Gemini API key (from https://aistudio.google.com/apikey)
- An OpenAI API key (from https://platform.openai.com/api-keys) — or Cohere/Voyage

---

## Step 1: Prepare your project for GitHub

Your `journal-recommender` folder should contain:

```
journal-recommender/
├── app.py                          ← Updated version (serves frontend + /report endpoint)
├── recommender.py
├── vector_store.py
├── report_generator.py             ← NEW: generates Word reports
├── 05_generate_embeddings.py
├── index.html                      ← NEW: the dark-themed frontend
├── requirements-deploy.txt         ← NEW: lightweight deps (no sentence-transformers)
├── render.yaml                     ← NEW: Render configuration
├── .gitignore                      ← NEW
│
├── data/
│   ├── journal_database.json       ← or journal_database_final.json
│   ├── journal_embeddings.npz      ← Pre-generated embeddings (~15MB)
│   └── embedding_metadata.json
│
├── enrichment/                     ← Optional, can exclude from repo
│   └── ...
│
└── (other files like setup.sh, QUICKSTART.md, etc.)
```

**Important:** The `data/` folder with `journal_embeddings.npz` MUST be in the repo. These are pre-generated locally, so the server doesn't need sentence-transformers.

---

## Step 2: Re-generate embeddings with an API provider (one-time)

Your current embeddings were generated with the local model. For deployment, the **runtime query embedding** must use the same provider. You have two options:

### Option A: Keep local embeddings, use OpenAI for queries (easier, slight quality mismatch)

This works but the query embeddings (OpenAI) won't be in the same vector space as your journal embeddings (local model). Results will be less accurate.

### Option B: Re-generate with OpenAI (recommended, best quality)

Run this locally once:

```bash
cd data
python ../05_generate_embeddings.py --provider openai --api-key YOUR_OPENAI_KEY
cd ..
```

This takes ~30 seconds and costs less than $0.01. Now both journal embeddings and runtime queries use the same model.

---

## Step 3: Create a GitHub repository

1. Go to https://github.com/new
2. Name it `journal-recommender` (or whatever you prefer)
3. Set it to **Public** (or Private — Render supports both)
4. Don't initialize with README (you already have files)

In your terminal (in the `journal-recommender` folder):

```bash
git init
git add .
git commit -m "Initial commit — journal recommender"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/journal-recommender.git
git push -u origin main
```

If you don't have Git installed, download it from https://git-scm.com/downloads

---

## Step 4: Deploy on Render

### Option A: Blueprint (automatic from render.yaml)

1. Go to https://dashboard.render.com
2. Click **New** → **Blueprint**
3. Connect your GitHub repo
4. Render detects `render.yaml` and configures everything
5. Set your environment variables when prompted:
   - `GEMINI_API_KEY` = your Gemini key
   - `OPENAI_API_KEY` = your OpenAI key (for embeddings)
6. Click **Apply**

### Option B: Manual setup

1. Go to https://dashboard.render.com
2. Click **New** → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Name:** `journal-recommender`
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements-deploy.txt`
   - **Start command:** `python app.py --data-dir data --llm gemini --embedding-provider openai --host 0.0.0.0 --port $PORT`
   - **Plan:** Free (or Starter for $7/mo)
5. Add environment variables:
   - `GEMINI_API_KEY` = your key
   - `OPENAI_API_KEY` = your key
6. Click **Deploy**

---

## Step 5: Wait for deployment

Render will:
1. Clone your repo
2. Install dependencies (~1 minute)
3. Start the server
4. Give you a URL like `https://journal-recommender-xxxx.onrender.com`

Open that URL → you should see the landing page!

---

## Step 6: Update index.html API_BASE (if needed)

The current `index.html` auto-detects the API URL:

```javascript
const API_BASE = window.location.port === "8000"
  ? window.location.origin
  : "http://localhost:8000";
```

For Render, we need it to use the same origin. Replace that line with:

```javascript
const API_BASE = window.location.origin;
```

This works both locally (when served by the backend) and on Render.

Commit and push:

```bash
git add index.html
git commit -m "Fix API_BASE for deployment"
git push
```

Render auto-redeploys on push.

---

## Connecting your Lovable frontend (optional)

If you want your Lovable app to call the deployed backend:

1. In your Lovable code, set the API URL to your Render URL:
   ```javascript
   const API_BASE = "https://journal-recommender-xxxx.onrender.com";
   ```

2. The CORS middleware in `app.py` already allows all origins (`allow_origins=["*"]`), so cross-origin requests work.

3. **HTTPS note:** Both Lovable and Render use HTTPS, so there's no mixed-content issue.

---

## Troubleshooting

**"No embeddings found — DEGRADED MODE"**
→ The `data/journal_embeddings.npz` file isn't in your repo. Make sure it's not in `.gitignore` and push it.

**Cold start takes 30+ seconds**
→ Normal for Render free tier. The service spins down after 15 minutes of inactivity. Upgrade to Starter ($7/mo) for always-on.

**"Report generation requires python-docx"**
→ Make sure `python-docx` is in `requirements-deploy.txt`.

**Embedding dimension mismatch**
→ Your query embeddings and journal embeddings must use the same provider. Re-generate with `05_generate_embeddings.py --provider openai`.

**RAM limit exceeded**
→ If using the free tier, make sure `sentence-transformers` is NOT in your requirements. Use API embeddings only.

---

## Cost summary

| Component | Free tier | Paid |
|-----------|-----------|------|
| Render hosting | $0 (cold starts) | $7/month (always on) |
| Gemini API (re-ranking) | Free tier (~15 RPM) | $0.15/1M input tokens |
| OpenAI embeddings (queries) | ~$0.02/1M tokens | Same |
| **Total** | **~$0/month** | **~$7/month** |

---

## Updating

To push changes:

```bash
git add .
git commit -m "Description of changes"
git push
```

Render auto-redeploys in ~1-2 minutes.
