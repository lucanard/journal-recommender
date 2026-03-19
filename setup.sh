#!/usr/bin/env bash
set -e

# ═══════════════════════════════════════════════════════════════
#  Journal Recommender — Quick Setup Script
#  Run this once to set everything up, then use start.sh to run
# ═══════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  AI Journal Recommender — Setup               ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# ─── Check Python ────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found."
    echo "   Install from: https://www.python.org/downloads/"
    exit 1
fi

PYVER=$(python3 --version 2>&1)
echo "✓ Found $PYVER"

# ─── Create project structure ────────────────────────────────
echo ""
echo "Setting up project structure..."

mkdir -p data
mkdir -p logs

# Move data files if they're in the current directory
[ -f journal_database.json ] && mv journal_database.json data/ 2>/dev/null || true
[ -f journal_database_final.json ] && mv journal_database_final.json data/ 2>/dev/null || true
[ -f journal_embeddings_input.jsonl ] && mv journal_embeddings_input.jsonl data/ 2>/dev/null || true
[ -f journal_embeddings.npz ] && mv journal_embeddings.npz data/ 2>/dev/null || true
[ -f embedding_metadata.json ] && mv embedding_metadata.json data/ 2>/dev/null || true

echo "✓ Project structure ready"

# ─── Install Python dependencies ─────────────────────────────
echo ""
echo "Installing Python dependencies..."
pip install fastapi uvicorn pydantic numpy openpyxl 2>/dev/null || \
pip3 install fastapi uvicorn pydantic numpy openpyxl 2>/dev/null || \
{
    echo ""
    echo "⚠ Pip install failed. Try manually:"
    echo "  pip install fastapi uvicorn pydantic numpy openpyxl"
    echo ""
}

echo ""
echo "Install sentence-transformers for local embeddings? (free, ~500MB download)"
echo "  This lets you generate embeddings without any API key."
echo "  Skip this if you plan to use OpenAI/Cohere/Voyage instead."
echo ""
read -p "Install sentence-transformers? [Y/n] " INSTALL_ST
if [[ ! "$INSTALL_ST" =~ ^[Nn]$ ]]; then
    echo "Installing sentence-transformers (this may take a minute)..."
    pip install sentence-transformers 2>/dev/null || \
    pip3 install sentence-transformers 2>/dev/null || \
    echo "⚠ Install failed. You can try later: pip install sentence-transformers"
fi

echo ""
echo "✓ Dependencies installed"

# ─── Check for data ──────────────────────────────────────────
echo ""
echo "═══ Data Status ═══"

if [ -f data/journal_database_final.json ]; then
    COUNT=$(python3 -c "import json; print(len(json.load(open('data/journal_database_final.json'))))")
    echo "✓ Enriched database found: $COUNT journals"
    DATA_FILE="data/journal_database_final.json"
elif [ -f data/journal_database.json ]; then
    COUNT=$(python3 -c "import json; print(len(json.load(open('data/journal_database.json'))))")
    echo "✓ Base database found: $COUNT journals (not yet enriched)"
    echo "  Run enrichment later for better results: python enrichment/run_enrichment.py"
    DATA_FILE="data/journal_database.json"
else
    echo "❌ No journal database found!"
    echo "   Place journal_database.json in the data/ directory"
    exit 1
fi

if [ -f data/journal_embeddings.npz ]; then
    echo "✓ Embeddings found — vector search ready"
    HAS_EMBEDDINGS=true
else
    echo "⚠ No embeddings yet — will need to generate them"
    HAS_EMBEDDINGS=false
fi

# ─── Generate embeddings if needed ───────────────────────────
if [ "$HAS_EMBEDDINGS" = false ]; then
    echo ""
    echo "═══ Embedding Generation ═══"
    echo ""
    echo "You need embeddings for semantic search. Options:"
    echo "  1) Local (free, ~5 min on CPU, no API key)"
    echo "  2) OpenAI (fast, ~\$0.02, needs API key)"
    echo "  3) Skip (start without vector search — LLM-only mode)"
    echo ""
    read -p "Choose [1/2/3]: " EMB_CHOICE

    case $EMB_CHOICE in
        1)
            echo "Generating local embeddings..."
            cd data
            python3 ../05_generate_embeddings.py --provider local
            cd ..
            echo "✓ Embeddings generated"
            ;;
        2)
            read -p "Enter OpenAI API key: " OPENAI_KEY
            cd data
            python3 ../05_generate_embeddings.py --provider openai --api-key "$OPENAI_KEY"
            cd ..
            echo "✓ Embeddings generated"
            ;;
        3)
            echo "Skipping embeddings. The API will start in LLM-only mode."
            ;;
    esac
fi

# ─── Configure LLM ──────────────────────────────────────────
echo ""
echo "═══ LLM Configuration ═══"
echo ""
echo "An LLM provides high-quality explanations for each recommendation."
echo "Without it, you get basic similarity-based ranking (still works, less detail)."
echo ""
echo "  1) Anthropic Claude (recommended — best explanations)"
echo "  2) OpenAI GPT-4o-mini (cheaper alternative)"
echo "  3) None (heuristic scoring only)"
echo ""
read -p "Choose [1/2/3]: " LLM_CHOICE

LLM_FLAG=""
case $LLM_CHOICE in
    1)
        read -p "Enter Anthropic API key: " ANTHRO_KEY
        echo "ANTHROPIC_API_KEY=$ANTHRO_KEY" > .env
        LLM_FLAG="--llm anthropic"
        echo "✓ Anthropic Claude configured"
        ;;
    2)
        read -p "Enter OpenAI API key: " OAI_KEY
        echo "OPENAI_API_KEY=$OAI_KEY" > .env
        LLM_FLAG="--llm openai"
        echo "✓ OpenAI configured"
        ;;
    3)
        echo "✓ Running without LLM (heuristic mode)"
        ;;
esac

# ─── Create start script ─────────────────────────────────────
cat > start.sh << STARTEOF
#!/usr/bin/env bash
# Load environment variables if .env exists
if [ -f .env ]; then
    export \$(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "Starting Journal Recommender API..."
echo "  API:  http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"
echo ""

python3 app.py --data-dir data $LLM_FLAG "\$@"
STARTEOF
chmod +x start.sh

# ─── Done ─────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  ✅ Setup complete!                           ║"
echo "╠═══════════════════════════════════════════════╣"
echo "║                                               ║"
echo "║  To start the API:                            ║"
echo "║    ./start.sh                                 ║"
echo "║                                               ║"
echo "║  Then open:                                   ║"
echo "║    http://localhost:8000/docs                  ║"
echo "║                                               ║"
echo "║  To enrich your database (optional):          ║"
echo "║    cd enrichment && python run_enrichment.py   ║"
echo "║                                               ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""
