import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "src" / "data"
# Ensure your PDF filename is exactly this:
PDF_PATH = DATA_DIR / "retrieval-augmented-generation-options.pdf"
VECTOR_INDEX_DIR = BASE_DIR / "faiss_index"

# --- OpenAI Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"

# --- RAG Config ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200