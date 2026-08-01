from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini model
LLM_MODEL = "gemini-2.5-flash"

# Chunk settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Vector database path
VECTOR_DB_PATH = "../data/faiss_index"
