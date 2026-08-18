"""All project settings in one easy-to-read file."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Input and output paths
PDF_PATH = ROOT / "data" / "raw" / "nice_ng151_colorectal_cancer.pdf"
MARKDOWN_PATH = ROOT / "data" / "processed" / "guideline.md"
PAGES_PATH = ROOT / "data" / "processed" / "pages.jsonl"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"
CHROMA_PATH = ROOT / "data" / "vector_store" / "chroma"
TEST_QUESTIONS_PATH = ROOT / "data" / "evaluation" / "test_questions.csv"
EVALUATION_RESULTS_PATH = ROOT / "data" / "evaluation" / "evaluation_results.csv"
EVALUATION_SUMMARY_PATH = ROOT / "data" / "evaluation" / "evaluation_summary.csv"

# Guideline information used in citations
DOCUMENT_NAME = "Colorectal cancer"
GUIDELINE_CODE = "NICE NG151"
SOURCE_URL = "https://www.nice.org.uk/guidance/ng151"


# Structure-aware Recursive Character Chunking with Overlap
# document-based chunking with recursive splitting and overlap.

# Chunking settings measured with the multilingual-e5-base tokenizer.
# Content-token budget. 450 leaves room for the retrieval metadata and stays
# within multilingual-e5-base's 512-token model limit.
CHUNK_SIZE = 450
CHUNK_OVERLAP = 80

# Useful guideline content only:
# overview/general guidance (5-6), recommendations and glossary (7-28),
# and rationale/impact (30-48). Cover, contents, research questions,
# committee/update pages and the ISBN page are excluded.
PAGES_TO_INDEX = list(range(5, 29)) + list(range(30, 49))

# Keep the tested multilingual Sentence Transformer model.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_BATCH_SIZE = 16
# The model is already downloaded. Avoid slow Hugging Face network checks at runtime.
EMBEDDING_LOCAL_FILES_ONLY = True

# Vector database and search settings
COLLECTION_NAME = "nice_ng151_colorectal"
TOP_K = 5

# Day 3 grounded generation settings (Groq free cloud API)
GENERATION_MODEL = "qwen/qwen3.6-27b"
GENERATION_MAX_OUTPUT_TOKENS = 1000
MIN_RETRIEVAL_SCORE = 0.60
