"""All project settings in one easy-to-read file."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Input and output paths
PDF_PATH = ROOT / "data" / "raw" / "nice_ng151_colorectal_cancer.pdf"
MARKDOWN_PATH = ROOT / "data" / "processed" / "guideline.md"
PAGES_PATH = ROOT / "data" / "processed" / "pages.jsonl"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"
CHROMA_PATH = ROOT / "data" / "vector_store" / "chroma"
TEST_QUERIES_PATH = ROOT / "data" / "evaluation" / "test_queries.json"

# Guideline information used in citations
DOCUMENT_NAME = "Colorectal cancer"
GUIDELINE_CODE = "NICE NG151"
SOURCE_URL = "https://www.nice.org.uk/guidance/ng151"

# Chunking settings. The starter kit also uses an approximate 4 chars/token.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# MVP scope: recommendations 1.1, 1.2, 1.3 and 1.6 only.
# This excludes advanced/metastatic treatment and the later rationale pages.
PAGES_TO_INDEX = list(range(7, 18)) + [26, 27]

# Keep the tested multilingual Sentence Transformer model.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_BATCH_SIZE = 16

# Vector database and search settings
COLLECTION_NAME = "nice_ng151_colorectal"
TOP_K = 5
