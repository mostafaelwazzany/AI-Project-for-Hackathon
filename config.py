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
TEST_QUESTIONS_PATH = ROOT / "data" / "evaluation" / "test_questions.csv"
EVALUATION_RESULTS_PATH = ROOT / "data" / "evaluation" / "evaluation_results.csv"

# Guideline information used in citations
DOCUMENT_NAME = "Colorectal cancer"
GUIDELINE_CODE = "NICE NG151"
SOURCE_URL = "https://www.nice.org.uk/guidance/ng151"

# Chunking settings. The starter kit also uses an approximate 4 chars/token.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Useful guideline content only:
# overview/general guidance (5-6), recommendations and glossary (7-28),
# and rationale/impact (30-48). Cover, contents, research questions,
# committee/update pages and the ISBN page are excluded.
PAGES_TO_INDEX = list(range(5, 29)) + list(range(30, 49))

# Keep the tested multilingual Sentence Transformer model.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_BATCH_SIZE = 16

# Vector database and search settings
COLLECTION_NAME = "nice_ng151_colorectal"
TOP_K = 5
