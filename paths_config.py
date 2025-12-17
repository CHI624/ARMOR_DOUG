"""
Central directory configuration for the ARMOR project.
Each entry lists where the path is used and what it should contain, so new
code can import these instead of hard-coding directory strings.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Directory map with lightweight documentation for quick reference.
DIRECTORIES = {
    # Project root and shared top-level assets
    "root": {
        "path": BASE_DIR,
        "used_in": ["app.py (Flask root)"],
        "contains": "Repository root for ARMOR services and assets",
    },
    "templates": {
        "path": BASE_DIR / "templates",
        "used_in": ["app.py (Flask templates)"],
        "contains": "Jinja templates for the main ARMOR Flask app",
    },
    "static_root": {
        "path": BASE_DIR / "static",
        "used_in": ["app.py (Flask static files)"],
        "contains": "Static assets (images, CSS/JS, tables, media)",
    },
    "uploads": {
        "path": BASE_DIR / "static" / "uploads",
        "used_in": ["app.py (file uploads, quer images, audio)"],
        "contains": "User-uploaded images/audio plus generated results",
    },
    "analysis": {
        "path": BASE_DIR / "static" / "analysis",
        "used_in": ["app.py (placeholder for analysis artifacts)"],
        "contains": "Generated analysis artifacts for the main app",
    },
    "static_text": {
        "path": BASE_DIR / "static" / "text",
        "used_in": ["app.py (severity_result.txt output location)"],
        "contains": "Text outputs exposed via the Flask static handler",
    },
    "temp": {
        "path": BASE_DIR / "temp",
        "used_in": ["app.py (hazard temp files)"],
        "contains": "Short-lived text/image working files",
    },
    "severity_outputs": {
        "path": BASE_DIR / "severity_outputs",
        "used_in": ["app.py (ensemble severity output directory)"],
        "contains": "Captured severity outputs from ensemble runs",
    },
    "css": {
        "path": BASE_DIR / "css",
        "used_in": ["templates referencing static CSS bundle"],
        "contains": "Shared CSS (e.g., bootstrap) served as static files",
    },
    "db_file": {
        "path": BASE_DIR / "armor.db",
        "used_in": ["app.py (armor_records, armor_scores, quer tables)"],
        "contains": "Primary SQLite database for ARMOR submissions",
    },
    # Model/data assets shared across inference scripts
    "model_dir": {
        "path": BASE_DIR / "model",
        "used_in": [
            "dmd_inference_ensemble.py (ensemble model weights/tokenizer)",
            "demo_crisis_mmd/app*.py (keras model checkpoints)",
            "demo_crisis_mmd/dmd_inference.py",
        ],
        "contains": "Model weights and tokenizer artifacts",
    },
    "data_dump": {
        "path": BASE_DIR / "data_dump",
        "used_in": ["demo_crisis_mmd/app*.py (pre-baked npy tensors)"],
        "contains": "Precomputed numpy dumps for datasets",
    },
    "metadata": {
        "path": BASE_DIR / "metadata",
        "used_in": ["demo_crisis_mmd/app*.py (dataset TSV metadata)"],
        "contains": "Metadata TSV/CSV files describing datasets",
    },
    "performance_measures": {
        "path": BASE_DIR / "performance_measures",
        "used_in": ["demo_crisis_mmd/app*.py (performance CSVs)"],
        "contains": "Task performance metric CSVs (informative/humanitarian/severity)",
    },
    "dmd_root": {
        "path": BASE_DIR / "dmd",
        "used_in": [
            "dmd_inference_ensemble.py (multimodal inputs)",
            "demo_crisis_mmd/dmd_inference.py",
            "demo_crisis_mmd/dmd_metadata.py",
        ],
        "contains": "DMD dataset (multimodal text/image directories and metadata)",
    },
    "dmd_multimodal": {
        "path": BASE_DIR / "dmd" / "multimodal",
        "used_in": [
            "dmd_inference_ensemble.py (default text/image samples)",
            "demo_crisis_mmd/dmd_inference.py",
            "demo_crisis_mmd/dmd_metadata.py",
        ],
        "contains": "Task-specific DMD multimodal text/images per category",
    },
    "crisis_mmd_root": {
        "path": BASE_DIR / "crisis-mmd",
        "used_in": ["demo_crisis_mmd/app*.py (stop words, data references)"],
        "contains": "Crisis-MMD resources (stop words, docs)",
    },
    # Demo app specific folders
    "demo_root": {
        "path": BASE_DIR / "demo_crisis_mmd",
        "used_in": ["demo_crisis_mmd/app*.py (Flask demo app root)"],
        "contains": "Crisis-MMD demo app source and assets",
    },
    "demo_static": {
        "path": BASE_DIR / "demo_crisis_mmd" / "static",
        "used_in": [
            "demo_crisis_mmd/app*.py (matplotlib saves)",
            "demo_crisis_mmd/utils.py",
        ],
        "contains": "Static assets for the demo Flask app",
    },
    "demo_templates": {
        "path": BASE_DIR / "demo_crisis_mmd" / "templates",
        "used_in": ["demo_crisis_mmd/app*.py (Flask templates)"],
        "contains": "Jinja templates for the demo Flask app",
    },
    "demo_performance": {
        "path": BASE_DIR / "demo_crisis_mmd" / "performance_measures",
        "used_in": ["demo_crisis_mmd/app*.py (performance CSVs)"],
        "contains": "Performance metric CSVs scoped to the demo app",
    },
    "demo_metadata": {
        "path": BASE_DIR / "demo_crisis_mmd" / "metadata",
        "used_in": ["demo_crisis_mmd/app*.py (TSV metadata)"],
        "contains": "Metadata files bundled with the demo app",
    },
    "demo_stop_words": {
        "path": BASE_DIR / "demo_crisis_mmd" / "stop_words",
        "used_in": ["demo_crisis_mmd/app*.py (tokenizer helpers)"],
        "contains": "Stop-word lists for the demo tokenizers",
    },
    "demo_data_dump": {
        "path": BASE_DIR / "demo_crisis_mmd" / "data_dump",
        "used_in": ["demo_crisis_mmd/app*.py (numpy dumps if present)"],
        "contains": "Demo-scoped numpy dumps (images/text vectors)",
    },
}


class Paths:
    """Simple accessors for configured directories."""

    BASE = DIRECTORIES["root"]["path"]
    TEMPLATES = DIRECTORIES["templates"]["path"]
    STATIC_ROOT = DIRECTORIES["static_root"]["path"]
    UPLOADS = DIRECTORIES["uploads"]["path"]
    ANALYSIS = DIRECTORIES["analysis"]["path"]
    STATIC_TEXT = DIRECTORIES["static_text"]["path"]
    TEMP = DIRECTORIES["temp"]["path"]
    SEVERITY_OUTPUTS = DIRECTORIES["severity_outputs"]["path"]
    CSS = DIRECTORIES["css"]["path"]
    DB_FILE = DIRECTORIES["db_file"]["path"]

    MODEL_DIR = DIRECTORIES["model_dir"]["path"]
    DATA_DUMP = DIRECTORIES["data_dump"]["path"]
    METADATA = DIRECTORIES["metadata"]["path"]
    PERFORMANCE_MEASURES = DIRECTORIES["performance_measures"]["path"]
    DMD_ROOT = DIRECTORIES["dmd_root"]["path"]
    DMD_MULTIMODAL = DIRECTORIES["dmd_multimodal"]["path"]
    CRISIS_MMD_ROOT = DIRECTORIES["crisis_mmd_root"]["path"]

    DEMO_ROOT = DIRECTORIES["demo_root"]["path"]
    DEMO_STATIC = DIRECTORIES["demo_static"]["path"]
    DEMO_TEMPLATES = DIRECTORIES["demo_templates"]["path"]
    DEMO_PERFORMANCE = DIRECTORIES["demo_performance"]["path"]
    DEMO_METADATA = DIRECTORIES["demo_metadata"]["path"]
    DEMO_STOP_WORDS = DIRECTORIES["demo_stop_words"]["path"]
    DEMO_DATA_DUMP = DIRECTORIES["demo_data_dump"]["path"]


def ensure_directories(keys=None) -> None:
    """
    Create the directories that should always exist for runtime use.
    """
    default_keys = (
        "uploads",
        "analysis",
        "static_text",
        "temp",
        "severity_outputs",
        "demo_static",
        "demo_templates",
    )
    target_keys = keys or default_keys
    for name in target_keys:
        DIRECTORIES[name]["path"].mkdir(parents=True, exist_ok=True)


def get_path(name: str) -> Path:
    """Helper to fetch a configured path by name."""
    return DIRECTORIES[name]["path"]

