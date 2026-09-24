"""Centralized defaults and environment-driven configuration for the presentation pipeline."""

from __future__ import annotations

import os
from pathlib import Path


# Repository root used to build absolute default paths.
ROOT = Path(__file__).resolve().parent
# Default RAG chunk source and output directory.
DEFAULT_CHUNKS_PATH = ROOT / "data" / "book_chunks.jsonl"
DEFAULT_OUT_DIR = ROOT / "outputs" / "llm_slides_voiceover"

# Text generation model/provider defaults.
DEFAULT_MODEL_ID = (
    os.environ.get("MODEL_ID")
    or os.environ.get("PLANNER_MODEL_ID")
    or "google/gemma-3-1b-it"
)
DEFAULT_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "huggingface").strip().lower() or "huggingface"
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("PLANNER_MAX_NEW_TOKENS", "1200"))
# Sampling temperature for repeated-run diversity experiments.
DEFAULT_TEMPERATURE = float(os.environ.get("PLANNER_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.environ.get("PLANNER_TOP_P", "0.95"))

# Image retrieval/generation defaults.
DEFAULT_IMAGE_PROVIDER = os.environ.get("WEB_IMAGES", "auto")
DEFAULT_IMAGES_PER_SLIDE = int(os.environ.get("WEB_IMAGES_PER_SLIDE", "5"))
DEFAULT_IMAGE_FALLBACK = os.environ.get("IMAGE_FALLBACK", "require_web")

# Retrieval mode defaults (semantic is the main validated mode).
DEFAULT_RAG_MODE = os.environ.get("RAG_MODE", "semantic")
DEFAULT_EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
DEFAULT_RAG_CONTEXT_MAX_CHARS = int(os.environ.get("RAG_CONTEXT_MAX_CHARS", "1800"))

# Recording/voiceover defaults.
DEFAULT_RECORDING_MODE = os.environ.get("RECORDING_MODE", "video")
DEFAULT_VOICEOVER_ENGINE = os.environ.get("VOICEOVER_ENGINE", "edge-tts")
DEFAULT_VOICE = os.environ.get("VOICE", "en-US-JennyNeural")
DEFAULT_VOICEOVER_PROFILE = os.environ.get("VOICEOVER_PROFILE", "short").strip().lower() or "short"
DEFAULT_TARGET_TOTAL_SECONDS = int(os.environ.get("TARGET_TOTAL_SECONDS", "180"))
DEFAULT_VIDEO_WIDTH = int(os.environ.get("VIDEO_WIDTH", "1920"))
DEFAULT_VIDEO_HEIGHT = int(os.environ.get("VIDEO_HEIGHT", "1080"))
DEFAULT_VIDEO_FPS = int(os.environ.get("VIDEO_FPS", "30"))
DEFAULT_MIN_SLIDE_SECONDS = float(os.environ.get("MIN_SLIDE_SECONDS", "4.0"))
DEFAULT_WORDS_PER_SECOND = float(os.environ.get("WORDS_PER_SECOND", "2.4"))

# Slide content density.
DEFAULT_CONTENT_MIN_BULLETS = int(os.environ.get("CONTENT_MIN_BULLETS", "2"))
DEFAULT_CONTENT_MAX_BULLETS = int(os.environ.get("CONTENT_MAX_BULLETS", "3"))
DEFAULT_NONCONTENT_MIN_BULLETS = int(os.environ.get("NONCONTENT_MIN_BULLETS", "2"))
DEFAULT_NONCONTENT_MAX_BULLETS = int(os.environ.get("NONCONTENT_MAX_BULLETS", "3"))
