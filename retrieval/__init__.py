"""Shared retrieval primitives for Yuki's local and web RAG paths.

This package retrieves and formats evidence only.  It never calls the chat
model, decides whether a tool is needed, or executes an action described by
retrieved content.
"""

from .core import RetrievalHit, reciprocal_rank_fusion, render_retrieval_context

__all__ = [
    "RetrievalHit",
    "reciprocal_rank_fusion",
    "render_retrieval_context",
]
