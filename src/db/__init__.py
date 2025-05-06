"""
📌 Purpose – Expose the Qdrant database integration API for the Video Timeline Analyzer backend.
🔄 Latest Changes – Initial creation. Imports QdrantClientWrapper for external use.
⚙️ Key Logic – Makes QdrantClientWrapper available as the main entry point for db operations.
📂 Expected File Path – src/db/__init__.py
🧠 Reasoning – Ensures modular, discoverable, and maintainable database integration.
"""

from .qdrant_client import QdrantClientWrapper 