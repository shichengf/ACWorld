"""Large real-catalog stress suite for ACWorld.

The package is deliberately separate from the original 200-task benchmark.
It shares ACWorld's VCP envelope and authority boundary while using a
streaming SQLite catalog backend suitable for hundreds of thousands of rows.
"""

from large_catalog.database import CatalogDatabase, CatalogIngestReport
from large_catalog.models import LargeCatalogTask, TaskResult

__all__ = [
    "CatalogDatabase",
    "CatalogIngestReport",
    "LargeCatalogTask",
    "TaskResult",
]
