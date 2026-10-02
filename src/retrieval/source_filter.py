"""Restricts retrieval to the knowledge sources selected in the chat UI."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class SourceFilter:
    """
    Registered sources are matched by source_id. Legacy chunks (uploaded
    before the source registry existed) have no source_id, so they are
    matched by their source_file instead.
    """
    source_ids: List[str] = field(default_factory=list)
    legacy_files: List[str] = field(default_factory=list)

    @classmethod
    def from_request(cls, source_ids: Optional[List[str]],
                     legacy_files: Optional[List[str]] = None) -> Optional["SourceFilter"]:
        """None for both fields means 'search everything' (older clients send neither)."""
        if source_ids is None and legacy_files is None:
            return None
        return cls(list(dict.fromkeys(source_ids or [])),
                   list(dict.fromkeys(legacy_files or [])))

    @property
    def is_empty(self) -> bool:
        return not self.source_ids and not self.legacy_files

    def matches(self, metadata: dict) -> bool:
        metadata = metadata or {}
        source_id = metadata.get("source_id")
        if source_id:
            return source_id in self.source_ids
        return metadata.get("source_file") in self.legacy_files

    def chroma_where(self) -> dict:
        clauses = []
        if self.source_ids:
            clauses.append({"source_id": {"$in": self.source_ids}})
        if self.legacy_files:
            clauses.append({"source_file": {"$in": self.legacy_files}})
        return clauses[0] if len(clauses) == 1 else {"$or": clauses}
