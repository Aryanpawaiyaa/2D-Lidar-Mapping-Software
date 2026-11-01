"""Relocalization and multi-room map merging package."""

from relocalization.correlative_matcher import CorrelativeGlobalMatcher, RelocalizationResult
from relocalization.session_merger import MultiRoomSessionManager, MappingStatus, OverlapCandidate

__all__ = [
    "CorrelativeGlobalMatcher",
    "RelocalizationResult",
    "MultiRoomSessionManager",
    "MappingStatus",
    "OverlapCandidate",
]

