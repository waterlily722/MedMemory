from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class MemoryQuery(SerializableMixin):
    """Retrieval query for memory search.

    ``query_text`` is a flat text string that bundles the following signals from
    the current case state for embedding / cosine-similarity matching:

    - Clinical situation overview & problem summary
    - Local goal and diagnostic uncertainty
    - Key positive and negative evidence
    - Critical missing information
    - Finalize risk level
    - Candidate next actions

    Construction:
    - Rule mode (default): concatenates :class:`CaseState` fields into one text.
    - LLM mode: asks the LLM to produce a concise query from CaseState fields.

    See :func:`online.query_builder.build_memory_query` for details.
    """
    case_id: str
    turn_id: int = 0
    query_text: str = ""