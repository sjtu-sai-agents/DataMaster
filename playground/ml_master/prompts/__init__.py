"""ML-Master Prompts Module

This module contains all prompts used by ML-Master agents.
The prompts are kept in sync with the original ML-Master implementation.
"""

from .draft import get_prompt as get_draft_prompt
from .improve import get_prompt as get_improve_prompt
from .debug import get_prompt as get_debug_prompt
from .review import get_prompt as get_review_prompt
from .review import get_evaluation_instructions

__all__ = [
    "get_draft_prompt",
    "get_improve_prompt",
    "get_debug_prompt",
    "get_review_prompt",
    "get_evaluation_instructions",
]
