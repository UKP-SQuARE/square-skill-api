from enum import Enum
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field


class ExplainKwargsMethod(str, Enum):
    SIMPLE_GRADS = "simple_grads"
    INTEGRATED_GRADS = "integrated_grads"
    SMOOTH_GRADS = "smooth_grads"
    ATTENTION = "attention"
    SCALED_ATTENTION = "scaled_attention"


class ExplainKwargsMode(str, Enum):
    ALL = "all"
    QUESTION = "question"
    CONTEXT = "context"


class ExplainKwargs(BaseModel):
    method: ExplainKwargsMethod
    top_k: int = Field(..., ge=0)
    mode: ExplainKwargsMode

    class Config:
        use_enum_values = True


class QueryRequest(BaseModel):
    """The model for a query request that the skill receives."""

    query: str = Field(
        ..., description="The input to the model that is entered by the user"
    )
    skill_args: Dict[str, Any] = Field(
        {}, description="Optional values for specific parameters of the skill"
    )
    skill: Dict[str, Any] = Field(
        {}, description="Skill information. See Skill-Manager for details."
    )
    user_id: str = Field("")
    explain_kwargs: Optional[Dict] = Field(
        {}, description="Optional values for obtaining explainability outputs."
    )
