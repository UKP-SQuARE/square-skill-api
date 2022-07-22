from typing import Dict, Any, Optional

from pydantic import BaseModel, Field


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
