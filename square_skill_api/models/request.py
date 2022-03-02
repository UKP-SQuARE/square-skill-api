from typing import Dict, Any

from pydantic import BaseModel, Field, PositiveInt


class QueryRequest(BaseModel):
    """The model for a query request that the skill receives."""

    query: str = Field(
        ..., description="The input to the model that is entered by the user"
    )
    skill_args: Dict[str, Any] = Field(
        {}, description="Optional values for specific parameters of the skill"
    )
    skill: Dict[str, Any] = Field({}, description="Skill information. See Skill-Manager for details.")
    user_id: str = Field("")
