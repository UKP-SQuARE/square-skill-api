from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, root_validator, validator


class SaliencyMethod(str, Enum):
    SIMPLE_GRADS = "simple_grads"
    INTEGRATED_GRADS = "integrated_grads"
    SMOOTH_GRADS = "smooth_grads"
    ATTENTION = "attention"
    SCALED_ATTENTION = "scaled_attention"
    BERT_VIZ = "bertviz"


class ExplainKwargsMode(str, Enum):
    ALL = "all"
    QUESTION = "question"
    CONTEXT = "context"


class ExplainKwargs(BaseModel):
    method: SaliencyMethod
    top_k: int = Field(..., ge=0)
    mode: ExplainKwargsMode

    class Config:
        use_enum_values = True


class AttackKwargsMethod(str, Enum):
    HOT_FLIP = "hotflip"
    INPUT_REDUCTION = "input_reduction"
    SUB_SPAN = "sub_span"
    TOPK_TOKENS = "topk_tokens"


class AttackKwargs(BaseModel):
    method: AttackKwargsMethod
    saliency_method: SaliencyMethod
    max_flips: int = Field(
        None, ge=1, description="Maximum number of flips to perform for HotFlip."
    )
    max_reductions: int = Field(
        None,
        ge=1,
        description="Maximum number of reductions to perform for Input Reduction.",
    )
    max_tokens: int = Field(
        None, ge=1, description="Maximum number of top-k to use for TopK."
    )

    class Config:
        use_enum_values = True

    @root_validator()
    def validate_param_pairs(cls, values):
        method2parm = {
            AttackKwargsMethod.HOTFLIP: "max_flips",
            AttackKwargsMethod.INPUT_REDUCTION: "max_reductions",
            AttackKwargsMethod.SUB_SPAN: "max_tokens",
            AttackKwargsMethod.TOPK_TOKENS: "max_tokens",
        }
        for method, param in method2parm.items():
            if values["method"] == method and values[param] is None:
                raise ValueError(f"{method.value} requires {param} to be set.")

        return values

    @root_validator()
    def mutually_exclusive(cls, values):
        mutually_exclusive_attributes = [
            "max_flips",
            "max_reductions",
            "span_length",
            "max_topk",
        ]
        num_not_none = sum(
            [
                0 if v is None else 1
                for k, v in values.items()
                if k in mutually_exclusive_attributes
            ]
        )
        if num_not_none > 1:
            raise ValueError(
                "Only one of the following attributes can be specified: {}".format(
                    mutually_exclusive_attributes
                )
            )
        return values


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
    attack_kwargs: Optional[Dict] = Field(
        {}, description="Optional values for obtaining adversarial outputs."
    )
    model_kwargs: Optional[Dict] = Field(
        {}, description="Optional values for the model forward pass."
    )
    task_kwargs: Optional[Dict] = Field({}, description="Optional values for the task.")
    preprocessing_kwargs: Optional[Dict] = Field(
        {}, description="Optional values for preprocessing."
    )
