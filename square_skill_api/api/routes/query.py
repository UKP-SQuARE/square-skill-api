import logging

from fastapi import APIRouter, Depends

from square_skill_api.models import QueryOutput, QueryRequest

logger = logging.getLogger(__name__)

router = APIRouter()


def predict():
    """Placeholder function, to be overwritten by the Skill implementation."""

    def predict_fn():
        raise NotImplementedError("Predict function needs to be overwritten.")

    return predict_fn


@router.post(
    "/query",
    response_model=QueryOutput,
    name="Skill Query",
)
def query(query: QueryRequest, predict_fn=Depends(predict)) -> QueryOutput:
    """Query a skill by providing an input (e.g. question and optional context) and
    receiving a prediction (e.g. an answer to a question)."""
    logger.info(f"Query: {query.dict()}")
    prediction = predict_fn(query)

    return prediction
