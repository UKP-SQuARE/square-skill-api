from typing import List
from pytest import fixture

from square_skill_api.models.prediction import (
    Prediction,
    PredictionDocument,
    PredictionOutput,
)


@fixture
def prediction_document_factory():
    def generate_prediction_document(
        index="",
        document_id="",
        document="lorem ipsum",
        span: List = [],
        url="",
        source="",
    ):
        return PredictionDocument(
            index=index,
            document_id=document_id,
            document=document,
            span=span,
            url=url,
            source=source,
        )

    return generate_prediction_document


@fixture
def prediction_output_factory():
    def generate_prediction_output(output="", output_score=0):
        return PredictionOutput(output=output, output_score=output_score)

    return generate_prediction_output


@fixture
def prediction_factory():
    def generate_prediction(
        prediction_score,
        prediction_output: PredictionOutput,
        prediction_documents: List[PredictionDocument] = [],
    ):
        return Prediction(
            prediction_score=prediction_score,
            prediction_output=prediction_output,
            prediction_documents=prediction_documents,
        )

    return generate_prediction
