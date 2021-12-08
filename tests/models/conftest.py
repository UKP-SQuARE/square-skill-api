import random
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


@fixture
def model_api_sequence_classification_ouput_factory():
    def model_api_sequence_classification_ouput(n: int):
        logits = [random.random() for _ in range(n)]
        max_logit = max(logits)
        argmax = logits.index(max_logit)
        return {
            "labels": [argmax],
            "id2label": {i: str(i) for i in range(n)},
            "model_outputs": {
                "logits": [logits],
            },
            "model_output_is_encoded": False,
        }
    return model_api_sequence_classification_ouput


@fixture
def model_api_question_answering_ouput_factory():
    def model_api_question_answering_ouput(n: int):
        return {
            "answers": [[
                {
                    "score": i / sum(range(n)),
                    "start": 0,
                    "end": 0,
                    "answer": "answer {i}".format(i=str(i)),
                }
                for i in range(n)
            ]],
            "model_outputs": {
                "start_logits": [[random.random() for _ in range(n*10)]],
                "end_logits": [[random.random() for _ in range(n*10)]],
            },
            "model_output_is_encoded": False,
        }
    return model_api_question_answering_ouput
