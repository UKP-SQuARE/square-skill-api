import random
from typing import List, Union

import numpy as np
from pytest import fixture

from square_skill_api.models.prediction import (
    Prediction,
    PredictionDocument,
    PredictionOutput,
)


@fixture(scope="module")
def monkeymodule():
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@fixture
def predictions_factory():
    def generate_predictions(
        answer_scores: List[float],
        document_scores: List[float],
        answers: List[str] = None,
        question: str = "test question",
    ):
        predictions = []
        for document_i, document_score in enumerate(document_scores):
            for answer_i, answer_score in enumerate(answer_scores):
                predictions.append(
                    Prediction(
                        question=question,
                        prediction_score=answer_score,
                        prediction_output=PredictionOutput(
                            output=answers[answer_i]
                            if answers
                            else f"answer {answer_i}",
                            output_score=answer_score,
                        ),
                        prediction_documents=[
                            PredictionDocument(
                                document=f"document {document_i}",
                                document_score=document_score,
                            )
                            if document_score is not None
                            else PredictionDocument(
                                document="document {document_i}",
                            )
                        ],
                    )
                )
        return predictions

    return generate_predictions


@fixture
def model_api_sequence_classification_ouput_factory():
    def model_api_sequence_classification_ouput(
        n: int, logits: Union[None, List] = None, questions: Union[None, List] = None
    ):
        logits = [random.random() for _ in range(n)] if not logits else logits
        max_logit = max(logits)
        argmax = logits.index(max_logit)
        model_api_output = {
            "labels": [argmax],
            "id2label": {i: str(i) for i in range(n)},
            "model_outputs": {
                "logits": [logits],
            },
            "model_output_is_encoded": False,
        }
        if questions:
            model_api_output["questions"] = questions
        return model_api_output

    return model_api_sequence_classification_ouput


@fixture
def model_api_sequence_classification_with_graph_ouput_factory():
    def model_api_sequence_classification_with_graph_ouput(
        n: int, logits: Union[None, List] = None
    ):
        logits = [random.random() for _ in range(n)] if not logits else logits
        max_logit = max(logits)
        argmax = logits.index(max_logit)
        return {
            "labels": [argmax],
            "id2label": {i: str(i) for i in range(n)},
            "model_outputs": {
                "logits": [logits],
            },
            "model_output_is_encoded": False,
            "lm_subgraph": {
                "nodes": {
                    "23505": {
                        "id": 23505,
                        "name": "google",
                        "q_node": True,
                        "ans_node": False,
                        "weight": -0.05396396666765213,
                    },
                    "184904": {
                        "id": 184904,
                        "name": "resection",
                        "q_node": False,
                        "ans_node": False,
                        "weight": -0.06964521110057831,
                    },
                },
                "edges": {
                    "0": {
                        "source": 23505,
                        "target": 23505,
                        "weight": 0.5,
                        "label": "isa",
                    },
                    "1": {
                        "source": 23505,
                        "target": 184904,
                        "weight": 1.0,
                        "label": "atlocation",
                    },
                },
            },
            "attn_subgraph": {
                "nodes": {
                    "23505": {
                        "id": 23505,
                        "name": "google",
                        "q_node": True,
                        "ans_node": False,
                        "weight": 0.2670625150203705,
                    },
                    "132569": {
                        "id": 132569,
                        "name": "gps",
                        "q_node": True,
                        "ans_node": False,
                        "weight": 0.047409314662218094,
                    },
                },
                "edges": {
                    "0": {
                        "source": 3296,
                        "target": 3296,
                        "weight": 1.0,
                        "label": "atlocation",
                    },
                    "1": {
                        "source": 3296,
                        "target": 2210,
                        "weight": 2.0,
                        "label": "isa",
                    },
                },
            },
        }

    return model_api_sequence_classification_with_graph_ouput


@fixture
def model_api_question_answering_output_factory():
    def model_api_question_answering_ouput(
        n_docs: int, n_answers: int, answer: str = None
    ):
        return {
            "answers": [
                [
                    {
                        "score": answer_i / sum(range(n_answers)),
                        "start": 0,
                        "end": 0,
                        "answer": "answer {answer_i} for doc {doc_i}".format(
                            answer_i=str(answer_i), doc_i=str(doc_i)
                        )
                        if answer is None
                        else answer,
                    }
                    for answer_i in range(n_answers)
                ]
                for doc_i in range(n_docs)
            ],
            "model_outputs": {
                "start_logits": "something encoded",
                "end_logits": "something encoded",
            },
            "model_output_is_encoded": True,
        }

    return model_api_question_answering_ouput


@fixture
def model_api_sequence_classification_with_bertviz_ouput_factory():
    def model_api_question_answering_ouput(
        n_docs: int, n_answers: int, answer: str = None
    ):
        return {
            "answers": [
                [
                    {
                        "score": 1,
                        "start": 0,
                        "end": 0,
                        "answer": "answer {answer_i} for doc {doc_i}".format(
                            answer_i=str(answer_i), doc_i=str(doc_i)
                        )
                        if answer is None
                        else answer,
                    }
                    for answer_i in range(n_answers)
                ]
                for doc_i in range(n_docs)
            ],
            "model_outputs": {
                "start_logits": "something encoded",
                "end_logits": "something encoded",
            },
            "model_output_is_encoded": True,
            "bertviz": ["<html>foo</html>"],
        }

    return model_api_question_answering_ouput


@fixture
def model_api_attribution_output_factory():
    def attribution_factory():
        return [
            {
                "topk_question_idx": [[0]],
                "topk_context_idx": [[0]],
                "question_tokens": [[(1, "hello", 0.1)]],
                "context_tokens": [[(1, "world", 0.2)]],
            }
        ]

    return attribution_factory


@fixture
def model_api_generation_output_factory():
    def model_api_generation_output_factory(
        batch_size: int = 1, num_sequences: int = 2, sequence_length: int = 5
    ):
        return {
            "model_outputs": {
                "sequences": [
                    np.random.randint(0, 100, (num_sequences, sequence_length))
                    for _ in range(batch_size)
                ],
                "sequences_scores": [
                    np.random.random(size=(num_sequences,)) for _ in range(batch_size)
                ],
                "scores": [],
                "beam_indices": [
                    np.random.randint(0, 100, (num_sequences, sequence_length))
                    for _ in range(batch_size)
                ],
                "model_output_is_encoded": True,
                "generated_texts": [
                    ["hello world"] * num_sequences for _ in range(batch_size)
                ],
            }
        }

    return model_api_generation_output_factory
