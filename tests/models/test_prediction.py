import itertools

import pytest

from square_skill_api.models.prediction import PredictionDocument, QueryOutput


@pytest.mark.parametrize("scores", list(itertools.permutations([1, 2, 3])))
def test_query_output_autosort(scores, prediction_factory, prediction_output_factory):
    max_score = max(scores)

    predictions = [
        prediction_factory(
            prediction_score=score,
            prediction_output=prediction_output_factory(output_score=score),
        )
        for score in scores
    ]

    query_output = QueryOutput(predictions=predictions)
    assert query_output.predictions[0].prediction_score == max_score
    assert all(
        query_output.predictions[i].prediction_score == score
        for i, score in enumerate(sorted(scores, reverse=True))
    )


@pytest.mark.parametrize(
    "context",
    [None, "document", ["documentA", "documentB"]],
    ids=["context=None", "context=str", "context=List[str]"],
)
def test_query_output_from_sequence_classification(
    context,
    model_api_sequence_classification_ouput_factory,
):
    n = 3
    answers = ["door {i}".format(i=i) for i in range(n)]
    model_api_output = model_api_sequence_classification_ouput_factory(n=n)
    context = None
    query_output = QueryOutput.from_sequence_classification(
        answers=answers, model_api_output=model_api_output, context=context
    )

    if context is None:
        assert all(p.prediction_documents == [] for p in query_output.predictions)
    elif isinstance(context, str):
        assert all(
            p.prediction_documents == [PredictionDocument(document=context)]
            for p in query_output.predictions
        )
    elif isinstance(context, list):
        assert all(
            p.prediction_documents == [PredictionDocument(document=c) for c in context]
            for p in query_output.predictions
        )


@pytest.mark.parametrize(
    "context,n",
    [(None, 3), ("document", 3), (["documentA", "documentB"], 2)],
    ids=["context=None", "context=str", "context=List[str]"],
)
def test_query_output_from_question_answering(
    context, n, model_api_question_answering_ouput_factory
):
    model_api_output = model_api_question_answering_ouput_factory(n=n)
    query_output = QueryOutput.from_question_answering(
        model_api_output=model_api_output, context=context
    )
    if context is None:
        assert all(p.prediction_documents == [] for p in query_output.predictions)
    elif isinstance(context, str):
        # span=[0, 0] is hardcoded in the model_api_question_answering_ouput_factory
        assert all(
            p.prediction_documents
            == [PredictionDocument(document=context, span=[0, 0])]
            for p in query_output.predictions
        ), query_output
    elif isinstance(context, list):
        # span=[0, 0] is hardcoded in the model_api_question_answering_ouput_factory
        scores = [qa["score"] for qa in model_api_output["answers"]]
        sorted_contexts = [c for _, c in sorted(zip(scores, context), reverse=True)]
        assert all(
            p.prediction_documents
            == [PredictionDocument(document=sorted_contexts[i], span=[0, 0])]
            for i, p in enumerate(query_output.predictions)
        ), (query_output, sorted_contexts)
