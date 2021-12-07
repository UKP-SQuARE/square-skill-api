import itertools

import pytest

from square_skill_api.models.prediction import QueryOutput

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
