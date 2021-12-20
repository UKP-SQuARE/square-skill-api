from typing import Dict, Union, Tuple, List, Optional, Iterable
from pydantic import Field, BaseModel, validator


class PredictionOutput(BaseModel):
    output: str = Field(
        ...,
        description="The actual output of the model as string. "
        "Could be an answer for QA, an argument for AR or a label for Fact Checking.",
    )
    output_score: float = Field(..., description="The score assigned to the output.")


class PredictionDocument(BaseModel):
    index: str = Field(
        "", description="From which document store the document has been retrieved"
    )
    document_id: str = Field("", description="Id of the document in the index")
    document: str = Field(..., description="The text of the document")
    span: Optional[List[int]] = Field(
        description="Start and end character index of the span used. (optional)"
    )
    url: str = Field("", description="URL source of the document (if available)")
    source: str = Field("", description="The source of the document (if available)")


class Prediction(BaseModel):
    """
    A single prediction for a query.
    """

    prediction_score: float = Field(
        ...,
        description="The overall score assigned to the prediction. Up to the Skill to decide how to calculate",
    )
    prediction_output: PredictionOutput = Field(
        ..., description="The prediction output of the skill."
    )
    prediction_documents: List[PredictionDocument] = Field(
        [],
        description="A list of the documents used by the skill to derive this prediction. "
        "Empty if no documents were used",
    )


class QueryOutput(BaseModel):
    """
    The model for output that the skill returns after processing a query.
    """

    predictions: List[Prediction] = Field(
        ...,
        description="All predictions for the query. Predictions are sorted by prediction_score (descending)",
    )

    @validator("predictions")
    def sort_predictions(cls, v):
        if isinstance(v[0], dict):
            return sorted(v, key=lambda p: p["prediction_score"], reverse=True)
        elif isinstance(v[0], Prediction):
            return sorted(v, key=lambda p: p.prediction_score, reverse=True)
        else:
            raise ValueError()

    @staticmethod
    def _prediction_documents_iter_from_context(
        iter_len: int, context: Union[None, str, List[str]]
    ) -> Iterable[PredictionDocument]:
        if context is None:
            # no context for all answers
            prediction_documents_iter = ([] for _ in range(iter_len))
        elif isinstance(context, str):
            # same context for all answers
            prediction_documents_iter = (
                [PredictionDocument(document=context)] for _ in range(iter_len)
            )
        elif isinstance(context, list):
            # different context for all answers
            if len(context) != iter_len:
                raise ValueError()
            prediction_documents_iter = [
                [PredictionDocument(document=c)] for c in context
            ]
        else:
            raise TypeError(type(context))

        return prediction_documents_iter

    @classmethod
    def from_sequence_classification(
        cls,
        answers: List[str],
        model_api_output: Dict,
        context: Union[None, str, List[str]] = None,
    ):
        """Constructor for QueryOutput from sequeunce classification of model api."""
        # TODO: make this work with the datastore api output to support all
        # prediction_document fields
        prediction_documents_iter = cls._prediction_documents_iter_from_context(
            iter_len=len(answers), context=context
        )

        predictions = []
        predictions_scores = model_api_output["model_outputs"]["logits"][0]
        for prediction_score, answer, prediction_documents in zip(
            predictions_scores, answers, prediction_documents_iter
        ):

            prediction_output = PredictionOutput(
                output=answer, output_score=prediction_score
            )

            prediction = Prediction(
                prediction_score=prediction_score,
                prediction_output=prediction_output,
                prediction_documents=prediction_documents,
            )
            predictions.append(prediction)

        return cls(predictions=predictions)

    @classmethod
    def from_question_answering(
        cls,
        model_api_output: Dict,
        context: Union[None, str, List[str]] = None,
    ):
        """Constructor for QueryOutput from question answering of model api."""
        # TODO: make this work with the datastore api output to support all
        # prediction_document fields
        qa_outputs = model_api_output["answers"][0]

        prediction_documents_iter = cls._prediction_documents_iter_from_context(
            iter_len=len(qa_outputs), context=context
        )

        predictions: List[Prediction] = []
        for qa, prediction_documents in zip(qa_outputs, prediction_documents_iter):

            answer = qa["answer"]
            if not answer:
                continue

            prediction_score = qa["score"]

            prediction_output = PredictionOutput(
                output=answer, output_score=prediction_score
            )
            if prediction_documents:

                for p in prediction_documents:
                    p.span = [qa["start"], qa["end"]]

            prediction = Prediction(
                prediction_score=prediction_score,
                prediction_output=prediction_output,
                prediction_documents=prediction_documents,
            )
            predictions.append(prediction)

        # No answer found
        if not len(predictions):
            prediction_documents_iter = cls._prediction_documents_iter_from_context(
                context
            )
            prediction_documents = next(prediction_documents_iter)
            prediction_documents[0].span = [0, 0]
            max_score = max(qa[0]["score"] for qa in qa_outputs["answers"])
            prediction = Prediction(
                prediction_score=max_score,
                prediction_output=PredictionOutput(
                    output="No answer found.", output_score=max_score
                ),
                prediction_documents=prediction_documents,
            )
            predictions.append(prediction)

        return cls(predictions=predictions)
