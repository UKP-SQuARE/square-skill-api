import logging
from collections.abc import Iterable as cIterable
from itertools import zip_longest
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, root_validator

logger = logging.getLogger(__name__)

NO_ANSWER_FOUND_STRING = "No answer found."


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class PredictionOutput(BaseModel):
    """Holds the output (e.g. an answer) and the score of that output."""

    output: str = Field(
        ...,
        description="The actual output of the model as string. "
        "Could be an answer for QA, an argument for AR or a label for Fact Checking.",
    )
    output_score: float = Field(..., description="The score assigned to the output.")


class PredictionDocument(BaseModel):
    """Holds a Document a prediction is based on."""

    index: str = Field(
        "", description="From which document store the document has been retrieved"
    )
    document_id: str = Field("", description="Id of the document in the index")
    document: str = Field(..., description="The text of the document")
    span: Optional[List[int]] = Field(
        None, description="Start and end character index of the span used. (optional)"
    )
    url: str = Field("", description="URL source of the document (if available)")
    source: str = Field("", description="The source of the document (if available)")
    document_score: float = Field(
        0, description="The score assigned to the document by retrieval"
    )


class Node(BaseModel):
    id: int
    name: str
    q_node: bool
    ans_node: bool
    weight: float


class Edge(BaseModel):
    source: int
    target: int
    weight: float
    label: str


class SubGraph(BaseModel):
    nodes: Dict[str, Node]
    edges: Dict[str, Edge]


class PredictionGraph(BaseModel):
    lm_subgraph: SubGraph
    attn_subgraph: SubGraph


class TokenAttribution(BaseModel):
    __root__: List = Field(
        ...,
        description="A list holding three items: (1) the index, (2) the word and (3) the score.",
    )


class Attributions(BaseModel):
    topk_question_idx: List[int]
    topk_context_idx: List[int]
    question_tokens: List[TokenAttribution]
    context_tokens: List[TokenAttribution]


class Adversarial(BaseModel):
    indices: List[int] = Field(None)


class Prediction(BaseModel):
    """A single prediction for a query."""

    question: str = Field(..., description="The question that was asked.")
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
    prediction_graph: Union[None, PredictionGraph] = Field(None)
    attributions: Union[None, Attributions] = Field(
        None, description="Feature attributions for the question and context"
    )
    bertviz: Union[None, str] = Field(None, description="BertViz visualization data")
    skill_id: str = Field(
        None, description="The id of the skill that made this prediction"
    )


class QueryOutput(BaseModel):
    """The model for output that the skill returns after processing a query."""

    predictions: List[Prediction] = Field(
        ...,
        description="All predictions for the query. Predictions are sorted by prediction_score (descending)",
    )
    adversarial: Union[None, Adversarial] = Field(None)

    @staticmethod
    def sort_predictions_key(p: Union[Prediction, Dict]) -> Tuple:
        """Returns a key for soring predictions."""
        document_score = 1
        if isinstance(p, Prediction):
            answer_found = p.prediction_output.output not in [
                "",
                NO_ANSWER_FOUND_STRING,
            ]
            answer_score = p.prediction_score
            if p.prediction_documents:
                document_score = getattr(p.prediction_documents[0], "document_score", 1)
        elif isinstance(p, Dict):
            answer_found = p["prediction_output"]["output"] not in [
                "",
                NO_ANSWER_FOUND_STRING,
            ]
            answer_score = p["prediction_score"]
            if p["prediction_documents"]:
                document_score = p["prediction_documents"][0].get("document_score", 1)
        else:
            raise TypeError(type(p))
        return (answer_found, answer_score, document_score)

    @staticmethod
    def overwrite_from_model_api_output(
        model_api_output, key: str, value, extend_to_len: int = None
    ) -> List[str]:
        """
        If `key` is given in the model_api_output, overwrite value with it.
        Else return provided value.
        """
        if key in model_api_output and model_api_output[key]:
            value = model_api_output[key]
        if value is None:
            value = ""
        if isinstance(value, str):
            value = [value] * extend_to_len
        return value

    @staticmethod
    def get_attribution_by_index(
        attributions: List[Dict[str, List[List[int]]]], index
    ) -> Attributions:
        attribution_keys = attributions[0].keys()
        context_attributions = {}
        for k in attribution_keys:
            context_attributions[k] = attributions[0][k][index]

        return Attributions.parse_obj(context_attributions)

    @root_validator
    def sort_predictions(cls, values):
        """Sorts predictions according the keys generated by `sort_prediction_key`. If
        `adversarial` are given, disable sorting.

        Args:
            v (List[Union[Prediction, Dict]]): List of unsorted predictions

        Returns:
            List[Union[Prediction, Dict]]: List of sorted predictions
        """

        if values["adversarial"] is None:
            values["predictions"] = sorted(
                values["predictions"], key=cls.sort_predictions_key, reverse=True
            )

        return values

    @classmethod
    def from_sequence_classification(
        cls,
        questions: Union[str, List[str]],
        answers: List[str],
        model_api_output: Dict,
        context: Union[None, str, List[str]] = None,
    ):
        """Constructor for QueryOutput from sequence classification of model api.

        Args:
            answers (List[str]): List of answer strings
            model_api_output (Dict): Output returned from the model api.
            context (Union[None, str, List[str]], optional): Context used to obtain
            model api output. Defaults to None.
        """

        is_attack = len(model_api_output.get("adversarial", [])) > 0
        model_api_logits = model_api_output["model_outputs"]["logits"]
        attributions = model_api_output.get("attributions", None)
        bertviz = model_api_output.get("bertviz", None)
        if len(model_api_logits) > 1:
            # for categorical skills when using attack method logits are 2d
            logits = model_api_logits
        else:
            logits = model_api_logits[0]
            top_answer_idx = np.argmax(logits)

        questions = cls.overwrite_from_model_api_output(
            model_api_output,
            key="questions",
            value=questions,
            extend_to_len=len(logits),
        )
        context = cls.overwrite_from_model_api_output(
            model_api_output,
            key="contexts",
            value=context,
            extend_to_len=len(logits),
        )

        logger.info(f"is_attack={is_attack}")
        logger.info(f"questions={questions}")
        logger.info(f"context={context}")
        logger.info(f"attributions={attributions}")
        if bertviz:
            logger.info(f"bertviz: {bertviz[:100]} ...")
        logger.info(f"logits={logits}")
        logger.info(f"answers={answers}")

        predictions = []
        for i, answer_score in enumerate(logits):
            if isinstance(answer_score, cIterable):
                top_answer_idx = np.argmax(answer_score)
                answer_score = answer_score[top_answer_idx]
                answer = answers[top_answer_idx]
            elif is_attack:
                answer = answers[top_answer_idx]
            else:
                answer = answers[i]

            prediction_output = PredictionOutput(
                output=answer, output_score=answer_score
            )
            prediction = Prediction(
                question=questions[i],
                prediction_score=answer_score,
                prediction_output=prediction_output,
                prediction_documents=[PredictionDocument(document=context[i])],
            )
            if attributions:
                if len(attributions[0]["topk_question_idx"]) == 1:
                    # attributions only for top_answer
                    if i == top_answer_idx:
                        index = 0
                    else:
                        continue
                else:
                    # attributions for all answers
                    index = i
                prediction.attributions = cls.get_attribution_by_index(
                    attributions, index=index
                )
            if bertviz and len(bertviz) < i:
                prediction.bertviz = bertviz[i]
            predictions.append(prediction)

        if is_attack:
            predictions = cls(
                predictions=predictions, adversarial=model_api_output["adversarial"]
            )
        else:
            predictions = cls(predictions=predictions)

        return predictions

    @classmethod
    def from_sequence_classification_with_graph(
        cls,
        questions: Union[str, List[str]],
        answers: List[str],
        model_api_output: Dict,
    ):
        questions = cls.overwrite_from_model_api_output(
            model_api_output,
            key="questions",
            value=questions,
            extend_to_len=len(model_api_output["model_outputs"]["logits"][0]),
        )
        predictions = []
        predictions_scores = model_api_output["model_outputs"]["logits"][0]
        for i, (question, prediction_score, answer) in enumerate(
            zip(questions, predictions_scores, answers)
        ):
            prediction_output = PredictionOutput(
                output=answer, output_score=prediction_score
            )
            prediction = Prediction(
                question=question,
                prediction_score=prediction_score,
                prediction_output=prediction_output,
            )

            if i == model_api_output["labels"][0]:
                # add subgraphs to the predicted answer
                prediction_graph = PredictionGraph(
                    lm_subgraph=model_api_output["lm_subgraph"],
                    attn_subgraph=model_api_output["attn_subgraph"],
                )
                prediction.prediction_graph = prediction_graph

            predictions.append(prediction)

        return cls(predictions=predictions)

    @classmethod
    def from_question_answering(
        cls,
        questions: Union[str, List[str]],
        model_api_output: Dict,
        context: Union[None, str, List[str]] = None,
        context_score: Union[None, float, List[float]] = None,
    ):
        """Constructor for QueryOutput from question answering of model api.

        Args:
            model_api_output (Dict): Output returned from the model api.
            context (Union[None, str, List[str]], optional): Context used to obtain
            model api output. Defaults to None.
            context_score (Union[None, float, List[float]], optional): Context scores
            from datastores.
        """
        logger.debug(f"input questions: {questions}")
        logger.debug(f"input context: {context}")

        questions = cls.overwrite_from_model_api_output(
            model_api_output,
            value=questions,
            key="questions",
            extend_to_len=len(model_api_output["answers"]),
        )

        context = cls.overwrite_from_model_api_output(
            model_api_output,
            value=context,
            key="contexts",
            extend_to_len=len(model_api_output["answers"]),
        )

        # TODO: make this work with the datastore api output to support all
        # prediction_document fields
        predictions: List[Prediction] = []

        attributions = model_api_output.get("attributions", None)
        bertviz = model_api_output.get("bertviz", None)
        logger.info(f"attributions: {attributions}")
        if bertviz:
            logger.info(f"bertviz: {bertviz[:100]} ...")
        logger.info(f"questions: {questions}")
        logger.info(f"context: {context}")
        logger.info(f"answers: {model_api_output['answers']}")
        # loop over contexts
        for i_context, (question, document, answers) in enumerate(
            zip(questions, context, model_api_output["answers"])
        ):
            if isinstance(context_score, list):
                document_score = context_score[i_context]
            elif isinstance(context_score, float):
                document_score = context_score
            else:
                document_score = 1
            # get the sorted attributions for the answers from one doc
            scores = [answer["score"] for answer in answers]
            top_answer_idx = np.argmax(scores)

            # loop over answers per doc
            for i_answer, (answer, prediction_score) in enumerate(
                zip(
                    answers,
                    scores,
                )
            ):
                answer_str = answer["answer"]
                if not answer_str:
                    answer_str = NO_ANSWER_FOUND_STRING

                prediction_output = PredictionOutput(
                    output=answer_str, output_score=prediction_score
                )
                # NOTE: currently only one document per answer is supported
                prediction_documents = [
                    PredictionDocument(
                        document=document,
                        span=[answer["start"], answer["end"]],
                        document_score=document_score,
                    )
                ]
                prediction = Prediction(
                    question=question,
                    prediction_score=prediction_score,
                    prediction_output=prediction_output,
                    prediction_documents=prediction_documents,
                )
                if attributions and i_answer == top_answer_idx:
                    prediction.attributions = cls.get_attribution_by_index(
                        attributions, index=i_context
                    )
                logger.debug(f"prediction: {prediction}")
                predictions.append(prediction)

        if "adversarial" in model_api_output:
            predictions = cls(
                predictions=predictions, adversarial=model_api_output["adversarial"]
            )
        else:
            predictions = cls(predictions=predictions)
        # add bertviz to the first prediction (bertviz is the same for all predictions)
        if bertviz:
            predictions.predictions[0].bertviz = bertviz[0]

        return predictions

    @classmethod
    def from_information_retrieval(
        cls,
        questions: str,
        context: Union[None, str, List[str]],
        context_score: Union[None, float, List[float]],
    ):
        """Constructor for QueryOutput from information retrieval of model api.

        Args:
            questions (Union[str, List[str]]): requested query
            context (Union[None, str, List[str]]): Context used to obtain model api output.
            context_score (Union[None, float, List[float]]): Context scores from datastores.
        """

        if isinstance(questions, str) and isinstance(context, list):
            questions = [questions] * len(context)

        predictions: List[Prediction] = []

        logger.info(f"questions: {questions}")
        logger.info(f"context: {context}")

        # loop over contexts, and add each document as the entire prediction
        for i_context, (question, document) in enumerate(zip(questions, context)):
            if isinstance(context_score, list):
                document_score = context_score[i_context]
            elif isinstance(context_score, float):
                document_score = context_score
            else:
                document_score = 1

            # prediction output is usually the answer from the qa-model
            # but in this case we're outputting the retrieved document
            prediction_output = PredictionOutput(
                output=document, output_score=document_score
            )

            prediction = Prediction(
                question=question,
                prediction_score=document_score,
                prediction_output=prediction_output,
            )

            logger.debug(f"prediction: {prediction}")
            predictions.append(prediction)

        predictions = cls(predictions=predictions)

        return predictions

    @classmethod
    def from_generation(
        cls,
        questions: Union[str, List[str]],
        model_api_output: Dict,
        context: Union[None, str, List[str]] = None,
        context_score: Union[None, float, List[float]] = None,
    ):
        """Constructor for QueryOutput from generation of model api.

        Args:
            model_api_output (Dict): Output returned from the model api.
            context (Union[None, str, List[str]], optional): Context used to obtain
            model api output. Defaults to None.
            context_score (Union[None, float, List[float]], optional): Context scores
            from datastores.
        """
        # TODO: add support for batched outputs
        generated_texts = model_api_output["generated_texts"][0]
        if "sequences_scores" in model_api_output["model_outputs"]:
            sequences_scores = model_api_output["model_outputs"]["sequences_scores"][0]
            sequences_scores = softmax(sequences_scores)
        else:
            sequences_scores = [1.0]*len(generated_texts)
            
        questions = cls.overwrite_from_model_api_output(
            key="",
            value=questions,
            model_api_output=model_api_output,
            extend_to_len=len(generated_texts),
        )
        context = cls.overwrite_from_model_api_output(
            key="",
            value=context,
            model_api_output=model_api_output,
            extend_to_len=len(generated_texts),
        )

        predictions: List[Prediction] = []
        for question, context, generation, score in zip(
            questions, context, generated_texts, sequences_scores
        ):
            prediction_output = PredictionOutput(output=generation, output_score=score)
            prediction = Prediction(
                question=question,
                prediction_score=score,
                prediction_output=prediction_output,
                prediction_documents=[PredictionDocument(document=context)],
            )
            predictions.append(prediction)

        return cls(predictions=predictions)
