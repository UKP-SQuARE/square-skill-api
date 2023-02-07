# SQuARE Skill API
This package is used for providing a unified API for all skills and facilitating skill developers. The package includes building an FastAPI application, but only the predict function of a skill needs to implemented.

## Installation
To install the latest stable version run:
```bash
pip install square-skill-api
```
To install from the master branch:
```bash
pip install git+https://github.com/UKP-SQuARE/square-skill-api/archive/master.tar.gz
```

## Usage
After installing, a simple predict function can be implemented and this package will create a FastAPI app from it.
```python3
from square_skill_api import get_app
from square_skill_api.models import QueryOutput, Prediction, PredictionOutput, QueryRequest

async def predict(request: QueryRequest) -> QueryOutput:
    # here goes the logic for handling the input request.
    # in this example, we simply return a static output.
    return QueryOutput(
        predictions=[
            Prediction(
                prediction_score=1, 
                prediction_output=PredictionOutput(output="42", output_score=1)
            )
        ]
    )

if __name__ == "__main__":
    app = get_app(predict_fn=predict)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
This builds an api with two endpoints `/health/heartbeat` and `/query`, that can be queried with curl, or via the docs at [http://localhost:8000/docs](http://localhost:8000/docs).
```bash
curl -X GET http://localhost:8000/health/heartbeat
# {"is_alive":true}
curl -X POST http://localhost:8000/query \
    -H 'Content-Type: application/json' \
    -d '{ "query": "string"}'
# {"predictions":[{"prediction_score":1.0,"prediction_output":{"output":"42","output_score":1.0},"prediction_documents":[]}]}
```

# How to Contribute
Create a PR with your changes and do create a test to check that your changes work. You can find the tests in `.test` folder. You can run them by `python -m pytest` in the root folder of this project.
