from fastapi import APIRouter, status
from .scheme import LogisticRegressionIn, LogisticRegressionOut, LogisticRegressionPredict


modules_router = APIRouter()


@modules_router.post(
    '/',
    summary='Logistic regression fit',
    status_code=status.HTTP_200_OK,
    response_model=LogisticRegressionOut
)
async def model_fit(
        logistic_regression_scheme: LogisticRegressionIn
) -> LogisticRegressionOut:
    pass


@modules_router.post(
    '/predict',
    summary='Logistic regression predict',
    status_code=status.HTTP_200_OK,
    response_model=int
)
async def model_predict(
        logistic_regression_scheme: LogisticRegressionPredict
) -> int:
    pass