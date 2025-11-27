from fastapi import APIRouter, status
from logistic_regression import LogisticRegressionGD
from .scheme import LogisticRegressionIn, LogisticRegressionOut, LogisticRegressionPredict


modules_router = APIRouter()

logistic_regression_gd = LogisticRegressionGD(n_iter=30_000)

@modules_router.post(
    '/',
    summary='Logistic regression fit',
    status_code=status.HTTP_200_OK,
    response_model=LogisticRegressionOut
)
async def model_fit(
        logistic_regression_scheme: LogisticRegressionIn
) -> LogisticRegressionOut:
    logistic_regression_out = LogisticRegressionOut(
        fit=logistic_regression_gd.fit(logistic_regression_scheme.X, logistic_regression_scheme.y),
        w_=logistic_regression_gd.w_
    )
    return logistic_regression_out


@modules_router.post(
    '/predict',
    summary='Logistic regression predict',
    status_code=status.HTTP_200_OK,
    response_model=int
)
async def model_predict(
        logistic_regression_scheme: LogisticRegressionPredict
) -> int:
    class_label = logistic_regression_gd.predict(logistic_regression_scheme.X)
    return class_label