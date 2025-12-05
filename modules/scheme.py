from pydantic import BaseModel, field_validator, model_validator
from fastapi import HTTPException, status
from numpy import array, isnan, unique


class LogisticRegressionOut(BaseModel):

    fit: bool
    w_: list[float]


class LogisticRegressionIn(BaseModel):
    model_config = {
        'extra': 'forbid'
    }

    X: list[list[float]]
    y: list[int]


    @field_validator('X')
    def verify_X(cls, value):
        X = array(value)
        if X.ndim != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='Must be a 2D matrix'
            )
        if isnan(X).any():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='There should be no NaN values.'
            )
        return X


    @field_validator('y')
    def verify_y(cls, value):
        y = array(value)
        if y.ndim != 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='1D vector must be entered'
            )
        if len(unique(y)) != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='the target values should be 2 different'
            )
        if isnan(y).any():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='There should be no NaN values.'
            )
        return y


    @model_validator(mode='after')
    def verify_object(self):
        if self.X.shape[0] != self.y.shape[0]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='the number of samples and the number of target values are not the same'
            )
        return self


class LogisticRegressionPredict(BaseModel):
    model_config = {
        'extra': 'forbid'
    }

    X: list[list]


    @field_validator('X')
    def verify_X(cls, value):
        X = array(value)
        if X.ndim > 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='1D or 2D vector must be entered'
            )
        if isnan(X).any():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='There should be no NaN values.'
            )
        return X