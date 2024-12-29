from pydantic import BaseModel
from typing import List, Dict, Optional, Literal


class AutoARIMAPredictRequest(BaseModel):
    data: List[float]
    n_periods: int


class HoltWintersPredictRequest(BaseModel):
    data: List[float]
    n_periods: int
    trend: Optional[
        Literal['additive', 'multiplicative']
    ] = None
    seasonal: Optional[
        Literal['additive', 'multiplicative']
    ] = None


class Response(BaseModel):
    response: Dict[str, List[float]]
