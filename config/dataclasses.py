from pydantic import BaseModel
from typing import List, Optional


class AutoARIMAPredictRequest(BaseModel):
    data: List[float]
    n_periods: int


class HoltWintersPredictRequest(BaseModel):
    data: List[float]
    n_periods: int
    trend: Optional[str] = None  # 'additive' или 'multiplicative'.
    seasonal: Optional[str] = None  # 'additive', 'multiplicative', или None.
    