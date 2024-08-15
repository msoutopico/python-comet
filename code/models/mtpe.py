from pydantic import BaseModel


class Translation(BaseModel):
    src: str
    mt: str


class ScoredTranslation(Translation):
    score: float
