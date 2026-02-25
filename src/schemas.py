from pydantic import BaseModel
from typing import List


class Objective(BaseModel):
    """A single objective with its endpoints."""
    objective: str
    endpoints: List[str]


class ObjectivesByCategory(BaseModel):
    """Objectives organized by category."""
    primary: List[Objective] = []
    secondary: List[Objective] = []
    exploratory: List[Objective] = []
    other: List[Objective] = []


class EligibilityCriteria(BaseModel):
    """Inclusion and exclusion criteria."""
    inclusion: List[str] = []
    exclusion: List[str] = []