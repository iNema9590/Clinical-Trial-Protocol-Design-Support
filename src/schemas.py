from pydantic import BaseModel, Field
from typing import List, Optional


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

class Procedure(BaseModel):
    name: str = Field(
        ...,
        description="Exact name of the procedure as written in the protocol."
    )

    description: str = Field(
        ...,
        description="Brief description of what the procedure involves."
    )


class Assessment(BaseModel):
    category: str = Field(
        ...,
        description="Category of the assessment, e.g. 'safety', 'efficacy', etc."
    )
    name: str = Field(
        ...,
        description="Exact name of the assessment as written in the protocol."
    )

    description: str = Field(
        ...,
        description="Brief description of the assessment."
    )

    procedures: List[Procedure] = Field(
        default_factory=list,
        description="Procedures that belong to this assessment."
    )


class KeyAssessmentsOutput(BaseModel):
    assessments: List[Assessment]


class VisitDefinition(BaseModel):
    name: str = Field(
        ...,
        description="Exact visit name as written in the protocol (e.g., Screening, Day 1, Follow-up)."
    )

    description: str = Field(
        ...,
        description="Brief description of the visit purpose or definition."
    )

    timing: Optional[str] = Field(
        None,
        description="When the visit occurs (e.g., Day 1, Week 4, 30 days after last dose)."
    )

    window: Optional[str] = Field(
        None,
        description="Visit window if specified (e.g., ±3 days)."
    )

    trigger: Optional[str] = Field(
        None,
        description="Trigger condition if the visit is conditional (e.g., symptom onset, positive test)."
    )


class VisitDefinitionsOutput(BaseModel):
    visits: List[VisitDefinition] = Field(
        default_factory=list,
        description="List of all visit definitions and timing rules."
    )