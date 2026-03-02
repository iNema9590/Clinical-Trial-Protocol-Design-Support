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


class EligibilityRule(BaseModel):
    """A structured eligibility rule that can be evaluated against patient data."""
    text: str = Field(
        ...,
        description="The original criterion text from the protocol."
    )
    field: Optional[str] = Field(
        None,
        description="The data field to evaluate (e.g., 'AGE', 'BMI', 'PCR_RESULT'). None if not machine-evaluable."
    )
    operator: Optional[str] = Field(
        None,
        description="The comparison operator (e.g., '>=', '<=', '==', 'between', 'in'). None if not machine-evaluable."
    )
    value: Optional[str] = Field(
        None,
        description="The value(s) to compare against. For ranges, use format 'min,max'. None if not machine-evaluable."
    )
    evaluable: bool = Field(
        default=False,
        description="Whether this rule can be automatically evaluated against structured data."
    )


class EligibilityCriteria(BaseModel):
    """Inclusion and exclusion criteria with structured rules."""
    inclusion: List[EligibilityRule] = []
    exclusion: List[EligibilityRule] = []

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