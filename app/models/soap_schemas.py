"""
Pydantic models for structured SOAP (Subjective, Objective, Assessment, Plan)
clinical notes.

These schemas enforce the exact JSON structure that the LLM must produce,
ensuring consistent, validated output for downstream consumers (EHR systems,
dashboards, etc.).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SubjectiveSection(BaseModel):
    """What the patient reports — symptoms, history, and review of systems."""

    chief_complaint: str = Field(
        ...,
        description="Primary reason for the visit in the patient's own words",
    )
    history_of_present_illness: str = Field(
        ...,
        description="Detailed narrative of symptom onset, duration, and progression",
    )
    review_of_systems: Optional[str] = Field(
        None,
        description="Additional symptoms the patient confirms or denies",
    )


class ObjectiveSection(BaseModel):
    """Clinician's observations — vitals, physical exam findings."""

    vitals: Optional[str] = Field(
        None,
        description="Recorded vital signs (BP, HR, temp, etc.) if mentioned",
    )
    physical_exam: Optional[str] = Field(
        None,
        description="Physical examination findings noted by the clinician",
    )


class AssessmentSection(BaseModel):
    """Clinician's clinical judgment — diagnosis and differentials."""

    diagnosis: str = Field(
        ...,
        description="Primary diagnosis or clinical impression",
    )
    differential_diagnosis: List[str] = Field(
        default_factory=list,
        description="Alternative diagnoses being considered",
    )


class PlanSection(BaseModel):
    """Treatment plan — medications, follow-up, and patient education."""

    medications: List[str] = Field(
        default_factory=list,
        description="Prescribed or recommended medications",
    )
    follow_up: Optional[str] = Field(
        None,
        description="Follow-up instructions and timeline",
    )
    patient_education: Optional[str] = Field(
        None,
        description="Advice and education provided to the patient",
    )


class SOAPNote(BaseModel):
    """Complete SOAP note combining all four clinical sections."""

    subjective: SubjectiveSection
    objective: ObjectiveSection
    assessment: AssessmentSection
    plan: PlanSection


class SOAPNoteResponse(BaseModel):
    """API response wrapping the generated SOAP note with metadata."""

    filename: Optional[str] = Field(
        None, description="Source audio filename, if generated from audio"
    )
    soap_note: SOAPNote
    model_used: str = Field(
        ..., description="Name of the LLM model used for generation"
    )
    processing_time_seconds: float = Field(
        ..., description="Time taken to generate the SOAP note"
    )
