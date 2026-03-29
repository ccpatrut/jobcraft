"""Data models for the Job Finder application."""

from typing import Optional

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Extracted profile from user's CV documents."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    raw_text: str = ""  # Full extracted text for context


class JobListing(BaseModel):
    """A job posting from an API."""

    title: str
    company: str
    description: str
    location: str
    url: str
    salary: Optional[str] = None
    contract_type: Optional[str] = None
    source: str = "adzuna"


class UserPreferences(BaseModel):
    """User temperament and writing preferences."""

    tone: str = "professional"  # formal, semi-formal, casual, enthusiastic
    style: str = "balanced"  # concise, detailed, balanced
    focus_areas: list[str] = Field(default_factory=list)
    pivot_enabled: bool = False
    pivot_from: list[str] = Field(default_factory=list)
    pivot_motivation: str = ""
