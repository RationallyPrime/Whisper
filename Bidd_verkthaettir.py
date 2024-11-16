from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date

class TaskSpecification(BaseModel):
    task_id: str
    description: str
    deliverables: List[str]
    estimated_hours: int
    required_skills: List[str]
    dependencies: Optional[List[str]] = None

class PhaseSpecification(BaseModel):
    phase_id: str
    name: str
    description: str
    start_date: date
    end_date: date
    tasks: List[TaskSpecification]
    estimated_cost: float
    key_risks: List[str]
    success_criteria: List[str]

