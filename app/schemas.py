from typing import List, Optional

from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    occurred_at: str
    severity: int = Field(ge=1, le=5)
    incident_type_code: Optional[str] = None


class AnalyzeIn(BaseModel):
    description: str = Field(min_length=3)
    incident_type_code: Optional[str] = None
    occurred_at: str
    neighborhood_id: Optional[int] = None
    history: List[HistoryItem] = []


class HeatReport(BaseModel):
    id: Optional[int] = None
    latitude: float
    longitude: float
    occurred_at: str
    severity: int = Field(ge=1, le=5)
    incident_type_code: str = "unknown"
    neighborhood_id: int = 0


class HeatNeighborhood(BaseModel):
    id: int
    name: str
    latitude: float
    longitude: float


class HeatmapIn(BaseModel):
    reports: List[HeatReport] = []
    neighborhoods: List[HeatNeighborhood] = []


class SummaryReport(BaseModel):
    title: str
    description: str = ""
    incident_type: str
    severity: int
    status: str
    occurred_at: str


class SummaryHotspot(BaseModel):
    day_of_week: int
    hour_block: int
    risk_score: float


class SummaryIn(BaseModel):
    neighborhood_name: str
    date: str
    reports: List[SummaryReport] = []
    week_reports: List[SummaryReport] = []
    hotspots: List[SummaryHotspot] = []


class SwitchIn(BaseModel):
    classifier_model: Optional[str] = None
    heatmap_model: Optional[str] = None
    summary_provider: Optional[str] = None
    summary_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
