from pydantic import BaseModel


class HealthIndexResult(BaseModel):
    asset_name: str
    health_index: float
    condition: str


class DGAInterpretationResult(BaseModel):
    asset_name: str
    fault_type: str
    r1: float
    r2: float
    r3: float
    code: str
    confidence: str
    reasoning: str
    recommended_action: str


class WindingTemperatureResult(BaseModel):
    asset_name: str
    thermal_status: str
    hot_spot_rise_c: float
    ageing_rate: float
    alarm_active: bool
    trip_active: bool
    risk_level: str
    reasoning: str
    recommended_action: str


class LoadProfileResult(BaseModel):
    asset_name: str
    load_mva: float
    load_factor_pct: float
    loading_status: str
    current_imbalance_pct: float
    neutral_current_flag: bool
    reasoning: str
    recommended_action: str
