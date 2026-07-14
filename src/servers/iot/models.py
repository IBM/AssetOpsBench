from typing import List, Optional

from pydantic import BaseModel


class ErrorResult(BaseModel):
    error: str


class SitesResult(BaseModel):
    sites: List[str]


class AssetsResult(BaseModel):
    site_name: str
    total_assets: int
    assets: List[str]
    message: str


class SensorsResult(BaseModel):
    site_name: str
    asset_id: str
    total_sensors: int
    sensors: List[str]
    message: str


class AssetDetail(BaseModel):
    site_name: str
    asset_id: str
    description: Optional[str]
    assettype: Optional[str]
    status: Optional[str]
    location: Optional[str]
    installdate: Optional[str]
    vintage: Optional[str]
    n_installed_sensors: int
    message: str


class AssetSummary(BaseModel):
    asset_id: str
    description: Optional[str]
    assettype: Optional[str]
    vintage: Optional[str]
    n_sensors: int


class AssetsWithMetadataResult(BaseModel):
    site_name: str
    total_assets: int
    assets: List[AssetSummary]
    message: str


class AssetSensorMatch(BaseModel):
    asset_id: str
    matched_sensors: List[str]


class FindAssetsResult(BaseModel):
    site_name: str
    query_sensors: List[str]
    match: str
    source: str
    total_assets: int
    assets: List[AssetSensorMatch]
    message: str


class StreamExtentResult(BaseModel):
    site_name: str
    asset_id: str
    sensor: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    total_records: int
    exceeds_page_limit: bool
    approx_interval_seconds: Optional[float]
    message: str


class SensorCoverage(BaseModel):
    sensor: str
    non_null_count: int
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]


class SensorCoverageResult(BaseModel):
    site_name: str
    asset_id: str
    docs_scanned: int
    sensors: List[SensorCoverage]
    message: str


class SensorStat(BaseModel):
    sensor: str
    count: int
    null_count: int
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    stddev: Optional[float]
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]


class SensorStatsResult(BaseModel):
    site_name: str
    asset_id: str
    stats: List[SensorStat]
    message: str
