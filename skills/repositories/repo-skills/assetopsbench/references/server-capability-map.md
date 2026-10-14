# Server capability map

Extracted from the six FastMCP server modules by AST, so a tool named here is a
tool that is registered. `scripts/check_servers.py` asserts these names against
the live surface; when the two disagree, the server is right and this file is
stale.

## `iot` (12 tools)

- `iot.sites()` -> `SitesResult`
- `iot.asset_ids(site_name: str)` -> `Union[AssetsResult, ErrorResult]`
- `iot.asset_detail(site_name: str, asset_id: str)` -> `Union[AssetDetail, ErrorResult]`
- `iot.measured_sensors(site_name: str, asset_id: str)` -> `Union[SensorsResult, ErrorResult]`
- `iot.installed_sensors(site_name: str, asset_id: str)` -> `Union[SensorsResult, ErrorResult]`
- `iot.assets(site_name: str, assettype: Optional[str])` -> `Union[AssetsWithMetadataResult, ErrorResult]`
- `iot.find_assets_by_sensors(site_name: str, sensors: List[str], match: str, substring: bool, source: str)` -> `Union[FindAssetsResult, ErrorResult]`
- `iot.stream_extent(site_name: str, asset_id: str, sensor: Optional[str], start: Optional[str], end: Optional[str])` -> `Union[StreamExtentResult, ErrorResult]`
- `iot.history(site_name: str, asset_id: str, start: Optional[str], end: Optional[str], sensors: Optional[List[str]], limit: int, cursor: Optional[str])` -> `Union[HistoryResult, ErrorResult]`
- `iot.latest_reading(site_name: str, asset_id: str, sensor: Optional[str])` -> `Union[LatestReadingResult, ErrorResult]`
- `iot.sensor_coverage(site_name: str, asset_id: str)` -> `Union[SensorCoverageResult, ErrorResult]`
- `iot.sensor_stats(site_name: str, asset_id: str, sensor: Optional[str], start: Optional[str], end: Optional[str])` -> `Union[SensorStatsResult, ErrorResult]`

## `fmsr` (3 tools)

- `fmsr.get_failure_modes(asset_class: str)` -> `Union[FailureModesResult, ErrorResult]`
- `fmsr.generate_failure_modes(asset_class: str, max_modes: int)` -> `Union[GenerateFailureModesResult, ErrorResult]`
- `fmsr.add_failure_modes(asset_class: str, failure_modes: List[str], exhaustive: Optional[bool], source: Optional[str])` -> `Union[AddFailureModesResult, ErrorResult]`

## `tsfm` (41 tools)

- `tsfm.list_tasks()` -> `Union[TasksResult, ErrorResult]`
- `tsfm.profile_series(dataset_path: str, timestamp_column: Optional[str], channels: Optional[List[str]])` -> `Union[ProfileResult, ErrorResult]`
- `tsfm.characterize_series(dataset_path: str, timestamp_column: Optional[str], channels: Optional[List[str]], groups: Optional[dict], group_rules: Optional[str])` -> `Union[CharacterizeResult, ErrorResult]`
- `tsfm.data_quality(dataset_path: str, timestamp_column: str)` -> `Union[DataQualityResult, ErrorResult]`
- `tsfm.list_features(kind: Optional[str], status: Optional[str])` -> `Union[FeaturesResult, ErrorResult]`
- `tsfm.list_models(task_id: Optional[str], domain: Optional[str], status: str)` -> `Union[ModelsResult, ErrorResult]`
- `tsfm.search_models(text: str, tags: Optional[List[str]], status: str)` -> `Union[ModelsResult, ErrorResult]`
- `tsfm.find_models(task_id: str, min_context_length: Optional[int], prediction_length: Optional[int], domain: Optional[str], top_k: int)` -> `Union[ModelsResult, ErrorResult]`
- `tsfm.describe_candidates(task_id: str, top_k: int, domain: Optional[str])` -> `Union[CandidatesResult, ErrorResult]`
- `tsfm.describe_models(model_ids: List[str])` -> `Union[DescribeModelsResult, ErrorResult]`
- `tsfm.count_models()` -> `Union[ModelCountResult, ErrorResult]`
- `tsfm.list_domains(task_id: Optional[str])` -> `Union[DomainsResult, ErrorResult]`
- `tsfm.get_model_lineage(model_id: str)` -> `Union[LineageResult, ErrorResult]`
- `tsfm.register_model(model: dict)` -> `Union[RegisterResult, ErrorResult]`
- `tsfm.model_template()` -> `ModelTemplateResult`
- `tsfm.register_finetuned(model_id: str, checkpoint_path: str, base_model_id: str, context_length: int, prediction_length: int, description: str, domain: str)` -> `Union[CardResult, ErrorResult]`
- `tsfm.update_model(model_id: str, fields: dict)` -> `Union[CardResult, ErrorResult]`
- `tsfm.deprecate_model(model_id: str, reason: Optional[str])` -> `Union[CardResult, ErrorResult]`
- `tsfm.new_model_version(model_id: str, fields: dict, new_model_id: Optional[str])` -> `Union[CardResult, ErrorResult]`
- `tsfm.resolve_model(model_id: str)` -> `Union[ResolveResult, ErrorResult]`
- `tsfm.hf_stats(model_id: Optional[str], hf_repo: Optional[str])` -> `Union[HfStatsResult, ErrorResult]`
- `tsfm.count_features()` -> `Union[FeatureCountResult, ErrorResult]`
- `tsfm.describe_features(names: List[str])` -> `Union[DescribeFeaturesResult, ErrorResult]`
- `tsfm.extract_features(dataset_path: str, extractors: List[str], target_columns: List[str], timestamp_column: Optional[str], window: Optional[int])` -> `Union[ExtractResult, ErrorResult]`
- `tsfm.select_features(dataset_path: str, channel: str, extractors: List[str], timestamp_column: Optional[str], reference_feature: str, cd_margin: float)` -> `Union[FeatureSelectionResult, ErrorResult]`
- `tsfm.search_features(text: str, tags: Optional[List[str]], status: Optional[str])` -> `Union[FeaturesResult, ErrorResult]`
- `tsfm.get_feature(feature_id: str)` -> `Union[CardResult, ErrorResult]`
- `tsfm.register_feature(feature: dict, overwrite: bool)` -> `Union[RegisterResult, ErrorResult]`
- `tsfm.update_feature(feature_id: str, fields: dict)` -> `Union[CardResult, ErrorResult]`
- `tsfm.deprecate_feature(feature_id: str, reason: Optional[str])` -> `Union[CardResult, ErrorResult]`
- `tsfm.new_feature_version(feature_id: str, fields: Optional[dict], new_feature_id: Optional[str])` -> `Union[CardResult, ErrorResult]`
- `tsfm.get_feature_lineage(feature_id: str)` -> `Union[LineageResult, ErrorResult]`
- `tsfm.recipe_template()` -> `RecipeTemplateResult`
- `tsfm.run_recipe(dataset_path: str, timestamp_column: str, target_columns: List[str], recipe: dict, asset_id: str, parent_run_id: Optional[str])` -> `Union[RecipeResult, ErrorResult]`
- `tsfm.run_tabular_recipe(dataset_path: str, recipe: dict, label_column: Optional[str], asset_id: str)` -> `Union[TabularResult, ErrorResult]`
- `tsfm.run_plan(plan_spec: dict, asset_id: str, scenario_id: Optional[str])` -> `Union[PlanResult, ErrorResult]`
- `tsfm.evaluate(recipe: dict, configs: List[dict])` -> `Union[EvaluateResult, ErrorResult]`
- `tsfm.get_result(task_type: str, result_id: str)` -> `Union[ResultRecord, ErrorResult]`
- `tsfm.list_results(task_type: str, asset_id: Optional[str], scenario_id: Optional[str])` -> `ResultsListResult`
- `tsfm.get_run(run_id: str)` -> `Union[RunRecord, ErrorResult]`
- `tsfm.list_runs(asset_id: Optional[str])` -> `RunsResult`

## `vibration` (8 tools)

- `vibration.get_vibration_data(site_name: str, asset_id: str, sensor_name: str, start: str, final: Optional[str])` -> `Union[dict, ErrorResult]`
- `vibration.list_vibration_sensors(site_name: str, asset_id: str)` -> `Union[dict, ErrorResult]`
- `vibration.compute_fft_spectrum(data_id: str, window: str, top_n: int)` -> `Union[dict, ErrorResult]`
- `vibration.compute_envelope_spectrum(data_id: str, band_low_hz: Optional[float], band_high_hz: Optional[float], top_n: int)` -> `Union[dict, ErrorResult]`
- `vibration.assess_vibration_severity(rms_velocity_mm_s: float, machine_group: str)` -> `dict`
- `vibration.calculate_bearing_frequencies(rpm: float, n_balls: int, ball_diameter_mm: float, pitch_diameter_mm: float, contact_angle_deg: float, bearing_name: str)` -> `dict`
- `vibration.list_known_bearings()` -> `dict`
- `vibration.diagnose_vibration(data_id: str, rpm: Optional[float], bearing_designation: Optional[str], bearing_n_balls: Optional[int], bearing_ball_dia_mm: Optional[float], bearing_pitch_dia_mm: Optional[float], bearing_contact_angle_deg: float, bpfo_hz: Optional[float], bpfi_hz: Optional[float], bsf_hz: Optional[float], ftf_hz: Optional[float], machine_group: str, machine_description: str)` -> `Union[dict, ErrorResult]`

## `utilities` (6 tools)

- `utilities.json_reader(file_name: str)` -> `str`
- `utilities.get_sensor_catalog(sensor: Optional[str])` -> `Union[CatalogResult, ErrorResult]`
- `utilities.get_asset_catalog(asset: Optional[str], category: Optional[str])` -> `Union[CatalogResult, ErrorResult]`
- `utilities.get_failure_mode_catalog(failure_mode: Optional[str], category: Optional[str])` -> `Union[CatalogResult, ErrorResult]`
- `utilities.current_date_time()` -> `DateTimeResult`
- `utilities.current_time_english()` -> `TimeEnglishResult`

## `wo` (15 tools)

- `wo.list_workorders`
- `wo.get_workorder`
- `wo.get_workorder_tasks`
- `wo.get_workorder_costs`
- `wo.get_workorder_actuals_vs_planned`
- `wo.get_workorder_kpis`
- `wo.get_schedule_calendar`
- `wo.get_my_assigned_workorders`
- `wo.get_failure_codes`
- `wo.generate_work_order`
- `wo.update_workorder`
- `wo.approve_workorder`
- `wo.assign_technician`
- `wo.close_workorder`
- `wo.cancel_workorder`

## What has no MCP tool and must be done in the code track



The servers retrieve, catalog and run recipes. They do not do the following, so
these belong in the terminal agent's workspace using the library's own scripts:

- responsible-variable attribution (contribution plots, RBC, SHAP)
- propagation direction (lead-lag, Granger, transfer entropy)
- spectral admissibility (Nyquist, resolution, defect separation)
- bearing defect frequencies from geometry when the bearing is not in the database
- refrigerant-side thermodynamics (superheat, subcooling, approach, cycle COP)
- heat-exchanger UA and fouling attribution
- Weibull and survival fitting, PM interval optimisation, P-F detection probability
- work-order code-quality auditing and crosswalk loss
- RPN lattice auditing and criticality ranking
- alarm rate, flood, chattering and Pareto metrics
- health-indicator suitability screening (monotonicity, trendability, prognosability)
