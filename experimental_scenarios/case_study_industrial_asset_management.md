# Industrial Asset Management: Utterance Design Case Study (Coming from AssetOpsBench)

## Overview

This document demonstrates the application of the [Utterance Design Guideline](utterance_design_guideline.md) to the Industrial Asset Management domain, specifically focusing on Facilities Management (HVAC/Chiller systems). This is a **partially accomplished case study** showing work-in-progress utterance development.

**Status**: This case study represents an ongoing effort to build a comprehensive utterance set for industrial asset management. It demonstrates the incremental development approach recommended in the guideline, with some areas fully developed and others identified for future work.

---

## Domain Context

### Equipment Types
Chillers, air handling units (AHUs), cooling towers, pumps, compressors, heat exchangers

### Measurement Types
Temperature, pressure, flow, power consumption, efficiency, tonnage, humidity

### Event Types
Alerts, anomalies, work orders, failure modes, alarms

### Time Granularity
Real-time to yearly (typically hourly for HVAC monitoring)

### Location Hierarchy
Enterprise → Sites → Buildings → HVAC Zones → Equipment → Components

### Action Types
Preventive maintenance (PM), corrective maintenance (CM), inspections, calibration

---

## Category Mapping for Industrial Asset Management

This section shows how the generic categories from the [main guideline](utterance_design_guideline.md) map to industrial asset management contexts.

### Information Retrieval → Equipment & Failure Knowledge
- **Industrial Context**: Querying equipment inventory, failure modes, sensor configurations
- **Examples**: "List all chillers at site MAIN", "What failure modes exist for Chiller 6?"

### Data Extraction → Sensor & Telemetry Data
- **Industrial Context**: Retrieving time-series sensor data, work order history, operational logs
- **Examples**: "Download Chiller 6 tonnage data from last week", "Get work orders for 2017"

### Analysis & Inference → Performance Forecasting
- **Industrial Context**: Time series forecasting, performance prediction, trend analysis
- **Examples**: "Forecast Chiller 9 energy consumption for next week", "Predict condenser water flow"

### Model Customization → Domain-Specific Model Training
- **Industrial Context**: Fine-tuning models for specific equipment or site conditions
- **Examples**: "Finetune forecasting model for Chiller 9 using site-specific data"

### Anomaly & Exception Detection → Equipment Anomaly Detection
- **Industrial Context**: Detecting abnormal equipment behavior, performance deviations
- **Examples**: "Detect anomalies in Chiller 6 tonnage", "Identify unusual power consumption patterns"

### Recommendation & Optimization → Maintenance Recommendations
- **Industrial Context**: Work order recommendations, maintenance scheduling, resource optimization
- **Examples**: "Recommend work orders for detected anomaly", "Suggest maintenance bundling strategy"

### Future State Prediction → Failure Prediction
- **Industrial Context**: Predicting equipment failures, work order probability, remaining useful life
- **Examples**: "Predict next work order probability", "Estimate failure risk over next 30 days"

### Multi-Step Orchestration → Integrated Diagnostics
- **Industrial Context**: Combined data retrieval, analysis, and recommendation workflows
- **Examples**: "Retrieve sensor data, detect anomalies, and recommend corrective actions"

---

## Applying the Three Core Categories

### 1. retrospective (Knowledge Extraction)

**Domain Translation**: Understanding historical equipment performance, current asset state, and operational history.

#### Knowledge Query Examples

**Inventory and Configuration**:
- "What IoT sites are available?"
- "Which assets are located at the MAIN facility?"
- "List all chillers at site MAIN"
- "What assets can be found at the MAIN site?"

**Failure Mode Analysis**:
- "List all failure modes of asset Chiller."
- "List all failure modes of asset Chiller 6."
- "List all failure modes of asset Wind Turbine." (non-deterministic - knowledge base dependent)

**Sensor and Monitoring**:
- "List all installed sensors of asset Chiller 6."
- "Can I list all the metrics monitored by CQPA AHU 2B? use site MAIN"
- "Provide some sensors of asset Wind Turbine." (non-deterministic)

**Failure Mode-Sensor Mapping**:
- "List all failure modes of Chiller 6 that can be detected by Chiller 6 Supply Temperature."
- "List all failure modes of Chiller 6 that can be detected by temperature sensors."
- "List all failure modes of Chiller 6 that can be detected by temperature sensors and power input sensors."
- "Get failure modes for Chiller 6 and only include in final response those that can be monitored using the available sensors."
- "Are there any failure modes of Chiller 6 that can be predicted by monitoring the vibration sensor data?"

**Sensor-Failure Mode Mapping**:
- "List all sensors of Chiller 6 that are potentially relevant to Compressor Overheating."
- "If compressor overheating occurs for Chiller 6, which sensor should be prioritized for monitoring this specific failure?"
- "If Evaporator Water side fouling occurs for Chiller 6, which sensor is most relevant for monitoring this specific failure?"

#### Data Query Examples

**Metadata Retrieval**:
- "Retrieve metadata for Chiller 6 located at the MAIN site."
- "Get the asset details for Chiller 9 at the MAIN site."
- "Download the metadata for Chiller 3 at the MAIN facility."

**Sensor Data Retrieval**:
- "Download sensor data for Chiller 6's Tonnage from the last week of 2020 at the MAIN site"
- "Retrieve sensor data for Chiller 6's % Loaded from June 2020 at MAIN."
- "Get sensor data for both Chiller 6 and Chiller 9's Tonnage from first week of June 2020 at MAIN in a single file."
- "Download all sensor data for Chiller 6 from the last week of April '20 at the MAIN site."
- "Retrieve sensor data for Chiller 6 from June 2020."

**Point-in-Time Queries**:
- "What was the latest supply humidity from CQPA AHU 1 at site MAIN on sept 3 2015? return in a file"
- "what was the supply temperature from CQPA AHU 2B on sept 19, 2020 at quarter to midnight, at site MAIN? return in a file"
- "how much power was CQPA AHU 1 (MAIN site) using on 6/14/20?"
- "What is the power consumption of CQPA AHU 1 on mar 13 '20, site MAIN?"
- "what was the return temperature from CQPA AHU 2B on sept 19, 2020 at 7pm, at site MAIN? return data in a file"

**Time Range Queries**:
- "retrieve the supply temperature data recorded last week for Chiller 3 (MAIN site)?"

#### Work Order History Examples

- "Get the work order of equipment CWC04013 for year 2017."

**Status**: Retrospective category is well-developed with 30+ utterances covering inventory, failure modes, sensors, and historical data retrieval.

---

### 2. predictive

**Domain Translation**: Forecasting equipment performance, predicting failures, and estimating future maintenance needs.

#### Time Series Forecasting Examples

**Model Availability Queries**:
- "What types of time series analysis are supported?"
- "What are time series pretrained models are available in system?"
- "Are any time series forecasting models supported?"
- "Is TTM (Tiny Time Mixture), a time series model supported?"
- "Is LSTM model supported in TSFM?" (Answer: No)
- "Is Chronos model supported in TSFM?" (Answer: No)
- "Is Anomaly Detection supported in TSFM?" (Answer: Yes)
- "Is Time Series Classification supported in TSFM?" (Answer: No)
- "Is any model with context length 96 supported in TSFM?"
- "Is any model with context length 1024 supported in TSFM?" (Answer: No)

**Forecasting Queries**:
- "What is the forecast for 'Chiller 9 Condenser Water Flow' in the week of 2020-04-27?"
- "Forecast 'Chiller 9 Condenser Water Flow' using data in 'chiller9_annotated_small_test.csv'."
- "Use data in 'chiller9_annotated_small_test.csv' to forecast with 'Timestamp' as a timestamp."

**Model Fine-tuning**:
- "Finetune a forecasting model for 'Chiller 9 Condenser Water Flow' using data in 'chiller9_finetuning_small.csv'."

#### Anomaly Detection Examples

- "I need to perform Time Series anomaly detection of 'Chiller 9 Condenser Water Flow' using data in chiller9_tsad.csv."
- "Is there any anomaly detected in Chiller 6's Tonnage in the week of 2020-04-27?"

#### Failure Prediction Examples

- "Can you predict next work order probability for equipment CWC04009?"
- "I would like to predict the next work order probability for equipment CWC04013."
- "Build a predictive model from historical alerts and work orders of CWC04009 to forecast failures."
- "Build a predictive model to forecast failures over the forthcoming 10-year horizon."

**Status**: Predictive category has moderate coverage (~20 utterances) focused on time series forecasting and anomaly detection. **Gap identified**: Need more utterances for remaining useful life (RUL) estimation and degradation trend analysis.

---

### 3. prescriptive

**Domain Translation**: Recommending maintenance actions, optimizing schedules, and supporting operational decisions.

#### Recommendation Examples

**Anomaly Response**:
- "When an anomaly happens for equipment CWC04009, can you recommend top three work orders to address this problem?"
- "How can I analyze anomalies across multiple KPIs to better diagnose the root cause of these issues?"

**Work Order Bundling**:
- "Which corrective work orders for equipment CWC04009 in year 2017 can be bundled in the next maintenance window?"

**Prioritization**:
- "Which work orders should I prioritize first for Chiller 9 in July 2020?"

#### Root Cause Analysis Examples

- "When power input of Chiller 6 drops, what is the potential failure that causes it?"
- "When the Liquid Refrigerant Evaporator Temperature of Chiller 6 drops, what failure is most likely to occur?"
- "When compressor motor of Chiller 6 fails, what is the temporal behavior of the power input?"

#### Diagnostic Recipe Examples

- "Purge unit of chiller 6 have possibility to excess purge, what is the plan by the maintenance experts to early detect the failure?"
- "Generate a machine learning recipe for detecting overheating failure for Chiller 6. Result should include feature sensors and target sensor."
- "I want to build an anomaly model for identifying a chiller trip failure for POKMAIN chiller 6. Provide me a list of sensors that I should use, along with the temporal behavior."
- "What are the failure modes of Chiller 6 that can be identified by analyzing the data from the available sensors?"

#### Alert Reasoning Examples

- "How can reasoning on operation alerts help in generating significant warning messages?"

**Status**: Prescriptive category is under-developed (~10 utterances). **Major gaps identified**: 
- Maintenance scheduling optimization
- Resource allocation
- Cost-benefit analysis
- Multi-asset prioritization
- Preventive maintenance planning

---

## Coverage Analysis

### Current Coverage Matrix

| Operational Scenario | Retrospective | Predictive | Prescriptive | Coverage Status |
|---------------------|---------------|------------|--------------|-----------------|
| Chiller performance monitoring | ✓ (8 utterances) | ✓ (4 utterances) | ✓ (2 utterances) | **Complete** |
| Failure mode analysis | ✓ (12 utterances) | ✗ (0 utterances) | ✓ (3 utterances) | **Partial** - Missing predictive |
| Sensor data retrieval | ✓ (15 utterances) | N/A | N/A | **Complete** |
| Work order management | ✓ (1 utterance) | ✓ (3 utterances) | ✓ (2 utterances) | **Partial** - Need more prescriptive |
| Energy optimization | ✗ (0 utterances) | ✓ (1 utterance) | ✗ (0 utterances) | **Incomplete** |
| Maintenance scheduling | ✗ (0 utterances) | ✗ (0 utterances) | ✓ (1 utterance) | **Incomplete** |
| Root cause diagnosis | ✓ (2 utterances) | ✗ (0 utterances) | ✓ (3 utterances) | **Partial** |
| Anomaly detection | ✓ (0 utterances) | ✓ (2 utterances) | ✓ (1 utterance) | **Partial** - Missing retrospective |

### Completeness Metrics

**Category Distribution** (Current):
- Retrospective: ~55% (38 utterances)
- Predictive: ~30% (20 utterances)
- Prescriptive: ~15% (10 utterances)

**Target Distribution**:
- Retrospective: 40%
- Predictive: 30%
- Prescriptive: 30%

**Assessment**: Prescriptive category is significantly under-represented. Need to develop ~10-15 more prescriptive utterances.

---

## SME Contribution Tracking

### Current Contributors

| SME Role | Expertise Area | Utterances Created | Priority Focus |
|----------|---------------|-------------------|----------------|
| Chiller Operations SME | Chiller monitoring & operations | 25 | Retrospective, Data queries |
| FMSA Expert | Failure modes & sensor analysis | 18 | Retrospective, Knowledge queries |
| Time Series Analytics SME | Forecasting & anomaly detection | 15 | Predictive |
| Maintenance Planning SME | Work order management | 5 | Prescriptive |
| **NEEDED** | Energy optimization | 0 | Prescriptive |
| **NEEDED** | Maintenance scheduling | 0 | Prescriptive |

### Identified Gaps Requiring SME Input

1. **Energy Optimization SME** (Priority: P1)
   - Energy consumption forecasting
   - Load optimization recommendations
   - Cost-benefit analysis for efficiency improvements

2. **Maintenance Scheduling SME** (Priority: P1)
   - Preventive maintenance planning
   - Resource allocation optimization
   - Multi-asset maintenance coordination

3. **Controls & Automation SME** (Priority: P2)
   - Control system integration queries
   - Setpoint optimization
   - Automated response strategies

---

## Utterance Schema for Industrial Asset Management

The original ALM dataset (`docs/alm/alm_utterance.json`) contains 152 utterances with 6 base fields. This guideline recommends adding 3 additional fields for better organization and metadata tracking.

### Base ALM Schema (6 fields - as in original data):
1. **id** (integer): Unique identifier
2. **text** (string): Natural language utterance
3. **type** (string): AI agent identifier
4. **category** (string): Classification
5. **deterministic** (boolean): Single correct answer or not
6. **characteristic_form** (string): Expected response description

### Recommended Additional Fields (3 fields - proposed enhancements):
7. **group** (string or array): RETROSPECTIVE, PREDICTIVE, PRESCRIPTIVE
8. **entity** (string or array): Primary subject(s)
9. **note** (string): Flexible metadata field

### Complete Enhanced Schema (9 fields):

Each utterance should follow this enhanced schema with 9 required fields in this order:

1. **id** (integer): Unique identifier
2. **text** (string): Natural language utterance
3. **type** (string): MCP server or AI agent identifier - which component processes this utterance
   - **IoT**: IoT data agent (handles sensor data, asset metadata, site information)
   - **FMSA**: Failure Mode & Sensor Analysis agent (handles failure modes, sensor relationships, diagnostic recipes)
   - **TSFM**: Time Series Foundation Model agent (handles forecasting, anomaly detection, model fine-tuning)
   - **Workorder**: Work order management agent (handles maintenance records, recommendations, scheduling)
   - **multiagent**: Multi-agent coordination (requires collaboration between multiple agents)
   - Note: In MCP systems, this could also identify MCP servers like "filesystem", "database", "web-search", etc.
4. **category** (string): Classification (see Category Mapping above)
5. **deterministic** (boolean): Single correct answer (true) or multiple valid responses (false)
6. **characteristic_form** (string): Expected response description
7. **group** (string or array): RETROSPECTIVE, PREDICTIVE, PRESCRIPTIVE (use lowercase in JSON: "retrospective", "predictive", "prescriptive")
8. **entity** (string or array): Primary physical subject(s) - equipment types (e.g., "Chiller", "AHU", "Pump", "Sensor", "Site"). Use physical things, not abstract concepts like "Anomaly" or "WorkOrder"
9. **note** (string): A flexible field for any information useful to explain or contextualize the utterance, including source, owner/contributor, omitted details, design rationale, implementation notes, or any other relevant metadata

**Important Notes**:
- **Original ALM Data**: The base dataset in `docs/alm/alm_utterance.json` contains only fields 1-6. Only one utterance (id 409) has a `note` field in the original data.
- **Recommended Enhancements**: Fields 7-9 (group, entity, note) are proposed additions for better organization, filtering, and metadata tracking.
- **Migration Path**: When adopting this guideline, existing utterances can be enhanced by adding the three recommended fields.
- **Flexibility**: The `note` field is intentionally flexible to accommodate diverse documentation needs.

---

## Example Utterance Schemas

### Example 1: RETROSPECTIVE - Site Inventory (IoT Type)

```json
{
  "id": 1,
  "text": "What IoT sites are available?",
  "type": "IoT",
  "category": "Knowledge Query",
  "deterministic": true,
  "characteristic_form": "The expected response should be the return value of all sites, either as text or as a reference to a file",
  "group": "retrospective",
  "entity": "Site",
  "note": "Source: Initial domain analysis; Owner: Domain SME Team; Basic inventory query pattern for site discovery"
}
```

### Example 2: PRESCRIPTIVE - Work Order Recommendation (Workorder Type)

```json
{
  "id": 416,
  "text": "When an anomaly happens for equipment CWC04009, can you recommend top three work orders to address this problem?",
  "type": "Workorder",
  "category": "Decision Support",
  "deterministic": false,
  "characteristic_form": "It gives a list of work order with a primary failure code. Based on an anomaly in CWC04009, recommend the top three most appropriate work orders for remediation.",
  "group": "prescriptive",
  "entity": "Chiller",
  "note": "Source: Operations team request; Owner: Maintenance SME; Supports proactive maintenance workflow; Requires anomaly detection integration"
}
```

### Example 3: RETROSPECTIVE - Event Summary with Note

```json
{
  "id": 409,
  "text": "Get the daily count of alert, anomaly and work order event for the May 2020 for equipment CWC04009.",
  "type": "Workorder",
  "category": "Knowledge Query",
  "deterministic": false,
  "characteristic_form": "There are 26 days with the records. Depending on the LLM used, the result could be daily total event summary or daily summary for each event type. The expected response should retrieve and summarize daily counts of alerts, anomalies, and work order events for CWC04009 for May 2020, verifying correct aggregation over time (daily) and filtering of asset, event type (alert, anomaly and work order), and time range - May 2020.",
  "group": "retrospective",
  "entity": ["Alert", "Anomaly", "WorkOrder"],
  "note": "Source: Operations dashboard requirements; Owner: Analytics Team; Multi-entity aggregation query; Design note: We have both work order business data object and work order event as a group type in the event file, so we made a change to the utterance; Omitted: Specific aggregation method preference"
}
```

### Example 4: PREDICTIVE - Time Series Forecasting (TSFM Type)

```json
{
  "id": 217,
  "text": "Forecast 'Chiller 9 Condenser Water Flow' using data in 'chiller9_annotated_small_test.csv'. Use parameter 'Timestamp' as a timestamp.",
  "type": "TSFM",
  "category": "Inference Query",
  "deterministic": true,
  "characteristic_form": "The expected response should be: Forecasting results of 'Chiller 9 Condenser Water Flow' using data in 'chiller9_annotated_small_test.csv' are stored in json file",
  "group": "predictive",
  "entity": "Chiller",
  "note": "Source: Predictive maintenance initiative; Owner: Data Science Team; Requires time series forecasting model; Implementation: Uses TSFM framework with timestamp parameter"
}
```

### Example 5: Multi-Group - Performance Review & Anomaly Detection

```json
{
  "id": 420,
  "text": "Assume today is early of July 2020, I would like to review the performance of chiller 9 with equipment ID CWC04009 for June 2020 and track any anomalies or operation violations as alerts.",
  "type": "Workorder",
  "category": "Decision Support",
  "deterministic": false,
  "characteristic_form": "There were 30 alerts for 'Chiller - Evaporator Approach High', and anomalies were observed in Cooling Load (12 instances), Flow Efficiency (9), Delta Setpoint (6), and Delta Temperature (3). The LLM ReAct process is to review the operational performance and detect anomalies or alerts for Chiller 9 (CWC04009) during June 2020, confirming correct equipment ID and timeframe.",
  "group": ["retrospective", "predictive"],
  "entity": "Chiller",
  "note": "Source: Operations dashboard requirements; Owner: Analytics Team; Multi-category query combining historical review with anomaly detection; Cross-reference: Related to anomaly detection queries in category 5; Implementation: Requires ReAct agent pattern"
}
```

### Example 6: PRESCRIPTIVE - Failure Prediction Model Building

```json
{
  "id": 436,
  "text": "Build a predictive model from historical alerts and work orders of CWC04009 to forecast failures and replacement needs over the forthcoming 10-year horizon.",
  "type": "Workorder",
  "category": "Prediction",
  "deterministic": false,
  "characteristic_form": "Leverage ALERT and WORK_ORDER records (features: description, event_group, event_category, event_time) for CWC04009 to train and validate a model that predicts the likelihood and timing of future failures or required replacements.",
  "group": ["retrospective", "predictive", "prescriptive"],
  "entity": "Chiller",
  "note": "Source: Advanced analytics initiative; Owner: Data Science & Operations Teams; Complex multi-phase query spanning all three problem categories; Implementation: Requires ML pipeline with data preparation, model training, prediction, and decision support components; Estimated complexity: High; Timeline: 10-year forecast horizon"
}
```

**Note**: All examples above follow the required schema structure with 9 fields. The `group` field maps utterances to the three core problem categories (RETROSPECTIVE, PREDICTIVE, PRESCRIPTIVE), the `entity` field identifies the primary subject(s) of the utterance, and the `note` field captures essential metadata including source, owner, design rationale, and implementation details. Note that in actual JSON, group values use lowercase for simplicity.

---

## Next Steps for Completion

### Phase 1: Fill Critical Gaps (Weeks 1-2)

**Priority P0 - Safety & Operations**:
1. Add 5 utterances for emergency response scenarios
2. Add 3 utterances for critical alarm handling
3. Add 2 utterances for safety interlock verification

**Priority P1 - High Business Impact**:
1. Develop 8 energy optimization utterances (Prescriptive)
2. Create 6 maintenance scheduling utterances (Prescriptive)
3. Add 4 multi-asset coordination utterances (Prescriptive)

### Phase 2: Expand Coverage (Weeks 3-4)

**Predictive Enhancements**:
1. Add 5 RUL (Remaining Useful Life) estimation utterances
2. Create 4 degradation trend analysis utterances
3. Develop 3 seasonal performance prediction utterances

**Prescriptive Enhancements**:
1. Add 5 resource allocation optimization utterances
2. Create 4 cost-benefit analysis utterances
3. Develop 3 policy recommendation utterances

### Phase 3: Refinement (Weeks 5-6)

1. Test all utterances with actual users
2. Refine characteristic_form descriptions based on feedback
3. Validate deterministic flags with domain experts
4. Update priority levels based on usage patterns

### Phase 4: Documentation & Handoff (Week 7)

1. Document all SME contributions
2. Create usage guidelines for each utterance category
3. Establish maintenance procedures for utterance updates
4. Train new SMEs on utterance creation process

---

## Lessons Learned

### What Worked Well

1. **Incremental Development**: Starting with Retrospective queries provided a solid foundation
2. **SME Specialization**: Having dedicated SMEs for different equipment types improved quality
3. **Coverage Matrix**: Visual tracking helped identify gaps quickly
4. **Priority-Based Approach**: Focusing on P0/P1 utterances first delivered immediate value

### Challenges Encountered

1. **Prescriptive Underrepresentation**: Harder to create good prescriptive utterances - requires deep operational knowledge
2. **SME Availability**: Maintenance planning SME had limited time, slowing prescriptive development
3. **Characteristic Form Precision**: Initial attempts were too vague; required multiple iterations
4. **Cross-Equipment Scenarios**: Multi-asset utterances are complex and need careful design

### Recommendations for Other Domains

1. **Start with Retrospective**: Build knowledge and data query foundation first
2. **Engage Multiple SMEs Early**: Don't rely on single expert for entire domain
3. **Use Real Scenarios**: Base utterances on actual operational incidents and decisions
4. **Iterate on Characteristic Forms**: Expect to refine these multiple times
5. **Track Coverage Explicitly**: Use matrices to visualize gaps and progress
6. **Prioritize Ruthlessly**: Focus on high-impact scenarios before edge cases

---

## Conclusion

This Industrial Asset Management case study demonstrates the practical application of the Utterance Design Guideline in a real-world context. As a **partially accomplished** example, it shows:

- **Successful areas**: Retrospective and Predictive categories with good coverage
- **Work in progress**: Prescriptive category needs significant expansion
- **Incremental approach**: Phased development with clear next steps
- **SME collaboration**: Multiple experts contributing to their areas of expertise
- **Continuous improvement**: Ongoing refinement based on usage and feedback

This case study can serve as a template for other domains, showing both the successes and challenges of building a comprehensive utterance set incrementally.

---

## References

- [Main Utterance Design Guideline](utterance_design_guideline.md)
- Background materials: `background/all_utterance.jsonl`
- Domain abstraction patterns: `background/asset_agnostic_utterances.docx`