# Utterance Design Guideline for Domain-Agnostic Applications

## Purpose

This guideline helps domain experts and developers create well-structured, consistent utterances across diverse domains. Utterances are natural language queries or commands that users make to interact with AI-powered systems.

**Why This Guideline**: Well-designed utterances enable:
- **Consistent format**: Standardized structure makes utterances digestible and analyzable by AI agents
- **Clear categorization**: Organized by problem type (understanding what happened, predicting what will happen, helping humans make decisions)
- **Domain adaptability**: Reusable patterns that work across different contexts

**Domain-Agnostic Design**: This guideline is designed to handle various domains through a consistent abstraction pattern. The same patterns apply across:
- **Manufacturing**: Production lines, robotics, quality control systems
- **Energy & Utilities**: Power generation, transmission, distribution networks
- **Transportation**: Fleet management, logistics, rail/aviation systems
- **Oil & Gas**: Refineries, pipelines, drilling operations
- **Process Industries**: Chemical plants, food processing, pharmaceuticals
- **Building Operations**: HVAC, elevators, lighting, security systems
- **Healthcare**: Patient monitoring, diagnosis support, treatment planning
- **Financial Services**: Risk assessment, fraud detection, portfolio optimization
- **Insurance**: Claims processing, risk evaluation, policy recommendations
- **Education**: Student performance tracking, curriculum planning, resource allocation
- **Industrial Asset Management**: Equipment monitoring, predictive maintenance, lifecycle optimization

**A Good Set of Utterances for a Given Domain Should**:
- **Help understand what happened**: Enable historical data analysis, knowledge extraction, and current state assessment
- **Help predict what will happen**: Support future state forecasting, risk estimation, and trend projection
- **Help humans make decisions**: Provide recommendations, optimization, and strategic guidance

**For Example, in Industrial Asset Management Contexts, This Translates To**:

- Operational telemetry and equipment monitoring (understanding what happened)
- Predictive maintenance and failure forecasting (predicting what will happen)
- Performance optimization and decision support (helping humans make decisions)

---

## Core Problem Groups

To design effective utterances across any domain, we define three universal problem categories that span the complete lifecycle of data-driven decision-making:

### 1. retrospective (Knowledge Extraction)

**Definition**  
Understanding historical or current state through data retrieval and analysis.

**Universal Coverage**
- Entity inventory and configuration
- Historical measurements and telemetry
- Event, incident, and transaction records
- Knowledge base and documentation lookup
- Data completeness and quality assessment

**Focus Question**  
*What has happened or what is known?*

**Cross-Domain Examples**
- **Manufacturing**: "What production lines are operational?" | "List all quality defects from last shift."
- **Energy & Utilities**: "What substations are in the network?" | "Retrieve all outage events from last month."
- **Transportation**: "What vehicles are in the fleet?" | "List all maintenance records for Train-456."
- **Healthcare**: "What is the patient's medical history?" | "List all lab results from the past year."
- **Financial Services**: "What were the transaction volumes last quarter?" | "Retrieve all trades for account XYZ."

---

### 2. predictive

**Definition**  
Estimating future behavior or states under status quo conditions (no interventions).

**Universal Coverage**
- Performance and condition forecasting
- Risk and anomaly likelihood estimation
- Event probability and trend projection
- Demand and workload prediction
- Degradation and lifecycle estimation

**Focus Question**  
*What is likely to happen next?*

**Cross-Domain Examples**
- **Manufacturing**: "Predict production line downtime risk." | "Forecast product quality metrics for next batch."
- **Energy & Utilities**: "Forecast power demand for next week." | "Predict transformer failure probability."
- **Transportation**: "Predict vehicle maintenance needs." | "Estimate delivery delays for Route-A."
- **Healthcare**: "What is the patient's readmission risk?" | "Predict disease progression over the next 6 months."
- **Financial Services**: "Forecast stock price for next quarter." | "Estimate default probability for this loan."

---

### 3. prescriptive

**Definition**  
Determining optimal interventions, actions, or strategies under operational constraints.

**Universal Coverage**
- Corrective and preventive action planning
- Strategy selection and timing optimization
- Entity prioritization based on risk and impact
- Resource, budget, and capacity allocation
- Policy and rule definition
- Trade-off analysis across multiple objectives

**Focus Question**  
*What should be done and when?*

**Cross-Domain Examples**
- **Manufacturing**: "Recommend corrective actions for quality issue." | "Prioritize production orders for next shift."
- **Energy & Utilities**: "Optimize load balancing strategy." | "Recommend preventive maintenance for grid equipment."
- **Transportation**: "Optimize route assignments." | "Recommend vehicle replacement strategy."
- **Healthcare**: "Recommend treatment plan for this diagnosis." | "Prioritize patients for ICU admission."
- **Financial Services**: "Suggest portfolio rebalancing strategy." | "Recommend fraud prevention actions."

---

## Domain Abstraction Pattern

To create domain-agnostic utterances that can be adapted across applications, use a consistent placeholder system:

### Core Placeholders

| Placeholder | Description | Examples |
|------------|-------------|----------|
| `{ENTITY}` | Primary object of analysis | Patient, Account, Equipment, Vehicle, Student |
| `{ENTITY_CLASS}` | Category or type | Diagnosis, Portfolio, HVAC, Fleet, Course |
| `{LOCATION}` | Physical or logical location | Hospital, Branch, Site, Route, Campus |
| `{METRIC}` | Measurement or KPI | Heart Rate, Balance, Temperature, Speed, Grade |
| `{EVENT_TYPE}` | Type of occurrence | Admission, Transaction, Alert, Delay, Enrollment |
| `{EVENT_NAME}` | Specific event label | Readmission, Withdrawal, Failure, Accident, Dropout |
| `{TIME_RANGE}` | Temporal scope | "Last month", "2020-06-01 to 2020-06-30", "Next quarter" |
| `{WINDOW}` | Time horizon | "Within 14 days", "Next maintenance window", "By end of semester" |
| `{ACTION}` | Intervention or decision | Treatment, Investment, Maintenance, Reroute, Tutoring |
| `{CONDITION}` | Trigger or state | "When anomaly detected", "If balance < threshold", "When grade drops" |
| `{OUTPUT_FORM}` | Response format | List, Table, Chart, File, Summary |

### Usage Example

**Generic Template**:  
"Retrieve {METRIC} for {ENTITY} at {LOCATION} over {TIME_RANGE}"

**Domain Adaptations**:
- Healthcare: "Retrieve blood pressure for Patient-123 at Boston General over last month"
- Finance: "Retrieve transaction volume for Account-456 at NYC Branch over Q1 2024"
- Manufacturing: "Retrieve cycle time for Robot-5 at Assembly Line A over last week"
- Education: "Retrieve attendance for Student-789 at Main Campus over last semester"

---

## Utterance Schema Structure

Each utterance should be documented with the following attributes in this order:

### Required Fields

1. **id** (integer): Unique identifier for the utterance
2. **text** (string): The actual utterance text in natural language
3. **type** (string): The MCP server, AI agent, service, or system component that processes this utterance
   - In MCP (Model Context Protocol) systems: Identifies which MCP server handles the request (e.g., "filesystem", "database", "web-search")
   - In multi-agent systems: Identifies which agent handles the request (e.g., "IoT", "FMSA", "TSFM", "Workorder", "multiagent")
   - In single-agent systems: Can represent the domain or service type (e.g., "Healthcare", "Finance", "CustomerService")
   - In microservices: Can indicate the service endpoint (e.g., "PatientAPI", "BillingService", "InventoryService")
4. **category** (string): Classification of the utterance type (see Category Definitions section)
5. **deterministic** (boolean): Whether the utterance has a single correct answer (true) or allows multiple valid responses (false)
6. **characteristic_form** (string): Description of the expected response format and content
7. **group** (string or array): Problem category - RETROSPECTIVE, PREDICTIVE, PRESCRIPTIVE, or combination (e.g., ["retrospective", "predictive"])
8. **entity** (string or array): Primary entity/entities the utterance addresses (e.g., "Chiller", "Patient", "Account", "Vehicle")
9. **note** (string): A flexible field for any information useful to explain or contextualize the utterance, including source, owner/contributor, omitted details, design rationale, implementation notes, or any other relevant metadata

---

### Field Descriptions

**type**: Specifies which MCP server, AI agent, service, or system component processes this utterance:
- **MCP (Model Context Protocol) systems**: MCP server identifier (e.g., "filesystem" for file operations, "database" for data queries, "web-search" for internet searches, "git" for version control operations)
- **Multi-agent systems**: Agent identifier (e.g., "IoT" for sensor data, "FMSA" for failure analysis, "TSFM" for time series forecasting, "Workorder" for maintenance management)
- **Single-agent systems**: Domain or service type (e.g., "Healthcare" for medical queries, "Finance" for financial operations)
- **Microservices architecture**: Service endpoint or API name (e.g., "PatientAPI", "BillingService")
- **Purpose**: Enables routing, load balancing, and proper request handling in distributed systems

**group**: Indicates which of the three core problem categories the utterance belongs to:
- **RETROSPECTIVE**: Understanding what happened or what is known
- **PREDICTIVE**: Estimating what will happen next
- **PRESCRIPTIVE**: Determining what should be done and when
- Can be a single value or array if utterance spans multiple categories. Note: Use lowercase values in actual JSON (e.g., "retrospective", "predictive", "prescriptive")

**entity**: Specifies the primary physical subject(s) of the utterance - the tangible things being analyzed, monitored, or acted upon:
- In industrial domains: Physical equipment (e.g., "Chiller", "Pump", "Transformer", "Sensor")
- In healthcare: Physical entities (e.g., "Patient", "MedicalDevice", "Facility")
- In finance: Physical or tangible assets (e.g., "Branch", "ATM", "DataCenter")
- In transportation: Physical assets (e.g., "Vehicle", "Aircraft", "Vessel")
- Can be a single value or array for multi-entity queries
- Note: Use physical things, not abstract concepts (e.g., use "Chiller" not "Anomaly", use "Equipment" not "WorkOrder")

**note**: A flexible, required field for storing any information useful to explain or contextualize the utterance. Common uses include:
- **Source**: Where the utterance originated (e.g., "SME Interview - John Doe", "Customer Request #1234", "Internal Analysis")
- **Owner/Contributor**: Who provided or is responsible for this utterance
- **Omitted Details**: Information intentionally left out for brevity or abstraction
- **Design Rationale**: Why this utterance was structured this way
- **Implementation Notes**: Technical considerations or constraints
- **Cross-references**: Related utterances or dependencies
- **Domain-Specific Context**: Any additional information that aids understanding, maintenance, or usage

The field is intentionally flexible to accommodate diverse documentation needs across different domains and use cases.

---

### Examples

**Basic Example**:
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
  "note": "Source: Initial domain analysis; Owner: Domain SME Team; Basic inventory query pattern"
}
```

**Predictive Example**:
```json
{
  "id": 416,
  "text": "When an anomaly happens for equipment CWC04009, can you recommend top three work orders?",
  "type": "Workorder",
  "category": "Decision Support",
  "deterministic": false,
  "characteristic_form": "It gives a list of work order with a primary failure code.",
  "group": "prescriptive",
  "entity": "Chiller",
  "note": "Source: Operations team request; Owner: Maintenance SME; Supports proactive maintenance workflow"
}
```

**Multi-Group Example**:
```json
{
  "id": 420,
  "text": "Review the performance of chiller 9 for June 2020 and track any anomalies or operation violations as alerts.",
  "type": "Workorder",
  "category": "Decision Support",
  "deterministic": false,
  "characteristic_form": "Review operational performance and detect anomalies or alerts for Chiller 9 during June 2020.",
  "group": ["retrospective", "predictive"],
  "entity": "Chiller",
  "note": "Source: Operations dashboard requirements; Owner: Analytics Team; Multi-category query combining historical review with anomaly detection"
}
```

**Example with Note**:
```json
{
  "id": 409,
  "text": "Get the daily count of alert, anomaly and work order event for May 2020.",
  "type": "Workorder",
  "category": "Knowledge Query",
  "deterministic": false,
  "characteristic_form": "There are 26 days with records. Depending on the LLM used, the result could be daily total event summary or daily summary for each event type.",
  "group": "retrospective",
  "entity": ["Alert", "Anomaly", "WorkOrder"],
  "note": "We have both work order business data object and work order event as a group type in the event file."
}
```

---

## Category Definitions (Suggested)

The following categories are **suggested classifications** that have proven useful across multiple domains. You may adapt, combine, or create new categories based on your specific domain needs.

### Information Retrieval
Requests for accessing existing information from knowledge bases or databases.
- **Characteristics**: Direct lookup, factual responses, deterministic answers
- **Generic Pattern**: "What is...", "List all...", "Show me...", "Retrieve..."
- **Cross-Domain Examples**:
  - Healthcare: "What is the patient's blood type?"
  - Finance: "List all accounts for customer ID 12345"
  - Education: "Show me all courses in the Computer Science department"
  - Manufacturing: "List all equipment in Building A"

### Data Extraction
Requests for retrieving specific data points, measurements, or records over time.
- **Characteristics**: Time-bound queries, data export, file generation
- **Generic Pattern**: "Download...", "Export...", "Get data for...", "Retrieve records from..."
- **Cross-Domain Examples**:
  - Healthcare: "Download patient vital signs from last week"
  - Finance: "Export transaction history for Q1 2024"
  - Education: "Get attendance records for Student-123 from last semester"
  - Manufacturing: "Retrieve production metrics from last shift"

### Analysis & Inference
Requests for AI-powered analysis, predictions, or pattern recognition on provided data.
- **Characteristics**: Model execution, computational analysis, insight generation
- **Generic Pattern**: "Analyze...", "Predict...", "Forecast...", "Identify patterns in..."
- **Cross-Domain Examples**:
  - Healthcare: "Predict readmission risk using patient data"
  - Finance: "Analyze portfolio performance and identify trends"
  - Education: "Predict student dropout risk based on engagement data"
  - Manufacturing: "Forecast production output for next week"

### Model Customization
Requests for adapting or fine-tuning AI models to specific contexts or data.
- **Characteristics**: Model training, parameter optimization, domain adaptation
- **Generic Pattern**: "Fine-tune...", "Train model on...", "Customize...", "Adapt model for..."
- **Cross-Domain Examples**:
  - Healthcare: "Fine-tune diagnosis model using hospital-specific data"
  - Finance: "Train fraud detection model on our transaction patterns"
  - Education: "Customize recommendation engine for our curriculum"
  - Manufacturing: "Adapt quality prediction model to our production line"

### Anomaly & Exception Detection
Requests for identifying unusual patterns, outliers, or deviations from normal behavior.
- **Characteristics**: Pattern deviation detection, threshold monitoring, outlier identification
- **Generic Pattern**: "Detect anomalies in...", "Identify unusual...", "Find outliers in...", "Are there any exceptions in..."
- **Cross-Domain Examples**:
  - Healthcare: "Detect anomalies in patient vital signs"
  - Finance: "Identify unusual transaction patterns"
  - Education: "Find students with abnormal attendance patterns"
  - Manufacturing: "Detect quality anomalies in production batch"

### Recommendation & Optimization
Requests for actionable recommendations, strategic guidance, or optimization suggestions.
- **Characteristics**: Multi-factor analysis, trade-off evaluation, decision support
- **Generic Pattern**: "Recommend...", "Suggest...", "What should I do when...", "Optimize..."
- **Cross-Domain Examples**:
  - Healthcare: "Recommend treatment plan for this diagnosis"
  - Finance: "Suggest portfolio rebalancing strategy"
  - Education: "Recommend courses for student based on interests and performance"
  - Manufacturing: "Suggest process improvements to reduce defects"

### Future State Prediction
Requests for forecasting future states, events, or probabilities.
- **Characteristics**: Probabilistic outputs, time-horizon projections, scenario analysis
- **Generic Pattern**: "What will happen...", "Forecast...", "Estimate probability of...", "Predict future..."
- **Cross-Domain Examples**:
  - Healthcare: "Predict patient deterioration risk over next 48 hours"
  - Finance: "Forecast revenue for next quarter"
  - Education: "Predict enrollment numbers for next academic year"
  - Manufacturing: "Estimate equipment failure probability over next month"

### Multi-Step Orchestration
Complex requests requiring coordination of multiple operations in sequence.
- **Characteristics**: Multiple tool invocations, sequential processing, workflow execution
- **Generic Pattern**: "First... then... finally...", "Retrieve X, analyze Y, and recommend Z"
- **Cross-Domain Examples**:
  - Healthcare: "Retrieve patient history, analyze trends, and recommend interventions"
  - Finance: "Analyze market data, forecast returns, and suggest investment strategy"
  - Education: "Review student performance, identify at-risk students, and recommend support programs"
  - Manufacturing: "Collect quality data, detect issues, and recommend corrective actions"

**Note**: These categories are not mutually exclusive. Some utterances may span multiple categories. The key is to choose the primary category that best describes the utterance's main intent. For domain-specific category examples, see the case studies below.

---

## Deterministic vs. Non-Deterministic Utterances

### Deterministic (deterministic: true)
- Has a **single, verifiable correct answer**
- Response can be validated against ground truth
- Typically involves data retrieval or well-defined calculations

### Non-Deterministic (deterministic: false)
- May have **multiple valid answers** or interpretations
- Response depends on model reasoning, domain knowledge, or heuristics
- Typically involves recommendations, predictions, or complex analysis

---

## Characteristic Form Guidelines

The `characteristic_form` field describes the expected response structure and validation criteria.

### For Deterministic Utterances

1. **Exact Expected Output**: Specify the precise data or format
2. **Validation Criteria**: How to verify correctness
3. **Data Format**: Specify file types, data structures

### For Non-Deterministic Utterances

1. **Response Constraints**: What must be included
2. **Quality Criteria**: What makes a good response
3. **Acceptable Variations**: Note when multiple approaches are valid

### General Guidelines

- **Be Specific**: Clearly define what constitutes a correct or acceptable response
- **Include Context**: Mention relevant constraints (time ranges, entity IDs, data sources)
- **Specify Format**: Indicate if response should be text, file reference, JSON, list, etc.
- **Note Dependencies**: Mention if response depends on available data or models
- **Validation Steps**: Describe how to verify the response meets requirements

---

## Domain-Agnostic Utterance Templates

### Template 1: Historical Data Retrieval
```
"Retrieve {METRIC} for {ENTITY} at {LOCATION} from {TIME_RANGE}"
```

### Template 2: Knowledge Lookup
```
"List all {ATTRIBUTE} of {ENTITY_CLASS}"
```

### Template 3: Predictive Analysis
```
"Forecast {METRIC} for {ENTITY} for {FUTURE_RANGE} based on data from {PAST_RANGE}"
```

### Template 4: Anomaly Detection
```
"Is there any anomaly detected in {ENTITY}'s {METRIC} in {TIME_RANGE} at {LOCATION}?"
```

### Template 5: Recommendation
```
"When {CONDITION} happens for {ENTITY}, recommend {ACTION_TYPE}"
```

### Template 6: Multi-Factor Decision Support
```
"Review {DATA_SOURCES} for {ENTITY} during {PERIOD} and {recommend/analyze/prioritize} {ACTION}"
```

---

## Utterance Set Completeness

### Definition
**Completeness** refers to the degree to which a set of utterances covers the full scope of requirements for a specific domain. A complete utterance set should address all critical operational scenarios, decision points, and information needs within the domain.

### Coverage Dimensions

**Operational Lifecycle Coverage**
- [ ] Normal operations monitoring
- [ ] Degradation detection and tracking
- [ ] Event prediction and diagnosis
- [ ] Action planning and execution
- [ ] Performance optimization
- [ ] Compliance and reporting

**Problem Category Balance**
- [ ] Retrospective queries (historical analysis, knowledge lookup)
- [ ] Predictive queries (forecasting, risk assessment)
- [ ] Prescriptive queries (recommendations, optimization)

**Entity/System Coverage**
- [ ] All critical entity types in the domain
- [ ] Key subsystems and components
- [ ] Cross-entity dependencies and interactions

**Stakeholder Perspective Coverage**
- [ ] Operations team needs
- [ ] Management team needs
- [ ] Engineering/Analysis team needs
- [ ] End-user needs

### Achieving Completeness

To ensure comprehensive coverage, consider mapping domain requirements to the three core categories (Retrospective, Predictive, Prescriptive) and identifying critical operational scenarios. Create a coverage matrix to highlight gaps, then iteratively refine the utterance set by starting with high-priority scenarios, testing with domain experts, and expanding based on actual usage patterns and user feedback.

---

## Utterance Specificity and Incremental Development

### Collaborative Development with Subject Matter Experts

**Subject Matter Experts (SMEs)** are domain specialists with deep knowledge of specific equipment, processes, or operational contexts. Rather than requiring one SME to create all utterances at once, the guideline supports **incremental, collaborative development**.

**Suggested Approach**:
- **Divide by expertise**: Different SMEs can contribute utterances for their specific areas (equipment types, operational functions, or problem categories)
- **Start small**: Begin with 5-10 critical, high-frequency queries per SME
- **Iterate and expand**: Add edge cases, test with users, refine based on feedback, and continuously improve
- **Vary specificity**: Use generic templates for reusability, domain-specific utterances for tailored needs, and scenario-specific queries for deep expertise

**Prioritization**: Focus on critical operations, high-frequency needs, and high-impact decisions first. Quality and relevance matter more than quantity.
- **Data Availability**: Create utterances for scenarios where data actually exists

---

## Validation Checklist

Before finalizing an utterance, verify:

- [ ] **Clarity**: Is the request unambiguous?
- [ ] **Completeness**: Are all necessary parameters specified?
- [ ] **Category**: Is the category correctly assigned?
- [ ] **Deterministic Flag**: Is the deterministic field accurately set?
- [ ] **Characteristic Form**: Does it clearly describe expected output?
- [ ] **Domain Relevance**: Does it align with real-world use cases?
- [ ] **Testability**: Can the response be validated or evaluated?
- [ ] **Consistency**: Does it follow established patterns and terminology?
- [ ] **Domain-Neutrality**: Can it be adapted to other domains using placeholders?
- [ ] **SME Validation**: Has a domain expert reviewed and approved this utterance?
- [ ] **Coverage Contribution**: Does this utterance fill a gap in the coverage matrix?

---

## Summary

Effective domain-agnostic utterance design requires:

### Framework Fundamentals
1. **Universal categorization** into Retrospective, Predictive, or Prescriptive
2. **Consistent abstraction** using domain-agnostic placeholders
3. **Proper classification** by query type
4. **Accurate deterministic flag** based on answer uniqueness
5. **Detailed characteristic form** describing expected responses

### Completeness and Coverage
6. **Comprehensive coverage** across operational lifecycle and stakeholder needs
7. **Balanced distribution** across problem categories
8. **Gap analysis** using coverage matrices
9. **Iterative refinement** based on feedback

### Collaborative Development
10. **SME-driven creation** leveraging domain experts
11. **Incremental development** allowing phased rollout
12. **Distributed ownership** with clear responsibilities
13. **Prioritization** focusing on critical scenarios
14. **Quality over quantity** ensuring practical utility

### Practical Implementation
15. **Domain-specific adaptation** while maintaining consistency
16. **Validation criteria** ensuring accuracy
17. **Documentation standards** including ownership and dependencies
18. **Continuous improvement** through monitoring and feedback

---

## Getting Started

**For New Domains**:
1. Identify 3-5 critical entity types or operational scenarios
2. Engage relevant SMEs for each area
3. Create 5-10 core utterances per SME covering high-priority needs
4. Build coverage matrix to track completeness
5. Iterate and expand based on user feedback

**For Existing Systems**:
1. Analyze current query patterns and user needs
2. Map existing queries to the three core categories
3. Identify gaps using the completeness assessment methods
4. Prioritize gap-filling based on business impact
5. Engage SMEs to create missing utterances incrementally

---

## Case Studies

For detailed examples of applying this guideline to specific domains, see:

### 1. Industrial Asset Management Case Study
**File**: [case_study_industrial_asset_management.md](case_study_industrial_asset_management.md)
**Domain**: Building automation and HVAC systems
**Focus**: Chiller and AHU equipment monitoring, predictive maintenance
**Dataset**: 152 utterances evolved incrementally with SME input
**Status**: Comprehensive production example showing real-world development process

**Key Highlights**:
- Demonstrates incremental development with domain experts
- Shows evolution from initial requirements to comprehensive coverage
- Heavily weighted toward retrospective queries (77/152) reflecting actual usage patterns
- Includes detailed coverage analysis and SME contribution tracking

### 2. Wind Turbine Operations Case Study
**File**: [case_study_wind_turbine.md](case_study_wind_turbine.md)
**Domain**: Renewable energy and wind power generation
**Focus**: Wind turbine monitoring, power forecasting, maintenance optimization
**Dataset**: 30 utterances designed complete from the start
**Status**: Complete demonstration example showing balanced framework application

**Key Highlights**:
- Demonstrates clean-slate application of the guideline to a new domain
- Perfectly balanced distribution (10 retrospective, 10 predictive, 10 prescriptive)
- Includes rigorous completeness framework with validation criteria
- Shows how to create a complete case study efficiently using the guideline

**Comparison**: The two case studies complement each other - Industrial Asset Management shows organic growth in a real project, while Wind Turbine demonstrates planned, balanced design from inception. Both validate the guideline's cross-domain applicability.
