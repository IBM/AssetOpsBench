# Extended Evaluation Framework: 8-Dimensional Scoring for Graph-Capable Agents

## Motivation

AssetOpsBench's original 6-dimensional evaluation criteria (Task Completion, Data Retrieval Accuracy, Generalized Result Verification, Agent Sequence Correct, Clarity & Justification, Hallucinations) are designed for single-agent and multi-agent scenarios operating over flat document stores. However, 40 new graph-native scenarios require evaluating capabilities that don't exist in the original framework:

- **Graph traversal quality**: Did the agent exploit relationship structure rather than reasoning over text?
- **Semantic matching precision**: Did the agent use vector similarity search effectively?

We propose 2 additional dimensions that complement (not replace) the original 6.

## Extended Dimensions

### Original 6 (Unchanged)

| # | Dimension | Description |
|---|---|---|
| 1 | **Task Completion** | Did the agent complete the requested task? |
| 2 | **Data Retrieval Accuracy** | Did the agent retrieve correct data from correct sources? |
| 3 | **Generalized Result Verification** | Are the results correct beyond just format? |
| 4 | **Agent Sequence Correct** | Did the agent follow the correct execution path? |
| 5 | **Clarity & Justification** | Is the response clear and well-justified? |
| 6 | **Hallucinations** | Did the agent make up false information? |

### New Dimensions (for Graph scenarios only)

| # | Dimension | Description |
|---|---|---|
| 7 | **Graph Utilization** | Did the agent leverage graph structure (edge traversal, path queries, graph algorithms) rather than text-based reasoning? Measures whether the agent recognized and exploited relationship topology. |
| 8 | **Semantic Precision** | For scenarios requiring similarity search: did the agent use vector embeddings effectively? Measures quality of semantic matching beyond keyword overlap. |

## When to Apply

- **Dimensions 1-6**: All 179 scenarios (original 139 + new 40)
- **Dimension 7 (Graph Utilization)**: New 40 graph scenarios only (IDs 601-640)
- **Dimension 8 (Semantic Precision)**: 6 failure similarity scenarios only (IDs 615-620)

## 40 New Scenarios

### Categories

| Category | IDs | Count | Graph Capabilities Tested |
|---|---|---|---|
| Multi-hop dependency | 601-608 | 8 | BFS/DFS over DEPENDS_ON, SHARES_SYSTEM_WITH edges |
| Cross-asset correlation | 609-614 | 6 | Anomaly correlation across connected equipment |
| Failure pattern similarity | 615-620 | 6 | Vector similarity search on FailureMode embeddings |
| Criticality analysis | 621-625 | 5 | PageRank, WCC, articulation point detection |
| Maintenance optimization | 626-630 | 5 | Constrained scheduling, Pareto optimization |
| Root cause analysis | 631-635 | 5 | Reverse edge traversal (TRIGGERED, DETECTED_ANOMALY) |
| Temporal pattern | 636-640 | 5 | Temporal aggregation over work order sequences |

### Graph Schema Required

The 40 scenarios assume a knowledge graph with the following structure built from AssetOpsBench data:

**Node labels**: Site, Location, Equipment, Sensor, FailureMode, WorkOrder, SparePart, Anomaly, Event

**Edge types**:
- `CONTAINS_LOCATION` (Site → Location)
- `CONTAINS_EQUIPMENT` (Location → Equipment)
- `HAS_SENSOR` (Equipment → Sensor)
- `DEPENDS_ON` (Equipment → Equipment)
- `SHARES_SYSTEM_WITH` (Equipment ↔ Equipment)
- `MONITORS` (Sensor → FailureMode)
- `EXPERIENCED` (Equipment → FailureMode)
- `FOR_EQUIPMENT` (WorkOrder/Event → Equipment)
- `DETECTED_ANOMALY` (Sensor → Anomaly)
- `TRIGGERED` (Anomaly → WorkOrder)
- `ADDRESSES` (WorkOrder → FailureMode)
- `USES_PART` (WorkOrder → SparePart)

### Design Principles

1. **Tool-agnostic**: Any graph-capable agent can attempt these scenarios. No Samyama-specific tools or APIs are assumed.
2. **Data-grounded**: All scenarios reference equipment, sensors, and failure modes present in the AssetOpsBench dataset.
3. **Non-deterministic**: Most scenarios accept multiple valid answers. Evaluation uses characteristic_form acceptance criteria.
4. **Complementary**: These scenarios test capabilities that IBM's original 139 scenarios do not cover (graph traversal, vector search, graph algorithms, multi-objective optimization).

## Scoring

For graph scenarios, the overall pass criterion is:

- Dimensions 1-6: Same as original (all must pass, hallucinations must be false)
- Dimension 7: Pass if the agent used graph traversal/algorithms rather than purely text-based reasoning
- Dimension 8: Pass if vector similarity scores are returned with reasonable rankings (failure similarity scenarios only)

The 8-dimensional score provides a richer signal for evaluating agent capabilities, especially for distinguishing between agents that reason over text versus agents that exploit graph structure.
