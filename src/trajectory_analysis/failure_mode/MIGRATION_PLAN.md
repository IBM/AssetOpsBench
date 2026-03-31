# TrajFM Migration Plan

## Overview

Migration of trajectory analysis and failure mode detection from `src/tmp/TrajFM/` to `src/trajectory_analysis/failure_mode/` with integration into AssetOpsBench's existing `src/llm` infrastructure.

---

## Phase 1: Structure Setup & Core Files ✅ COMPLETE

**Goal**: Create new directory structure and migrate core pipeline files

**Status**: ✅ Completed

### What Was Created

```
src/trajectory_analysis/
├── __init__.py                    # Module entry point
└── failure_mode/
    ├── __init__.py                # Submodule with documentation
    ├── MIGRATION_PLAN.md          # This file - complete migration plan
    ├── pipeline.py                # High-level pipeline API
    ├── prompts.py                 # LLM system prompts
    ├── README.md                  # Quick start guide
    ├── README_detail.md           # Detailed documentation
    ├── processed_trajectories/    # Data storage directory
    └── tests/
        ├── __init__.py            # Test module marker
        ├── conftest.py            # Test fixtures with secure env loading
        └── README.md              # Test setup guide

4 directories, 10 files
```

### Completed Tasks

- ✅ Created directory structure at `src/trajectory_analysis/failure_mode/`
- ✅ Copied core files: `pipeline.py`, `prompts.py`
- ✅ Migrated documentation: `README.md`, `README_detail.md`
- ✅ Created all `__init__.py` files with proper docstrings
- ✅ Created data directories: `processed_trajectories/`, `tests/`
- ✅ Created `MIGRATION_PLAN.md` with complete 6-phase plan
- ✅ Created test infrastructure: `conftest.py` with LLM fixtures
- ✅ Created `tests/README.md` with setup instructions
- ✅ Configured secure environment variable loading (project root `.env`)
- ✅ Verified structure with `tree` command

### Key Design Decisions

1. **Location**: `src/trajectory_analysis/` (consistent with existing `src/` structure)
2. **Fixed typo**: `trajectory_analysis` (not `trajectroy_analysis`)
3. **Renamed files**: `prompt.py` → `prompts.py` (more descriptive)
4. **Preserved structure**: Kept `processed_trajectories/` and `tests/` subdirectories
5. **Security**: Environment variables loaded from project root `.env` (not in module)
6. **LLM Support**: Configured for Claude 4 Sonnet (default), Llama 3.3 70B, and Granite
7. **Test fixtures**: Centralized LLM backend configuration in `conftest.py`

### Environment Configuration

**Location**: `.env` file at **project root** (`/path/to/AssetOpsBench/.env`)

**Required variables**:
```bash
# For Claude 4 Sonnet (via LiteLLM proxy)
LITELLM_API_KEY=your-litellm-api-key
LITELLM_BASE_URL=https://ete-litellm.ai-models.vpc-int.res.ibm.com

# For Llama 3.3 70B and Granite (via WatsonX)
WATSONX_APIKEY=your-watsonx-api-key
WATSONX_PROJECT_ID=c4bfae5a-377f-44d6-b37a-68435a056744
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

⚠️ **Security Note**: `.env` file is in `.gitignore` - never commit API keys!

---

## Phase 2: LLM Integration Refactoring 🔄 NEXT

**Goal**: Replace ReactXen dependency with `src/llm`

**Duration**: Week 2

### LLM Model Selection

**Default Model**: `litellm_proxy/GCP/claude-4-sonnet` (recommended for best accuracy)

**Supported Models**:

| Model | Provider | Use Case | Setup |
|-------|----------|----------|-------|
| **Claude 4 Sonnet** ⭐ | Anthropic/GCP | Best accuracy, superior reasoning | `LITELLM_API_KEY`, `LITELLM_BASE_URL` |
| **Llama 3.3 70B** | WatsonX | Cost-effective, good performance | `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` |
| **Granite** | IBM | Lowest cost, IBM native | `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` |

**Example Usage**:
```python
from src.llm.litellm import LiteLLMBackend

# Default: Claude 4 Sonnet (best accuracy)
llm = LiteLLMBackend("litellm_proxy/GCP/claude-4-sonnet")

# Alternative: Llama 3.3 70B (cost-effective)
llm = LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct")

# Alternative: Granite (lowest cost)
llm = LiteLLMBackend("watsonx/ibm/granite-13b-instruct-v2")
```

### Tasks

1. **Refactor `utils.py`**
   - Remove: `from reactxen.utils.model_inference import watsonx_llm`
   - Add: `from src.llm.base import LLMBackend`
   - Update `get_llm_answer_from_json()` signature:
     ```python
     # OLD
     def get_llm_answer_from_json(data: dict, model_id: int) -> str:
         ans = watsonx_llm(prompt=prompt, model_id=model_id)
     
     # NEW
     def get_llm_answer_from_json(
         data: dict,
         llm_backend: LLMBackend,
         temperature: float = 0.0
     ) -> str:
         ans = llm_backend.generate(prompt=prompt, temperature=temperature)
     ```
   - Copy refactored file to `src/trajectory_analysis/failure_mode/utils.py`

2. **Update `failure_mode_generator.py`**
   - Modify `process_trajectories()` to accept `llm_backend: LLMBackend` parameter
   - Set default to Claude 4 Sonnet if not provided
   - Update internal calls to use `llm_backend` instead of `model_id`
   - Copy to `src/trajectory_analysis/failure_mode/generator.py`

3. **Update `pipeline.py`**
   - Add `llm_backend` parameter with Claude 4 Sonnet as default
   - Example:
     ```python
     def run_failure_mode_pipeline(
         traj_root_base: str,
         llm_backend: LLMBackend = None,  # Default to Claude 4 Sonnet
         timestamps=None,
         summary_dir: str = "summary",
         model_name: str = "all-MiniLM-L6-v2",
         k: int | None = None,
     ):
         if llm_backend is None:
             from src.llm.litellm import LiteLLMBackend
             llm_backend = LiteLLMBackend("litellm_proxy/GCP/claude-4-sonnet")
     ```

4. **Test with multiple models**
   - Create test script comparing Claude 4 Sonnet vs Llama 3.3 70B
   - Verify LLM integration works correctly with both models
   - Document performance differences

### Files to Migrate

- `src/tmp/TrajFM/utils.py` → `src/trajectory_analysis/failure_mode/utils.py` (refactored)
- `src/tmp/TrajFM/failure_mode_generator.py` → `src/trajectory_analysis/failure_mode/generator.py` (updated)
- `src/trajectory_analysis/failure_mode/pipeline.py` (update with default LLM)

### Environment Setup

**For Claude 4 Sonnet** (default):
```bash
export LITELLM_API_KEY="your-api-key"
export LITELLM_BASE_URL="https://your-litellm-proxy-url"
```

**For WatsonX models** (Llama/Granite):
```bash
export WATSONX_APIKEY="your-api-key"
export WATSONX_PROJECT_ID="your-project-id"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"  # optional
```

### Deliverable

Core files work with `src/llm`, tested with Claude 4 Sonnet (default) and Llama 3.3 70B (alternative)

---

## Phase 3: Pipeline Components Migration 🔜

**Goal**: Migrate remaining pipeline components

**Duration**: Week 3

### Tasks

1. **Copy and refactor remaining files**
   - `failure_mode_extractor.py` → `extractor.py`
   - `failure_mode_reduction.py` → `reducer.py`
   - `plot_failure_mode.py` → `visualizer.py`

2. **Update all internal imports**
   - Change: `from failure_mode_generator import ...`
   - To: `from src.trajectory_analysis.failure_mode.generator import ...`

3. **Update `pipeline.py`**
   - Update imports to use new module structure
   - Ensure all components work together

### Files to Migrate

- `src/tmp/TrajFM/failure_mode_extractor.py` → `extractor.py`
- `src/tmp/TrajFM/failure_mode_reduction.py` → `reducer.py`
- `src/tmp/TrajFM/plot_failure_mode.py` → `visualizer.py`

### Deliverable

All core components migrated and imports updated

---

## Phase 4: Testing & Documentation 🔜

**Goal**: Migrate tests and update documentation

**Duration**: Week 4

### Tasks

1. **Migrate test files**
   - `failure_mode_generator_test.py` → `tests/test_generator.py`
   - `failure_mode_reduction_test.py` → `tests/test_reducer.py`
   - Update test imports and paths

2. **Update `__init__.py` files**
   ```python
   # src/trajectory_analysis/__init__.py
   from .failure_mode import pipeline
   
   # src/trajectory_analysis/failure_mode/__init__.py
   from .pipeline import run_failure_mode_pipeline
   from .generator import process_trajectories
   from .reducer import failure_mode_reduction
   ```

3. **Update documentation**
   - Update import examples in README files
   - Add migration guide
   - Document LLM backend usage

### Deliverable

Tests pass, documentation updated

---

## Phase 5: Integration & Validation 🔜

**Goal**: Integrate with existing AssetOpsBench workflows

**Duration**: Week 5

### Tasks

1. **Create example scripts**
   
   **Example 1: Using Claude 4 Sonnet (Default - Best Accuracy)**
   ```python
   from src.llm.litellm import LiteLLMBackend
   from src.trajectory_analysis.failure_mode import run_failure_mode_pipeline
   
   # Option 1: Use default (Claude 4 Sonnet)
   results = run_failure_mode_pipeline(
       traj_root_base="/path/to/trajectories",
       summary_dir="summary_codabench"
   )
   
   # Option 2: Explicitly specify Claude 4 Sonnet
   llm = LiteLLMBackend("litellm_proxy/GCP/claude-4-sonnet")
   results = run_failure_mode_pipeline(
       traj_root_base="/path/to/trajectories",
       llm_backend=llm,
       summary_dir="summary_codabench"
   )
   ```
   
   **Example 2: Using Llama 3.3 70B (Cost-Effective)**
   ```python
   from src.llm.litellm import LiteLLMBackend
   from src.trajectory_analysis.failure_mode import run_failure_mode_pipeline
   
   llm = LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct")
   results = run_failure_mode_pipeline(
       traj_root_base="/path/to/trajectories",
       llm_backend=llm,
       summary_dir="summary_codabench"
   )
   ```
   
   **Example 3: Using Granite (Lowest Cost)**
   ```python
   from src.llm.litellm import LiteLLMBackend
   from src.trajectory_analysis.failure_mode import run_failure_mode_pipeline
   
   llm = LiteLLMBackend("watsonx/ibm/granite-13b-instruct-v2")
   results = run_failure_mode_pipeline(
       traj_root_base="/path/to/trajectories",
       llm_backend=llm,
       summary_dir="summary_codabench"
   )
   ```

2. **Test with real trajectory data**
   - Validate against existing test cases with all three models
   - Compare accuracy: Claude 4 Sonnet vs Llama 3.3 70B vs Granite
   - Ensure output format matches expectations
   - Document performance/cost trade-offs

3. **Update existing scripts/notebooks**
   - Update any code that uses TrajFM
   - Add model selection examples
   - Document recommended model for different use cases

### Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Production/Critical Analysis | Claude 4 Sonnet | Best accuracy, superior reasoning |
| Development/Testing | Llama 3.3 70B | Good balance of cost and performance |
| Batch Processing/Large Scale | Granite | Lowest cost, acceptable for bulk analysis |
| Research/Experimentation | All three | Compare results across models |

### Deliverable

Working integration examples with all three supported models, performance comparison documentation

---

## Phase 6: Cleanup & Deprecation 🔜

**Goal**: Remove old code and finalize migration

**Duration**: Week 6

### Tasks

1. **Add deprecation warnings to `src/tmp/TrajFM/`**
   ```python
   import warnings
   warnings.warn(
       "This module has moved to src.trajectory_analysis.failure_mode. "
       "Please update your imports.",
       DeprecationWarning
   )
   ```

2. **Update project documentation**
   - Update main README.md
   - Add migration notes

3. **Create migration guide for external users**

4. **Remove `src/tmp/TrajFM/` after 1-2 release cycles**

### Deliverable

Clean codebase, migration complete

---

## Timeline Summary

| Phase | Duration | Status | Key Milestone |
|-------|----------|--------|---------------|
| 1 | Week 1 | ✅ Complete | Structure created |
| 2 | Week 2 | 🔄 Next | LLM integration working |
| 3 | Week 3 | 🔜 Pending | All components migrated |
| 4 | Week 4 | 🔜 Pending | Tests & docs updated |
| 5 | Week 5 | 🔜 Pending | Integration validated |
| 6 | Week 6 | 🔜 Pending | Cleanup complete |

**Total**: ~6 weeks for complete migration

---

## Files Status

### ✅ Migrated (Phase 1)

- `prompt.py` → `prompts.py`
- `failure_mode_pipeline.py` → `pipeline.py`
- `README.md` → `README.md`
- `README_detail.md` → `README_detail.md`

### 🔄 To Migrate (Phase 2-3)

- `utils.py` → `utils.py` (needs refactoring)
- `failure_mode_generator.py` → `generator.py`
- `failure_mode_reduction.py` → `reducer.py`
- `failure_mode_extractor.py` → `extractor.py`
- `plot_failure_mode.py` → `visualizer.py`
- `failure_mode_generator_test.py` → `tests/test_generator.py`
- `failure_mode_reduction_test.py` → `tests/test_reducer.py`

### 📦 Data Files (preserved)

- `processed_trajectories/*.pkl` (will be generated in new location)
- `summary_codabench/*.csv` (will be generated in new location)
- Visualization outputs (will be generated in new location)

---

## Risk Mitigation

- **Parallel existence**: Old and new code coexist during Phases 1-5
- **Backward compatibility**: Wrapper functions maintain old API during transition
- **Incremental testing**: Each phase has validation step
- **Rollback capability**: Original code remains until Phase 6

---

## Current Status

**Phase 1**: ✅ **COMPLETE**  
**Next Step**: Begin Phase 2 - LLM Integration Refactoring

---

*Last Updated: 2026-03-31*