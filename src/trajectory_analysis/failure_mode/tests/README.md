# Trajectory Analysis Tests - Phase 2

Test suite for trajectory analysis and failure mode detection (Phase 1 & 2 complete).

## Phase 2 Test Coverage

### Test Files Created:

1. **`test_utils.py`** (254 lines)
   - Tests for `get_llm_answer_from_json()` function
   - Tests for `extract_json_from_response()` function
   - Mock-based tests (no API calls required)
   - Coverage: JSON extraction, LLM prompt formatting, error handling

2. **`test_generator.py`** (363 lines)
   - Tests for `process_trajectories()` function
   - Tests for helper functions (`_load_all_json_files`, `_normalize_additional_failure_modes`)
   - Mock-based and integration tests
   - Coverage: File loading, trajectory processing, DataFrame generation

3. **`test_pipeline.py`** (254 lines)
   - Tests for `run_failure_mode_pipeline()` function
   - Tests for default LLM backend creation
   - Tests for parameter passing
   - Coverage: Pipeline orchestration, LLM backend integration

**Total: 871 lines of test code covering Phase 2 implementation**

## Setup

### Option 1: Using pip

1. **Install test dependencies**:
   ```bash
   pip install pytest pytest-cov pytest-mock pandas
   ```

### Option 2: Using uv (Recommended)

1. **Install test dependencies with uv**:
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install dependencies
   uv pip install pytest pytest-cov pytest-mock pandas
   ```

2. **Configure environment variables** (optional for Phase 2 tests):
   
   The `.env` file should be at the **project root** (not in this module):
   ```bash
   # .env file location: /path/to/AssetOpsBench/.env
   
   # For Claude 4 Sonnet (via LiteLLM proxy)
   LITELLM_API_KEY=your-litellm-api-key
   LITELLM_BASE_URL=https://ete-litellm.ai-models.vpc-int.res.ibm.com
   
   # For Llama 3.3 70B and Granite (via WatsonX)
   WATSONX_APIKEY=your-watsonx-api-key
   WATSONX_PROJECT_ID=c4bfae5a-377f-44d6-b37a-68435a056744
   WATSONX_URL=https://us-south.ml.cloud.ibm.com
   ```
   
   ⚠️ **Note**: Phase 2 tests use mocks and don't require API keys!

## Running Tests

### Using pip/pytest (Standard)

#### Run All Phase 2 Tests:
```bash
# From project root
pytest src/trajectory_analysis/failure_mode/tests/ -v
```

#### Run Specific Test Files:
```bash
# Test utilities
pytest src/trajectory_analysis/failure_mode/tests/test_utils.py -v

# Test generator
pytest src/trajectory_analysis/failure_mode/tests/test_generator.py -v

# Test pipeline
pytest src/trajectory_analysis/failure_mode/tests/test_pipeline.py -v
```

#### Run with Coverage Report:
```bash
pytest --cov=src.trajectory_analysis.failure_mode \
       --cov-report=html \
       --cov-report=term \
       src/trajectory_analysis/failure_mode/tests/
```

View HTML coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

#### Run Specific Test Classes:
```bash
# Test JSON extraction only
pytest src/trajectory_analysis/failure_mode/tests/test_utils.py::TestExtractJsonFromResponse -v

# Test trajectory processing only
pytest src/trajectory_analysis/failure_mode/tests/test_generator.py::TestProcessTrajectories -v

# Test pipeline orchestration
pytest src/trajectory_analysis/failure_mode/tests/test_pipeline.py::TestRunFailureModePipeline -v
```

#### Run with Verbose Output:
```bash
pytest src/trajectory_analysis/failure_mode/tests/ -v -s
```

#### Run Tests in Parallel (faster):
```bash
pip install pytest-xdist
pytest src/trajectory_analysis/failure_mode/tests/ -n auto
```

### Using uv (Recommended for faster execution)

#### Run All Phase 2 Tests:
```bash
# From project root
uv run pytest src/trajectory_analysis/failure_mode/tests/ -v
```

#### Run Specific Test Files:
```bash
# Test utilities
uv run pytest src/trajectory_analysis/failure_mode/tests/test_utils.py -v

# Test generator
uv run pytest src/trajectory_analysis/failure_mode/tests/test_generator.py -v

# Test pipeline
uv run pytest src/trajectory_analysis/failure_mode/tests/test_pipeline.py -v
```

#### Run with Coverage Report:
```bash
uv run pytest --cov=src.trajectory_analysis.failure_mode \
              --cov-report=html \
              --cov-report=term \
              src/trajectory_analysis/failure_mode/tests/
```

#### Run Tests in Parallel with uv:
```bash
uv pip install pytest-xdist
uv run pytest src/trajectory_analysis/failure_mode/tests/ -n auto
```

#### Quick Test (uv with minimal output):
```bash
uv run pytest src/trajectory_analysis/failure_mode/tests/ -q
```

## Test Structure

### Mock-Based Tests (No API Calls Required)
Most Phase 2 tests use mocks to avoid actual LLM API calls:
- ✅ Fast execution (< 1 second per test)
- ✅ No API costs
- ✅ Deterministic results
- ✅ Test error conditions easily
- ✅ No environment setup required

### Integration Tests
Some tests create temporary files and test complete flows:
- File I/O operations
- DataFrame generation and pickle files
- Complete pipeline execution with mocked LLMs

## Test Fixtures (conftest.py)

The `conftest.py` file provides LLM backend fixtures for future integration testing:

- `llm_claude`: Claude 4 Sonnet backend (requires API key)
- `llm_llama`: Llama 3.3 70B backend (requires WatsonX credentials)
- `llm_granite`: Granite backend (requires WatsonX credentials)
- `sample_trajectory`: Sample trajectory data for testing
- `temp_output_dir`: Temporary directory for test outputs

**Note**: Current Phase 2 tests use mocks and don't require these fixtures.

Example usage for future integration tests:
```python
def test_with_real_llm(llm_claude, sample_trajectory):
    """Integration test with real Claude 4 Sonnet."""
    result = get_llm_answer_from_json(
        data=sample_trajectory,
        llm_backend=llm_claude,
        temperature=0.0
    )
    assert result is not None
```

## Test Coverage Summary

### `test_utils.py` (254 lines):
- ✅ JSON extraction from markdown code fences
- ✅ JSON extraction from plain text
- ✅ Multiline JSON handling
- ✅ Error handling for invalid JSON
- ✅ Error handling for malformed JSON
- ✅ LLM prompt formatting with trajectories
- ✅ Empty trajectory handling
- ✅ Multiple trajectory steps
- ✅ Missing fields in trajectory data
- ✅ Temperature parameter passing
- ✅ Error handling for LLM failures
- ✅ Integration test combining functions

**12 test classes, 15+ test methods**

### `test_generator.py` (363 lines):
- ✅ Normalize additional failure modes (None, list, dict)
- ✅ File loading (recursive, error handling)
- ✅ Skip invalid JSON files
- ✅ Empty directory handling
- ✅ Trajectory processing with mock LLM
- ✅ Default LLM backend creation (Claude 4 Sonnet)
- ✅ Custom LLM backend support
- ✅ Temperature parameter passing
- ✅ Error handling and retries
- ✅ Additional failure modes handling
- ✅ Output directory creation
- ✅ DataFrame and pickle file generation
- ✅ End-to-end integration test

**10 test classes, 20+ test methods**

### `test_pipeline.py` (254 lines):
- ✅ Default Claude 4 Sonnet backend creation
- ✅ Custom LLM backend support
- ✅ Temperature parameter passing
- ✅ Timestamps parameter passing
- ✅ All optional parameters
- ✅ Default temperature (0.0)
- ✅ Result structure validation
- ✅ End-to-end mock integration
- ✅ Documentation verification
- ✅ Function signature validation

**3 test classes, 12+ test methods**

## Running Tests Without API Keys

✅ **All Phase 2 tests work without any API configuration!**

### Using pytest:
```bash
# These work immediately after cloning the repo
pytest src/trajectory_analysis/failure_mode/tests/test_utils.py
pytest src/trajectory_analysis/failure_mode/tests/test_generator.py
pytest src/trajectory_analysis/failure_mode/tests/test_pipeline.py
```

### Using uv:
```bash
# Same tests with uv (faster dependency resolution)
uv run pytest src/trajectory_analysis/failure_mode/tests/test_utils.py
uv run pytest src/trajectory_analysis/failure_mode/tests/test_generator.py
uv run pytest src/trajectory_analysis/failure_mode/tests/test_pipeline.py
```

No `.env` file needed for Phase 2 tests!

## Example Test Output

### Using pytest:
```bash
$ pytest src/trajectory_analysis/failure_mode/tests/ -v

test_utils.py::TestExtractJsonFromResponse::test_extract_json_with_markdown_code_fence PASSED
test_utils.py::TestExtractJsonFromResponse::test_extract_json_without_code_fence PASSED
test_utils.py::TestGetLlmAnswerFromJson::test_get_llm_answer_basic_trajectory PASSED
...
test_generator.py::TestProcessTrajectories::test_process_trajectories_basic PASSED
test_generator.py::TestProcessTrajectories::test_process_trajectories_default_llm PASSED
...
test_pipeline.py::TestRunFailureModePipeline::test_pipeline_with_default_llm PASSED
test_pipeline.py::TestRunFailureModePipeline::test_pipeline_with_custom_llm PASSED
...

======================== 47 passed in 2.34s ========================
```

### Using uv:
```bash
$ uv run pytest src/trajectory_analysis/failure_mode/tests/ -v

test_utils.py::TestExtractJsonFromResponse::test_extract_json_with_markdown_code_fence PASSED
test_utils.py::TestExtractJsonFromResponse::test_extract_json_without_code_fence PASSED
...

======================== 47 passed in 1.89s ========================
```

**Note**: uv typically runs faster due to optimized dependency resolution and caching.

## Continuous Integration

Add to your CI/CD pipeline:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install pytest pytest-cov pandas
      - run: pytest src/trajectory_analysis/failure_mode/tests/ --cov
```

## Next Steps (Phase 4)

Phase 4 will add:
- ✅ Tests for `reducer.py` (failure mode clustering)
- ✅ Tests for `extractor.py` (failure mode extraction)
- ✅ Tests for `visualizer.py` (plotting)
- ✅ Integration tests with real LLM backends
- ✅ End-to-end pipeline tests with all components
- ✅ Performance benchmarks

## Security Notes

⚠️ **IMPORTANT**: 
- Never commit `.env` files with real API keys
- Use `.env.example` as a template
- API keys are loaded from environment variables only
- Phase 2 tests don't require API keys (use mocks)
- Integration tests will skip if required environment variables are missing

## Troubleshooting

### Import Errors
```bash
# Make sure you're running from project root
cd /path/to/AssetOpsBench

# With pytest
pytest src/trajectory_analysis/failure_mode/tests/

# With uv
uv run pytest src/trajectory_analysis/failure_mode/tests/
```

### Missing Dependencies

**Using pip:**
```bash
pip install pytest pytest-cov pytest-mock pandas
```

**Using uv:**
```bash
uv pip install pytest pytest-cov pytest-mock pandas
```

### Test Discovery Issues
```bash
# Ensure __init__.py files exist
ls src/trajectory_analysis/failure_mode/tests/__init__.py
```

### uv Not Found
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### Slow Test Execution
```bash
# Use uv for faster execution
uv run pytest src/trajectory_analysis/failure_mode/tests/

# Or run tests in parallel
uv pip install pytest-xdist
uv run pytest src/trajectory_analysis/failure_mode/tests/ -n auto
```

## Contributing

When adding new tests:
1. Follow existing test structure and naming conventions
2. Use mocks for external dependencies (LLMs, file I/O when possible)
3. Add docstrings to test functions
4. Group related tests in classes
5. Update this README with new test coverage