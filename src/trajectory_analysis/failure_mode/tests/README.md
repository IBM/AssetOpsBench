# Trajectory Analysis Tests

## Setup

1. **Install test dependencies**:
   ```bash
   pip install pytest python-dotenv
   ```

2. **Configure environment variables**:
   
   The `.env` file should be at the **project root** (not in this module):
   ```bash
   # .env file location: /path/to/AssetOpsBench/.env
   
   # Required variables:
   WATSONX_APIKEY=your-watsonx-api-key
   WATSONX_PROJECT_ID=your-project-id
   WATSONX_URL=https://us-south.ml.cloud.ibm.com
   
   LITELLM_API_KEY=your-litellm-api-key
   LITELLM_BASE_URL=https://your-litellm-proxy-url
   ```
   
   ⚠️ **DO NOT commit .env file to git!**

3. **Run tests**:
   ```bash
   # Run all tests
   pytest src/trajectory_analysis/failure_mode/tests/
   
   # Run specific test file
   pytest src/trajectory_analysis/failure_mode/tests/test_generator.py
   
   # Run with verbose output
   pytest -v src/trajectory_analysis/failure_mode/tests/
   ```

## Test Fixtures

The `conftest.py` provides several fixtures:

- `llm_claude`: Claude 4 Sonnet backend (default, best accuracy)
- `llm_llama`: Llama 3.3 70B backend (cost-effective)
- `llm_granite`: Granite backend (lowest cost)
- `sample_trajectory`: Sample trajectory data for testing
- `temp_output_dir`: Temporary directory for test outputs

## Example Test

```python
def test_trajectory_analysis(llm_claude, sample_trajectory, temp_output_dir):
    """Test trajectory analysis with Claude 4 Sonnet."""
    from src.trajectory_analysis.failure_mode.utils import get_llm_answer_from_json
    
    result = get_llm_answer_from_json(
        data=sample_trajectory,
        llm_backend=llm_claude,
        temperature=0.0
    )
    
    assert result is not None
    assert len(result) > 0
```

## Security Notes

⚠️ **IMPORTANT**: 
- Never commit `.env` files with real API keys
- Use `.env.example` as a template
- API keys are loaded from environment variables only
- Tests will skip if required environment variables are missing