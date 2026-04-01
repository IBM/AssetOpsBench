# LiteLLM Proxy Models Test Report

**Date:** 2026-04-01  
**Test Tool:** `test_all_litellm_models.py`  
**Total Models Tested:** 19

## Executive Summary

✅ **18 out of 19 models work successfully (95% success rate)**

### Overall Results
- ✅ Working: 18/19
- ❌ Failed: 1/19

## Results by Provider

### Claude Models

#### AWS Claude (3/3 working) ✅
- ✅ `litellm_proxy/aws/claude-sonnet-4-6` - **RECOMMENDED**
- ✅ `litellm_proxy/aws/claude-opus-4-6`
- ✅ `litellm_proxy/aws/claude-3-5-haiku`

#### GCP Claude (0/1 working) ❌
- ❌ `litellm_proxy/GCP/claude-3-7-sonnet` - Configuration issue

### Gemini Models (5/5 working) ✅
- ✅ `litellm_proxy/GCP/gemini-2.0-flash-lite`
- ✅ `litellm_proxy/GCP/gemini-2.5-flash-lite`
- ✅ `litellm_proxy/gcp/gemini-3.1-pro-preview`
- ✅ `litellm_proxy/gemini-2.5-pro`
- ✅ `litellm_proxy/gemini-2.5-flash`

### GPT Models (10/10 working) ✅
- ✅ `litellm_proxy/Azure/gpt-5-2025-08-07`
- ✅ `litellm_proxy/Azure/gpt-5-mini-2025-08-07`
- ✅ `litellm_proxy/Azure/gpt-5-nano-2025-08-07`
- ✅ `litellm_proxy/Azure/gpt-5-chat-2025-08-07`
- ✅ `litellm_proxy/Azure/gpt-4.1`
- ✅ `litellm_proxy/Azure/gpt-4.1-mini`
- ✅ `litellm_proxy/Azure/gpt-4.1-nano`
- ✅ `litellm_proxy/Azure/o4-mini`
- ✅ `litellm_proxy/azure/gpt-5.4`
- ✅ `litellm_proxy/azure/gpt-5.3-chat`

## Recommendations

### For Trajectory Analysis

**Primary Recommendation:**
```python
LiteLLMBackend("litellm_proxy/aws/claude-sonnet-4-6")
```

This is the most advanced Claude model available and works reliably.

**Alternative Options:**
1. `litellm_proxy/aws/claude-opus-4-6` - Most capable Claude model
2. `litellm_proxy/Azure/gpt-5-2025-08-07` - Latest GPT model
3. `litellm_proxy/gemini-2.5-pro` - Google's most capable model

### For Different Use Cases

**Fast responses:**
- `litellm_proxy/aws/claude-3-5-haiku` (Claude, fastest)
- `litellm_proxy/GCP/gemini-2.5-flash-lite` (Gemini, very fast)
- `litellm_proxy/Azure/gpt-4.1-nano` (GPT, smallest/fastest)

**Cost-effective:**
- `litellm_proxy/Azure/gpt-5-mini-2025-08-07`
- `litellm_proxy/Azure/gpt-5-nano-2025-08-07`
- `litellm_proxy/GCP/gemini-2.0-flash-lite`

**Maximum capability:**
- `litellm_proxy/aws/claude-opus-4-6` (Claude's most capable)
- `litellm_proxy/Azure/gpt-5-2025-08-07` (GPT's latest)
- `litellm_proxy/gemini-2.5-pro` (Gemini's most capable)

## Known Issues

### GCP Claude Configuration Error
**Model:** `litellm_proxy/GCP/claude-3-7-sonnet`  
**Error:** "Invalid model name passed in model=GCP/claude-3-7-sonnet"  
**Status:** Proxy configuration issue  
**Workaround:** Use AWS Claude models instead

### INSTRUCTIONS.md Discrepancy
**Issue:** INSTRUCTIONS.md references `litellm_proxy/GCP/claude-4-sonnet` which is not available in the proxy  
**Resolution:** Use `litellm_proxy/aws/claude-sonnet-4-6` instead

## Code Updates Required

### Update Default Model in `generator.py`

**Current (broken):**
```python
llm_backend = LiteLLMBackend("litellm_proxy/GCP/claude-3-7-sonnet")
```

**Should be (working):**
```python
llm_backend = LiteLLMBackend("litellm_proxy/aws/claude-sonnet-4-6")
```

## Testing Tools

### Individual Model Testing
```bash
uv run python src/trajectory_analysis/failure_mode/test_llm_model_connection.py \
  --model-id <model_id>
```

### Comprehensive Testing
```bash
uv run python src/trajectory_analysis/failure_mode/test_all_litellm_models.py
```

## Environment Requirements

**For LiteLLM Proxy models:**
- `LITELLM_API_KEY` - Your LiteLLM proxy API key
- `LITELLM_BASE_URL` - Your LiteLLM proxy URL

**For WatsonX models (Llama):**
- `WATSONX_APIKEY` - IBM WatsonX API key
- `WATSONX_PROJECT_ID` - WatsonX project ID

## Conclusion

The LiteLLM proxy is well-configured with 18 working models across Claude (AWS), Gemini, and GPT providers. The only issue is with the GCP Claude endpoint, which can be easily worked around by using AWS Claude models instead.

**Action Items:**
1. ✅ Update default model to `aws/claude-sonnet-4-6`
2. ✅ Document that GCP Claude is not available
3. ✅ Update INSTRUCTIONS.md to reference working models