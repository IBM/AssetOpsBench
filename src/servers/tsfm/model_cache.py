"""Module-level model caching and pre-compilation.

Models are loaded and compiled once at server startup, then reused for all inference calls.
This eliminates the loading overhead from individual tool invocations.
"""
# model_cache.py


from __future__ import annotations

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger("tsfm-mcp-server")

# Module-level singletons
_COMPILED_MODELS: Dict[str, object] = {}
_COMPILED_MODEL_CONFIGS: Dict[str, dict] = {}
_TSP_CACHE: Dict[str, object] = {}  # TimeSeriesPreprocessor instances


def preload_and_compile_models(model_names: list[str], model_dir: str) -> None:
    """Pre-load and compile all TTM models at server startup.
    
    Args:
        model_names: List of model checkpoint names (e.g., ["ttm_96_28", "ttm_512_96"])
        model_dir: Directory containing model checkpoint folders
    
    This function:
    1. Loads each model checkpoint
    2. Compiles it with torch.compile(mode='reduce-overhead')
    3. Stores references in module-level singletons
    4. Should be called once at server startup
    """
    try:
        import torch
        from tsfm_public import TinyTimeMixerForPrediction
        import json
    except ImportError as exc:
        logger.warning(f"Cannot preload models; dependencies unavailable: {exc}")
        return
    
    for model_name in model_names:
        checkpoint_path = os.path.join(model_dir, model_name)
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Model checkpoint not found: {checkpoint_path}")
            continue
        
        try:
            logger.info(f"Pre-loading model: {model_name}")

                        # Load and cache model config
            config_path = os.path.join(checkpoint_path, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            _COMPILED_MODEL_CONFIGS[model_name] = config
            
            # Load model
            logger.info(f"preapring Compiling model: {model_name} path is {checkpoint_path} with torch.compile(mode='reduce-overhead') and second arg as {_COMPILED_MODEL_CONFIGS[model_name]["prediction_length"]}")

            model = TinyTimeMixerForPrediction.from_pretrained(checkpoint_path
                                                               , prediction_filter_length=_COMPILED_MODEL_CONFIGS[model_name]["prediction_length"]
                                                               )
            
            # Move to eval mode
            model.eval()
            
            # Compile with reduce-overhead mode for maximum inference speedup
            logger.info(f"Compiling model: {model_name} with torch.compile(mode='reduce-overhead')")
            compiled_model = torch.compile(model, mode='reduce-overhead')
            
            # Store compiled model
            _COMPILED_MODELS[model_name] = compiled_model
            

            
            logger.info(f"✓ Successfully pre-loaded and compiled: {model_name}")
            
        except Exception as exc:
            logger.error(f"Failed to pre-load model {model_name}: {exc}")


def get_compiled_model(model_name: str) -> Optional[object]:
    """Retrieve a pre-compiled model from cache.
    
    Args:
        model_name: Model checkpoint name
    
    Returns:
        Compiled model instance or None if not found
    """
    if model_name not in _COMPILED_MODELS:
        logger.warning(f"Model not in cache: {model_name}. Available: {list(_COMPILED_MODELS.keys())}")
        return None
    return _COMPILED_MODELS[model_name]


def get_model_config(model_name: str) -> Optional[dict]:
    """Retrieve cached model config.
    
    Args:
        model_name: Model checkpoint name
    
    Returns:
        Model config dict or None if not found
    """
    return _COMPILED_MODEL_CONFIGS.get(model_name)


def cache_tsp(key: str, tsp: object) -> None:
    """Cache a TimeSeriesPreprocessor instance for reuse.
    
    Args:
        key: Cache key (e.g., dataset hash or config signature)
        tsp: TimeSeriesPreprocessor instance
    """
    _TSP_CACHE[key] = tsp


def get_cached_tsp(key: str) -> Optional[object]:
    """Retrieve a cached TimeSeriesPreprocessor.
    
    Args:
        key: Cache key
    
    Returns:
        TimeSeriesPreprocessor or None if not found
    """
    return _TSP_CACHE.get(key)


def clear_tsp_cache() -> None:
    """Clear the TimeSeriesPreprocessor cache."""
    _TSP_CACHE.clear()
    logger.info("TSP cache cleared")


def list_cached_models() -> list[str]:
    """List all pre-loaded model names."""
    return list(_COMPILED_MODELS.keys())