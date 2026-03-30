# from collections import OrderedDict
# from sentence_transformers import SentenceTransformer


# class ModelCache:
#     """Cache for storing at most three sentence-transformer models using LRU eviction."""

#     def __init__(self, max_size: int = 3):
#         """
#         Initialize the model cache.

#         Args:
#             max_size: Maximum number of models to cache (default: 3)
#         """
#         self.max_size = max_size
#         self.cache = OrderedDict()

#     def get(self, model_name: str) -> SentenceTransformer:
#         """
#         Get a model from cache or load it if not present.

#         Args:
#             model_name: Name of the sentence-transformer model

#         Returns:
#             The loaded SentenceTransformer model
#         """
#         if model_name in self.cache:
#             # Move to end to mark as recently used
#             self.cache.move_to_end(model_name)
#             return self.cache[model_name]

#         # Load the model
#         model = SentenceTransformer(model_name)

#         # Add to cache
#         self.cache[model_name] = model
#         self.cache.move_to_end(model_name)

#         # Evict least recently used if cache is full
#         if len(self.cache) > self.max_size:
#             self.cache.popitem(last=False)

#         return model

#     def clear(self):
#         """Clear all models from the cache."""
#         self.cache.clear()

#     def __len__(self) -> int:
#         """Return the number of models currently in cache."""
#         return len(self.cache)

#     def __contains__(self, model_name: str) -> bool:
#         """Check if a model is in the cache."""
#         return model_name in self.cache
