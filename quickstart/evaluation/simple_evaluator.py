"""
Simple Evaluator for AssetOpsBench Quickstart.
A lightweight evaluation system for the quickstart mode.
"""

import re
from typing import Dict, List, Any


class SimpleEvaluator:
    """Simple evaluator for quickstart mode."""
    
    def __init__(self):
        self.scoring_rules = {
            'response_completeness': 0.3,
            'tool_usage': 0.2,
            'accuracy': 0.3,
            'clarity': 0.2
        }
    
    def evaluate(self, task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate agent performance on a task."""
        
        # Extract relevant information
        task_text = task.get('text', '').lower()
        response = result.get('response', '').lower()
        tools_used = result.get('tools_used', [])
        expected_tools = task.get('expected_tools', [])
        
        # Initialize scores
        scores = {
            'response_completeness': self._evaluate_completeness(task_text, response),
            'tool_usage': self._evaluate_tool_usage(tools_used, expected_tools),
            'accuracy': self._evaluate_accuracy(task_text, response),
            'clarity': self._evaluate_clarity(response)
        }
        
        # Calculate weighted total score
        total_score = sum(
            score * self.scoring_rules[metric]
            for metric, score in scores.items()
        )
        
        # Generate feedback
        feedback = self._generate_feedback(scores, task_text, response, tools_used)
        
        return {
            'score': round(total_score, 1),
            'detailed_scores': scores,
            'feedback': feedback,
            'tools_used': tools_used,
            'expected_tools': expected_tools
        }
    
    def _evaluate_completeness(self, task: str, response: str) -> float:
        """Evaluate if the response addresses the task completely."""
        if not response:
            return 0.0
        
        # Check for common response patterns
        if 'not sure' in response or 'don\'t know' in response:
            return 20.0
        
        # Check if response provides information
        if len(response) < 10:
            return 30.0
        
        # Check for specific information based on task type
        if 'sites' in task and ('site' in response or 'facility' in response):
            return 90.0
        elif 'assets' in task and ('asset' in response or 'chiller' in response or 'pump' in response):
            return 90.0
        elif 'status' in task and ('status' in response or 'operational' in response or 'maintenance' in response):
            return 90.0
        elif 'maintenance' in task and ('maintenance' in response or 'schedule' in response):
            return 90.0
        
        # Default score for reasonable responses
        return 70.0
    
    def _evaluate_tool_usage(self, tools_used: List[str], expected_tools: List[str]) -> float:
        """Evaluate if appropriate tools were used."""
        if not expected_tools:
            return 100.0  # No specific tools expected
        
        if not tools_used:
            return 0.0  # Tools were expected but none used
        
        # Calculate overlap between used and expected tools
        used_set = set(tools_used)
        expected_set = set(expected_tools)
        
        if not expected_set:
            return 100.0
        
        overlap = len(used_set & expected_set)
        expected_count = len(expected_set)
        
        # Score based on proportion of expected tools used
        return (overlap / expected_count) * 100.0
    
    def _evaluate_accuracy(self, task: str, response: str) -> float:
        """Evaluate the accuracy of the response."""
        if not response:
            return 0.0
        
        # Check for error indicators
        error_indicators = ['error', 'failed', 'cannot', 'unable', 'not found']
        if any(indicator in response for indicator in error_indicators):
            return 30.0
        
        # Check for reasonable data patterns
        if re.search(r'\d+ items? found', response):
            return 90.0
        elif re.search(r'chiller|pump|site', response):
            return 85.0
        elif re.search(r'operational|maintenance|unknown', response):
            return 85.0
        
        # Default score for non-error responses
        return 75.0
    
    def _evaluate_clarity(self, response: str) -> float:
        """Evaluate the clarity and readability of the response."""
        if not response:
            return 0.0
        
        # Length-based scoring
        if len(response) < 10:
            return 40.0
        elif len(response) > 500:
            return 70.0  # Too long
        
        # Structure-based scoring
        if ':' in response or ',' in response:
            return 90.0  # Well-structured
        
        # Grammar and readability heuristics
        words = response.split()
        if len(words) < 3:
            return 50.0
        elif len(words) > 50:
            return 80.0
        
        return 85.0
    
    def _generate_feedback(self, scores: Dict[str, float], task: str, response: str, tools_used: List[str]) -> str:
        """Generate feedback based on evaluation scores."""
        feedback_parts = []
        
        # Response completeness feedback
        if scores['response_completeness'] >= 80:
            feedback_parts.append("✅ Response addresses the task well")
        elif scores['response_completeness'] >= 60:
            feedback_parts.append("⚠️ Response partially addresses the task")
        else:
            feedback_parts.append("❌ Response doesn't fully address the task")
        
        # Tool usage feedback
        if scores['tool_usage'] >= 80:
            feedback_parts.append("✅ Appropriate tools were used")
        elif scores['tool_usage'] >= 60:
            feedback_parts.append("⚠️ Some tools could have been used better")
        else:
            feedback_parts.append("❌ Tool usage needs improvement")
        
        # Accuracy feedback
        if scores['accuracy'] >= 80:
            feedback_parts.append("✅ Response appears accurate")
        elif scores['accuracy'] >= 60:
            feedback_parts.append("⚠️ Response may have some inaccuracies")
        else:
            feedback_parts.append("❌ Response accuracy needs improvement")
        
        # Clarity feedback
        if scores['clarity'] >= 80:
            feedback_parts.append("✅ Response is clear and well-structured")
        elif scores['clarity'] >= 60:
            feedback_parts.append("⚠️ Response could be clearer")
        else:
            feedback_parts.append("❌ Response clarity needs improvement")
        
        return " | ".join(feedback_parts)
