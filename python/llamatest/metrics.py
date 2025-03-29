"""
Metrics module for evaluating AI test results.

This module provides various metrics for evaluating the performance
of AI models against test cases, including accuracy, precision,
recall, F1 score, and specialized metrics for NLP and LLM outputs.
"""
from typing import List, Dict, Any, Union, Optional, Callable
import numpy as np
from collections import Counter


def accuracy(expected: List[Any], actual: List[Any]) -> float:
    """
    Calculate the accuracy of predictions.
    
    Args:
        expected: List of expected outputs
        actual: List of actual outputs
        
    Returns:
        Accuracy score between 0 and 1
    """
    if len(expected) != len(actual):
        raise ValueError("Expected and actual lists must have the same length")
    
    if not expected:
        return 0.0
    
    correct = sum(1 for e, a in zip(expected, actual) if e == a)
    return correct / len(expected)


def precision_recall_f1(
    expected: List[Any], 
    actual: List[Any], 
    positive_class: Any = True
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        expected: List of expected outputs
        actual: List of actual outputs
        positive_class: The class to consider as positive
        
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    if len(expected) != len(actual):
        raise ValueError("Expected and actual lists must have the same length")
    
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Calculate true positives, false positives, false negatives
    tp = sum(1 for e, a in zip(expected, actual) if e == positive_class and a == positive_class)
    fp = sum(1 for e, a in zip(expected, actual) if e != positive_class and a == positive_class)
    fn = sum(1 for e, a in zip(expected, actual) if e == positive_class and a != positive_class)
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def confusion_matrix(expected: List[Any], actual: List[Any], labels: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Generate a confusion matrix for multiclass classification.
    
    Args:
        expected: List of expected outputs
        actual: List of actual outputs
        labels: Optional list of class labels
        
    Returns:
        Dictionary with confusion matrix and derived metrics
    """
    if len(expected) != len(actual):
        raise ValueError("Expected and actual lists must have the same length")
    
    if not expected:
        return {"matrix": {}, "accuracy": 0.0, "per_class": {}}
    
    # Determine all unique labels if not provided
    if labels is None:
        labels = sorted(list(set(expected) | set(actual)))
    
    # Initialize confusion matrix
    matrix = {e: {a: 0 for a in labels} for e in labels}
    
    # Fill confusion matrix
    for e, a in zip(expected, actual):
        if e in matrix and a in labels:
            matrix[e][a] = matrix[e].get(a, 0) + 1
    
    # Calculate per-class metrics
    per_class = {}
    total_samples = len(expected)
    correct_samples = sum(1 for e, a in zip(expected, actual) if e == a)
    
    for label in labels:
        # True positives: diagonal element
        tp = matrix[label][label] if label in matrix and label in matrix[label] else 0
        
        # False positives: sum of column - true positives
        fp = sum(matrix.get(e, {}).get(label, 0) for e in labels) - tp
        
        # False negatives: sum of row - true positives
        fn = sum(matrix.get(label, {}).get(a, 0) for a in labels) - tp
        
        # True negatives: total - (tp + fp + fn)
        tn = total_samples - (tp + fp + fn)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn  # Total samples of this class
        }
    
    return {
        "matrix": matrix,
        "accuracy": correct_samples / total_samples if total_samples > 0 else 0.0,
        "per_class": per_class
    }


def text_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
    """
    Calculate similarity between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        method: Similarity method ('jaccard', 'cosine', 'levenshtein')
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 and not text2:
        return 1.0
    
    if not text1 or not text2:
        return 0.0
    
    if method == "jaccard":
        # Jaccard similarity on word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    elif method == "cosine":
        # Cosine similarity on word counts
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        if not words1 and not words2:
            return 1.0
        
        # Count word frequencies
        counter1 = Counter(words1)
        counter2 = Counter(words2)
        
        # Get all unique words
        all_words = set(counter1.keys()) | set(counter2.keys())
        
        # Create vectors
        vector1 = [counter1.get(word, 0) for word in all_words]
        vector2 = [counter2.get(word, 0) for word in all_words]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = sum(a * a for a in vector1) ** 0.5
        magnitude2 = sum(b * b for b in vector2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    elif method == "levenshtein":
        # Levenshtein distance (normalized)
        m, n = len(text1), len(text2)
        
        # Create distance matrix
        d = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j
        
        # Fill distance matrix
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if text1[i - 1] == text2[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(
                        d[i - 1][j] + 1,  # deletion
                        d[i][j - 1] + 1,  # insertion
                        d[i - 1][j - 1] + 1  # substitution
                    )
        
        # Calculate normalized distance
        max_len = max(m, n)
        distance = d[m][n]
        
        if max_len == 0:
            return 1.0
        
        # Convert distance to similarity (1 - normalized distance)
        return 1.0 - (distance / max_len)
    
    else:
        raise ValueError(f"Unsupported similarity method: {method}")


def llm_output_evaluation(
    expected: str, 
    actual: str, 
    metrics: List[str] = ["similarity", "content", "format"]
) -> Dict[str, float]:
    """
    Comprehensive evaluation of LLM output against expected output.
    
    Args:
        expected: Expected LLM output
        actual: Actual LLM output
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary with evaluation scores
    """
    results = {}
    
    if "similarity" in metrics:
        # Calculate basic text similarity
        results["text_similarity"] = text_similarity(expected, actual, "jaccard")
        results["cosine_similarity"] = text_similarity(expected, actual, "cosine")
        results["levenshtein_similarity"] = text_similarity(expected, actual, "levenshtein")
    
    if "content" in metrics:
        # Content-based metrics
        # Calculate word overlap
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if expected_words:
            # Word coverage (what fraction of expected words are in the actual)
            word_coverage = len(expected_words & actual_words) / len(expected_words)
            results["word_coverage"] = word_coverage
        else:
            results["word_coverage"] = 1.0 if not actual_words else 0.0
        
        # Calculate keyword presence
        if expected_words:
            # Extract potential keywords (longer words are more likely to be important)
            keywords = {word for word in expected_words if len(word) > 4}
            if keywords:
                keyword_presence = len(keywords & actual_words) / len(keywords)
                results["keyword_presence"] = keyword_presence
            else:
                results["keyword_presence"] = 1.0
        else:
            results["keyword_presence"] = 1.0 if not actual_words else 0.0
    
    if "format" in metrics:
        # Format-based metrics
        expected_lines = expected.strip().split("\n")
        actual_lines = actual.strip().split("\n")
        
        # Line count similarity
        max_lines = max(len(expected_lines), len(actual_lines))
        line_diff = abs(len(expected_lines) - len(actual_lines))
        if max_lines > 0:
            results["line_count_similarity"] = 1.0 - (line_diff / max_lines)
        else:
            results["line_count_similarity"] = 1.0
        
        # Calculate whitespace pattern similarity
        def get_whitespace_pattern(text):
            lines = text.split('\n')
            return [len(line) - len(line.lstrip()) for line in lines]
        
        expected_pattern = get_whitespace_pattern(expected)
        actual_pattern = get_whitespace_pattern(actual)
        
        # Truncate to the shorter length
        min_lines = min(len(expected_pattern), len(actual_pattern))
        if min_lines > 0:
            # Calculate mean absolute difference in indentation
            indent_diff = sum(abs(e - a) for e, a in zip(expected_pattern[:min_lines], actual_pattern[:min_lines])) / min_lines
            # Normalize to a similarity score (assume 10 spaces is maximum difference)
            results["whitespace_similarity"] = max(0.0, 1.0 - (indent_diff / 10.0))
        else:
            results["whitespace_similarity"] = 1.0 if not expected and not actual else 0.0
    
    # Calculate overall score (average of all metrics)
    results["overall_score"] = sum(results.values()) / len(results) if results else 0.0
    
    return results


def regression_metrics(expected: List[float], actual: List[float]) -> Dict[str, float]:
    """
    Calculate metrics for regression tasks.
    
    Args:
        expected: List of expected values
        actual: List of actual values
        
    Returns:
        Dictionary with regression metrics
    """
    if len(expected) != len(actual):
        raise ValueError("Expected and actual lists must have the same length")
    
    if not expected:
        return {
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "r2": 0.0
        }
    
    # Convert to numpy arrays for vectorized operations
    y_true = np.array(expected)
    y_pred = np.array(actual)
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    # R² = 1 - (sum of squared residuals / total sum of squares)
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }


class MetricRegistry:
    """
    Registry for custom evaluation metrics.
    
    This class allows users to register and retrieve custom
    evaluation metrics for use with TestSuite and TestCase.
    """
    
    _metrics = {}
    
    @classmethod
    def register(cls, name: str, metric_fn: Callable, description: str = "") -> None:
        """
        Register a custom metric function.
        
        Args:
            name: Name of the metric
            metric_fn: Function implementing the metric
            description: Optional description of what the metric does
        """
        cls._metrics[name] = {
            "fn": metric_fn,
            "description": description
        }
    
    @classmethod
    def get(cls, name: str) -> Callable:
        """
        Get a registered metric function by name.
        
        Args:
            name: Name of the registered metric
            
        Returns:
            The metric function
            
        Raises:
            KeyError: If the metric is not registered
        """
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' is not registered")
        
        return cls._metrics[name]["fn"]
    
    @classmethod
    def list_metrics(cls) -> Dict[str, str]:
        """
        List all registered metrics and their descriptions.
        
        Returns:
            Dictionary mapping metric names to descriptions
        """
        return {name: info["description"] for name, info in cls._metrics.items()}
    
    @classmethod
    def remove(cls, name: str) -> None:
        """
        Remove a registered metric.
        
        Args:
            name: Name of the metric to remove
            
        Raises:
            KeyError: If the metric is not registered
        """
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' is not registered")
        
        del cls._metrics[name]


# Register built-in metrics
MetricRegistry.register(
    "accuracy",
    accuracy,
    "Calculate accuracy for classification tasks"
)

MetricRegistry.register(
    "precision_recall_f1",
    precision_recall_f1,
    "Calculate precision, recall, and F1 score for classification"
)

MetricRegistry.register(
    "confusion_matrix",
    confusion_matrix,
    "Generate confusion matrix and per-class metrics"
)

MetricRegistry.register(
    "text_similarity",
    text_similarity,
    "Calculate similarity between two text strings"
)

MetricRegistry.register(
    "llm_evaluation",
    llm_output_evaluation,
    "Comprehensive evaluation of LLM output"
)

MetricRegistry.register(
    "regression_metrics",
    regression_metrics,
    "Calculate metrics for regression tasks (RMSE, MAE, R2)"
) 