"""
Assertions module for specialized AI testing assertions.

This module provides assertion functions for validating AI model outputs,
including text similarity, semantic equivalence, and model performance metrics.
"""
from typing import Any, List, Dict, Optional, Union, Callable
import re
import math
from .metrics import text_similarity


def assert_equal(expected: Any, actual: Any, msg: Optional[str] = None) -> bool:
    """
    Assert that two values are equal.
    
    Args:
        expected: Expected value
        actual: Actual value
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if expected != actual:
        error_msg = msg or f"Expected {expected!r}, but got {actual!r}"
        raise AssertionError(error_msg)
    return True


def assert_not_equal(expected: Any, actual: Any, msg: Optional[str] = None) -> bool:
    """
    Assert that two values are not equal.
    
    Args:
        expected: Expected value to differ from
        actual: Actual value
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if expected == actual:
        error_msg = msg or f"Expected {actual!r} to be different from {expected!r}"
        raise AssertionError(error_msg)
    return True


def assert_true(condition: bool, msg: Optional[str] = None) -> bool:
    """
    Assert that a condition is True.
    
    Args:
        condition: Condition to check
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if not condition:
        error_msg = msg or "Expected condition to be True, but got False"
        raise AssertionError(error_msg)
    return True


def assert_false(condition: bool, msg: Optional[str] = None) -> bool:
    """
    Assert that a condition is False.
    
    Args:
        condition: Condition to check
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if condition:
        error_msg = msg or "Expected condition to be False, but got True"
        raise AssertionError(error_msg)
    return True


def assert_in(item: Any, container: Any, msg: Optional[str] = None) -> bool:
    """
    Assert that an item is in a container.
    
    Args:
        item: Item to check for
        container: Container to check in
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if item not in container:
        error_msg = msg or f"Expected {item!r} to be in {container!r}"
        raise AssertionError(error_msg)
    return True


def assert_not_in(item: Any, container: Any, msg: Optional[str] = None) -> bool:
    """
    Assert that an item is not in a container.
    
    Args:
        item: Item to check for
        container: Container to check in
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if item in container:
        error_msg = msg or f"Expected {item!r} not to be in {container!r}"
        raise AssertionError(error_msg)
    return True


def assert_almost_equal(
    expected: Union[float, List[float]],
    actual: Union[float, List[float]],
    tolerance: float = 1e-7,
    msg: Optional[str] = None
) -> bool:
    """
    Assert that two floating point values or lists are almost equal.
    
    Args:
        expected: Expected value(s)
        actual: Actual value(s)
        tolerance: Tolerance for equality
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            error_msg = msg or f"Lists have different lengths: {len(expected)} != {len(actual)}"
            raise AssertionError(error_msg)
        
        for i, (e, a) in enumerate(zip(expected, actual)):
            if abs(e - a) > tolerance:
                error_msg = msg or f"Lists differ at index {i}: {e} != {a} (tolerance: {tolerance})"
                raise AssertionError(error_msg)
    else:
        if abs(expected - actual) > tolerance:
            error_msg = msg or f"Expected {expected} to be almost equal to {actual} (tolerance: {tolerance})"
            raise AssertionError(error_msg)
    
    return True


def assert_text_contains(text: str, substring: str, case_sensitive: bool = True, msg: Optional[str] = None) -> bool:
    """
    Assert that a text contains a substring.
    
    Args:
        text: Text to check
        substring: Substring to look for
        case_sensitive: Whether the check should be case sensitive
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if not case_sensitive:
        text = text.lower()
        substring = substring.lower()
    
    if substring not in text:
        error_msg = msg or f"Expected text to contain substring '{substring}', but it does not"
        raise AssertionError(error_msg)
    
    return True


def assert_text_similarity(
    expected: str,
    actual: str,
    threshold: float = 0.7,
    method: str = "jaccard",
    msg: Optional[str] = None
) -> bool:
    """
    Assert that two texts are similar according to a similarity metric.
    
    Args:
        expected: Expected text
        actual: Actual text
        threshold: Minimum similarity threshold (0.0 to 1.0)
        method: Similarity method ('jaccard', 'cosine', 'levenshtein')
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    similarity = text_similarity(expected, actual, method)
    
    if similarity < threshold:
        error_msg = msg or (
            f"Text similarity ({method}) is {similarity:.2f}, "
            f"which is below the threshold of {threshold:.2f}"
        )
        raise AssertionError(error_msg)
    
    return True


def assert_matches_regex(text: str, pattern: str, msg: Optional[str] = None) -> bool:
    """
    Assert that a text matches a regular expression pattern.
    
    Args:
        text: Text to check
        pattern: Regular expression pattern
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if not re.search(pattern, text):
        error_msg = msg or f"Text does not match pattern '{pattern}'"
        raise AssertionError(error_msg)
    
    return True


def assert_not_matches_regex(text: str, pattern: str, msg: Optional[str] = None) -> bool:
    """
    Assert that a text does not match a regular expression pattern.
    
    Args:
        text: Text to check
        pattern: Regular expression pattern
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if re.search(pattern, text):
        error_msg = msg or f"Text matches pattern '{pattern}' when it should not"
        raise AssertionError(error_msg)
    
    return True


def assert_json_structure(
    data: Dict[str, Any],
    expected_structure: Dict[str, Any],
    msg: Optional[str] = None
) -> bool:
    """
    Assert that a JSON/dict has the expected structure.
    
    Args:
        data: Data to check
        expected_structure: Expected structure with types
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    def _check_structure(actual, expected, path=""):
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False, f"{path}: Expected a dictionary, got {type(actual).__name__}"
            
            for key, exp_value in expected.items():
                if key not in actual:
                    return False, f"{path}.{key}: Key is missing"
                
                act_value = actual[key]
                result, error = _check_structure(act_value, exp_value, f"{path}.{key}")
                if not result:
                    return False, error
            
            return True, ""
        
        elif isinstance(expected, list):
            if not isinstance(actual, list):
                return False, f"{path}: Expected a list, got {type(actual).__name__}"
            
            if not expected:  # Empty list, no type checking
                return True, ""
            
            # Check first item's type against all items
            expected_type = expected[0]
            for i, item in enumerate(actual):
                result, error = _check_structure(item, expected_type, f"{path}[{i}]")
                if not result:
                    return False, error
            
            return True, ""
        
        elif isinstance(expected, type):
            if not isinstance(actual, expected):
                return False, f"{path}: Expected type {expected.__name__}, got {type(actual).__name__}"
            
            return True, ""
        
        else:
            return True, ""  # Exact value checking not implemented
    
    result, error = _check_structure(data, expected_structure)
    if not result:
        error_msg = msg or f"JSON structure validation failed: {error}"
        raise AssertionError(error_msg)
    
    return True


def assert_greater_than(actual: Union[int, float], value: Union[int, float], msg: Optional[str] = None) -> bool:
    """
    Assert that a value is greater than another value.
    
    Args:
        actual: Actual value
        value: Value to compare against
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if not actual > value:
        error_msg = msg or f"Expected {actual} to be greater than {value}"
        raise AssertionError(error_msg)
    
    return True


def assert_less_than(actual: Union[int, float], value: Union[int, float], msg: Optional[str] = None) -> bool:
    """
    Assert that a value is less than another value.
    
    Args:
        actual: Actual value
        value: Value to compare against
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if not actual < value:
        error_msg = msg or f"Expected {actual} to be less than {value}"
        raise AssertionError(error_msg)
    
    return True


def assert_length(obj: Union[List, Dict, str], length: int, msg: Optional[str] = None) -> bool:
    """
    Assert that an object has a specific length.
    
    Args:
        obj: Object to check
        length: Expected length
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    actual_length = len(obj)
    if actual_length != length:
        error_msg = msg or f"Expected length {length}, but got {actual_length}"
        raise AssertionError(error_msg)
    
    return True


def assert_model_performance(
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
    msg: Optional[str] = None
) -> bool:
    """
    Assert that model performance metrics meet specified thresholds.
    
    Args:
        metrics: Dictionary of metric names to values
        thresholds: Dictionary of metric names to minimum acceptable values
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    failures = []
    
    for metric_name, threshold in thresholds.items():
        if metric_name not in metrics:
            failures.append(f"Metric '{metric_name}' not found in provided metrics")
            continue
        
        actual_value = metrics[metric_name]
        if actual_value < threshold:
            failures.append(
                f"Metric '{metric_name}' value {actual_value:.4f} is below threshold {threshold:.4f}"
            )
    
    if failures:
        error_msg = msg or f"Model performance assertion failed:\n" + "\n".join(failures)
        raise AssertionError(error_msg)
    
    return True


def assert_no_runtime_errors(
    func: Callable,
    *args,
    expected_exceptions: Optional[List[type]] = None,
    **kwargs
) -> bool:
    """
    Assert that a function runs without raising unexpected exceptions.
    
    Args:
        func: Function to run
        *args: Arguments to pass to the function
        expected_exceptions: Optional list of exception types that are acceptable
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    expected_exceptions = expected_exceptions or []
    
    try:
        func(*args, **kwargs)
    except Exception as e:
        if not any(isinstance(e, exc_type) for exc_type in expected_exceptions):
            error_msg = f"Function raised unexpected exception: {type(e).__name__}: {e}"
            raise AssertionError(error_msg) from e
    
    return True


def assert_not_nan(value: float, msg: Optional[str] = None) -> bool:
    """
    Assert that a value is not NaN (not a number).
    
    Args:
        value: Value to check
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
    """
    if math.isnan(value):
        error_msg = msg or f"Expected value not to be NaN"
        raise AssertionError(error_msg)
    
    return True


def assert_text_format(text: str, format_type: str, msg: Optional[str] = None) -> bool:
    """
    Assert that text conforms to a specific format.
    
    Args:
        text: Text to check
        format_type: Format type to check ('json', 'xml', 'yaml', 'email', etc.)
        msg: Optional message for assertion failure
        
    Returns:
        True if assertion passes, raises AssertionError otherwise
        
    Raises:
        AssertionError: If the assertion fails
        ValueError: If format_type is not supported
    """
    # Define format validators
    format_validators = {
        "json": lambda t: _is_valid_json(t),
        "xml": lambda t: _is_valid_xml(t),
        "email": lambda t: bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", t)),
        "url": lambda t: bool(re.match(r"^(https?|ftp)://[^\s/$.?#].[^\s]*$", t)),
        "date": lambda t: bool(re.match(r"^\d{4}-\d{2}-\d{2}$", t)),
        "time": lambda t: bool(re.match(r"^\d{2}:\d{2}(:\d{2})?$", t)),
        "datetime": lambda t: bool(re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$", t)),
    }
    
    if format_type not in format_validators:
        raise ValueError(f"Unsupported format type: {format_type}")
    
    if not format_validators[format_type](text):
        error_msg = msg or f"Text does not conform to {format_type} format"
        raise AssertionError(error_msg)
    
    return True


def _is_valid_json(text: str) -> bool:
    """Check if text is valid JSON."""
    import json
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def _is_valid_xml(text: str) -> bool:
    """Check if text is valid XML."""
    import xml.etree.ElementTree as ET
    try:
        ET.fromstring(text)
        return True
    except Exception:
        return False 