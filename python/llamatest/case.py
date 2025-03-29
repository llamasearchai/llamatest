"""
TestCase class for defining individual test cases.
"""
from typing import Dict, Any, List, Optional, Callable, Union
import time
import uuid
from datetime import datetime

from .result import TestResult


class TestCase:
    """
    An individual test case with input, expected output, and metadata.
    
    TestCase represents a single unit of testing, including what to test,
    what the expected results are, and any additional metadata.
    """
    
    def __init__(
        self,
        input: Any,
        expected_output: Any = None,
        name: Optional[str] = None,
        description: str = "",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        timeout: Optional[float] = None,
        # For similarity-based test cases
        expected_similar: List[Any] = None,
        expected_different: List[Any] = None
    ):
        """
        Initialize a new test case.
        
        Args:
            input: The input to the function or model being tested
            expected_output: The expected output (for exact comparisons)
            name: Optional name for the test case
            description: Optional description
            tags: Optional list of tags for categorization
            metadata: Optional additional metadata
            timeout: Optional timeout in seconds
            expected_similar: Optional list of inputs that should be similar to the input
            expected_different: Optional list of inputs that should be different from the input
        """
        self.id = str(uuid.uuid4())
        self.input = input
        self.expected_output = expected_output
        self.name = name or f"TestCase-{self.id[:8]}"
        self.description = description
        self.tags = tags or []
        self.metadata = metadata or {}
        self.timeout = timeout
        self.expected_similar = expected_similar or []
        self.expected_different = expected_different or []
        self.created_at = datetime.now()
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the test case using the provided functions.
        
        This method should be overridden in subclasses to implement
        specific testing logic. The base implementation handles common
        patterns like testing a function with the input and comparing
        to expected output.
        
        Args:
            **kwargs: Functions and parameters needed to run the test.
                     Typically includes at least a 'function' or 'model'
                     parameter that will be called with the test input.
        
        Returns:
            Dictionary containing test results
        """
        start_time = time.time()
        
        # Extract the function to test from kwargs
        func = kwargs.get('function')
        if not func and 'model' in kwargs:
            func = kwargs['model']
        
        if not callable(func):
            raise ValueError("No callable function or model provided in kwargs")
        
        # Run the function with the input
        try:
            actual_output = func(self.input)
            passed = self._evaluate_result(actual_output, **kwargs)
            error = None
        except Exception as e:
            actual_output = None
            passed = False
            error = str(e)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Build the result dictionary
        result = {
            'test_id': self.id,
            'test_name': self.name,
            'input': self.input,
            'expected_output': self.expected_output,
            'actual_output': actual_output,
            'passed': passed,
            'error': error,
            'duration': duration,
            'timestamp': datetime.now()
        }
        
        return result
    
    def _evaluate_result(self, actual_output: Any, **kwargs) -> bool:
        """
        Evaluate if the actual output matches the expected output.
        
        Can be overridden in subclasses to implement custom comparison logic.
        
        Args:
            actual_output: The actual output from running the test
            **kwargs: Additional parameters for evaluation
        
        Returns:
            Boolean indicating if the test passed
        """
        # Handle similarity-based tests if comparison function is provided
        if (self.expected_similar or self.expected_different) and 'similarity_function' in kwargs:
            similarity_func = kwargs['similarity_function']
            similarity_threshold = kwargs.get('similarity_threshold', 0.7)
            
            # Test similar items
            for similar_item in self.expected_similar:
                similarity = similarity_func(self.input, similar_item)
                if similarity < similarity_threshold:
                    return False
            
            # Test different items
            for different_item in self.expected_different:
                similarity = similarity_func(self.input, different_item)
                if similarity >= similarity_threshold:
                    return False
            
            return True
        
        # Handle regular output comparison
        if self.expected_output is not None:
            # Default to exact comparison
            return actual_output == self.expected_output
        
        # If no expected output was defined, and no errors occurred,
        # consider the test passed (useful for testing that a function runs without error)
        return True
    
    def __str__(self) -> str:
        """Return a string representation of the test case."""
        return f"TestCase(name='{self.name}')" 