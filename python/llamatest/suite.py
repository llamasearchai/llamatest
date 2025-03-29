"""
TestSuite class for organizing and running collections of test cases.
"""
from typing import List, Dict, Any, Callable, Optional, Union
import time
import uuid
from datetime import datetime

from .case import TestCase
from .result import TestResult
from .report import Report, ReportFormat


class TestSuite:
    """
    A collection of related test cases with shared configuration.
    
    TestSuite allows organizing tests into logical groups, running them,
    and analyzing the results.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a new test suite.
        
        Args:
            name: Name of the test suite
            description: Optional description
            tags: Optional list of tags for categorization
            metadata: Optional additional metadata
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.tags = tags or []
        self.metadata = metadata or {}
        self.test_cases: List[TestCase] = []
        self.created_at = datetime.now()
    
    def add_test(self, test_case: TestCase) -> None:
        """
        Add a test case to the suite.
        
        Args:
            test_case: The test case to add
        """
        self.test_cases.append(test_case)
    
    def add_tests(self, test_cases: List[TestCase]) -> None:
        """
        Add multiple test cases to the suite.
        
        Args:
            test_cases: List of test cases to add
        """
        self.test_cases.extend(test_cases)
    
    def run(
        self, 
        **kwargs
    ) -> TestResult:
        """
        Run all test cases in the suite.
        
        Any keyword arguments are passed to the test cases when they are executed.
        This can include functions to test, model references, etc.
        
        Returns:
            TestResult object containing the results of the test run
        """
        if not self.test_cases:
            raise ValueError("No test cases in the suite")
        
        start_time = time.time()
        test_results = []
        
        for test_case in self.test_cases:
            # Run the individual test case
            result = test_case.run(**kwargs)
            test_results.append(result)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Create a combined result for the entire suite
        suite_result = TestResult(
            suite_id=self.id,
            suite_name=self.name,
            test_results=test_results,
            duration=duration,
            timestamp=datetime.now()
        )
        
        return suite_result
    
    def generate_report(
        self, 
        results: TestResult,
        format: Union[str, ReportFormat] = ReportFormat.MARKDOWN,
        output_path: Optional[str] = None
    ) -> Report:
        """
        Generate a report from test results.
        
        Args:
            results: TestResult object from a test run
            format: Output format (markdown, html, json, etc.)
            output_path: Optional path to save the report
            
        Returns:
            Report object containing the formatted report
        """
        if isinstance(format, str):
            format = ReportFormat(format)
            
        report = Report(
            suite=self,
            results=results,
            format=format
        )
        
        if output_path:
            report.save(output_path)
            
        return report
    
    def filter_tests(self, 
                    tags: Optional[List[str]] = None, 
                    metadata_filter: Optional[Dict[str, Any]] = None) -> 'TestSuite':
        """
        Create a new test suite with filtered test cases.
        
        Args:
            tags: List of tags to filter by (test must have at least one)
            metadata_filter: Dict of metadata key/values to match
            
        Returns:
            A new TestSuite containing only the matching test cases
        """
        filtered_suite = TestSuite(
            name=f"{self.name} (filtered)",
            description=self.description,
            tags=self.tags,
            metadata=self.metadata
        )
        
        for test_case in self.test_cases:
            # Filter by tags
            if tags and not any(tag in test_case.tags for tag in tags):
                continue
                
            # Filter by metadata
            if metadata_filter:
                metadata_match = True
                for key, value in metadata_filter.items():
                    if key not in test_case.metadata or test_case.metadata[key] != value:
                        metadata_match = False
                        break
                if not metadata_match:
                    continue
            
            # If we got here, the test passed all filters
            filtered_suite.add_test(test_case)
            
        return filtered_suite
    
    def __len__(self) -> int:
        """Return the number of test cases in the suite."""
        return len(self.test_cases)
    
    def __str__(self) -> str:
        """Return a string representation of the test suite."""
        return f"TestSuite(name='{self.name}', tests={len(self.test_cases)})" 