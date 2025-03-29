"""
TestResult class for representing the results of a test run.
"""
from typing import List, Dict, Any, Optional, Union
import json
from datetime import datetime

from .report import Report, ReportFormat


class TestResult:
    """
    Represents the results of a test run, including individual test case results.
    
    TestResult stores the outcomes of a TestSuite run, including aggregate statistics,
    individual test case results, and metadata about the test run.
    """
    
    def __init__(
        self,
        suite_id: str,
        suite_name: str,
        test_results: List[Dict[str, Any]],
        duration: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new test result.
        
        Args:
            suite_id: ID of the test suite that was run
            suite_name: Name of the test suite that was run
            test_results: List of individual test case results
            duration: Total run time in seconds
            timestamp: When the test was run (defaults to now)
            metadata: Optional additional metadata about the test run
        """
        self.suite_id = suite_id
        self.suite_name = suite_name
        self.test_results = test_results
        self.duration = duration
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        
        # Calculate aggregate statistics
        self.total_tests = len(test_results)
        self.passed_tests = sum(1 for r in test_results if r.get('passed', False))
        self.failed_tests = self.total_tests - self.passed_tests
        self.pass_rate = self.passed_tests / self.total_tests if self.total_tests > 0 else 0
    
    def get_passed_tests(self) -> List[Dict[str, Any]]:
        """
        Get all passing test results.
        
        Returns:
            List of test results that passed
        """
        return [r for r in self.test_results if r.get('passed', False)]
    
    def get_failed_tests(self) -> List[Dict[str, Any]]:
        """
        Get all failing test results.
        
        Returns:
            List of test results that failed
        """
        return [r for r in self.test_results if not r.get('passed', False)]
    
    def get_test_by_id(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific test result by ID.
        
        Args:
            test_id: ID of the test to retrieve
            
        Returns:
            Test result or None if not found
        """
        for result in self.test_results:
            if result.get('test_id') == test_id:
                return result
        return None
    
    def get_test_by_name(self, test_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific test result by name.
        
        Args:
            test_name: Name of the test to retrieve
            
        Returns:
            Test result or None if not found
        """
        for result in self.test_results:
            if result.get('test_name') == test_name:
                return result
        return None
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the test results.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            'suite_id': self.suite_id,
            'suite_name': self.suite_name,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'pass_rate': self.pass_rate,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the test result to a dictionary.
        
        Returns:
            Dictionary representation of the test result
        """
        return {
            'suite_id': self.suite_id,
            'suite_name': self.suite_name,
            'test_results': self.test_results,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'pass_rate': self.pass_rate,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert the test result to a JSON string.
        
        Args:
            indent: Number of spaces for indentation
            
        Returns:
            JSON string representation of the test result
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, file_path: str) -> None:
        """
        Save the test result to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    def generate_report(
        self, 
        format: Union[str, ReportFormat] = ReportFormat.MARKDOWN,
        output_path: Optional[str] = None
    ) -> Report:
        """
        Generate a report from the test results.
        
        Args:
            format: Output format (markdown, html, json, etc.)
            output_path: Optional path to save the report
            
        Returns:
            Report object containing the formatted report
        """
        if isinstance(format, str):
            format = ReportFormat(format)
            
        report = Report(
            results=self,
            format=format
        )
        
        if output_path:
            report.save(output_path)
            
        return report
    
    def __str__(self) -> str:
        """Return a string representation of the test result."""
        return (f"TestResult(suite='{self.suite_name}', "
                f"passed={self.passed_tests}/{self.total_tests}, "
                f"rate={self.pass_rate:.2%})")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestResult':
        """
        Create a TestResult instance from a dictionary.
        
        Args:
            data: Dictionary representation of a test result
            
        Returns:
            TestResult instance
        """
        # Convert timestamp string back to datetime object
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            
        return cls(
            suite_id=data['suite_id'],
            suite_name=data['suite_name'],
            test_results=data['test_results'],
            duration=data['duration'],
            timestamp=data.get('timestamp'),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TestResult':
        """
        Create a TestResult instance from a JSON string.
        
        Args:
            json_str: JSON string representation of a test result
            
        Returns:
            TestResult instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'TestResult':
        """
        Create a TestResult instance from a JSON file.
        
        Args:
            file_path: Path to a JSON file containing test results
            
        Returns:
            TestResult instance
        """
        with open(file_path, 'r') as f:
            return cls.from_json(f.read()) 