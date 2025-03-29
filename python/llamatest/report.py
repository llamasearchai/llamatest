"""
Report class for generating formatted reports from test results.
"""
from typing import Dict, Any, Optional, Union
import json
import enum
import os
from datetime import datetime


class ReportFormat(enum.Enum):
    """Enum for supported report formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    TEXT = "text"


class Report:
    """
    Generates formatted reports from test results.
    
    The Report class takes test results and formats them into
    various output formats like Markdown, HTML, JSON, etc.
    """
    
    def __init__(
        self,
        results,  # TestResult, can't type hint due to circular import
        format: ReportFormat = ReportFormat.MARKDOWN,
        title: Optional[str] = None,
        suite = None  # TestSuite, can't type hint due to circular import
    ):
        """
        Initialize a new report.
        
        Args:
            results: TestResult object with the results to report
            format: Output format (markdown, html, json, etc.)
            title: Optional title for the report
            suite: Optional TestSuite object for additional context
        """
        self.results = results
        self.format = format
        self.title = title or f"Test Report: {results.suite_name}"
        self.suite = suite
        self.generated_at = datetime.now()
        
        # Generate the report content based on the format
        self.content = self._generate()
    
    def _generate(self) -> str:
        """
        Generate the report content based on the format.
        
        Returns:
            Formatted report content as a string
        """
        if self.format == ReportFormat.MARKDOWN:
            return self._generate_markdown()
        elif self.format == ReportFormat.HTML:
            return self._generate_html()
        elif self.format == ReportFormat.JSON:
            return self._generate_json()
        elif self.format == ReportFormat.CSV:
            return self._generate_csv()
        elif self.format == ReportFormat.TEXT:
            return self._generate_text()
        else:
            raise ValueError(f"Unsupported report format: {self.format}")
    
    def _generate_markdown(self) -> str:
        """Generate a Markdown report."""
        lines = []
        
        # Title and summary
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"Generated at: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary statistics
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Suite Name**: {self.results.suite_name}")
        lines.append(f"- **Total Tests**: {self.results.total_tests}")
        lines.append(f"- **Passed**: {self.results.passed_tests}")
        lines.append(f"- **Failed**: {self.results.failed_tests}")
        lines.append(f"- **Pass Rate**: {self.results.pass_rate:.2%}")
        lines.append(f"- **Duration**: {self.results.duration:.3f} seconds")
        lines.append("")
        
        # Failed tests (if any)
        if self.results.failed_tests > 0:
            lines.append("## Failed Tests")
            lines.append("")
            lines.append("| Test Name | Error |")
            lines.append("|-----------|-------|")
            
            for result in self.results.get_failed_tests():
                name = result.get('test_name', 'Unnamed Test')
                error = result.get('error', 'No error message')
                lines.append(f"| {name} | {error} |")
            
            lines.append("")
        
        # All test results
        lines.append("## Test Results")
        lines.append("")
        lines.append("| Test Name | Status | Duration (s) |")
        lines.append("|-----------|--------|-------------|")
        
        for result in self.results.test_results:
            name = result.get('test_name', 'Unnamed Test')
            status = "✅ Pass" if result.get('passed', False) else "❌ Fail"
            duration = result.get('duration', 0)
            lines.append(f"| {name} | {status} | {duration:.3f} |")
        
        return "\n".join(lines)
    
    def _generate_html(self) -> str:
        """Generate an HTML report."""
        # Basic HTML with some styling
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <p>Generated at: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Suite Name:</strong> {self.results.suite_name}</p>
        <p><strong>Total Tests:</strong> {self.results.total_tests}</p>
        <p><strong>Passed:</strong> <span class="pass">{self.results.passed_tests}</span></p>
        <p><strong>Failed:</strong> <span class="fail">{self.results.failed_tests}</span></p>
        <p><strong>Pass Rate:</strong> {self.results.pass_rate:.2%}</p>
        <p><strong>Duration:</strong> {self.results.duration:.3f} seconds</p>
    </div>
"""
        
        # Failed tests section (if any)
        if self.results.failed_tests > 0:
            html += """    <h2>Failed Tests</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Error</th>
        </tr>
"""
            for result in self.results.get_failed_tests():
                name = result.get('test_name', 'Unnamed Test')
                error = result.get('error', 'No error message')
                html += f"""        <tr>
            <td>{name}</td>
            <td>{error}</td>
        </tr>
"""
            html += "    </table>\n"
        
        # All test results
        html += """    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Status</th>
            <th>Duration (s)</th>
        </tr>
"""
        for result in self.results.test_results:
            name = result.get('test_name', 'Unnamed Test')
            passed = result.get('passed', False)
            status = '<span class="pass">Pass</span>' if passed else '<span class="fail">Fail</span>'
            duration = result.get('duration', 0)
            html += f"""        <tr>
            <td>{name}</td>
            <td>{status}</td>
            <td>{duration:.3f}</td>
        </tr>
"""
        
        html += """    </table>
</body>
</html>"""
        
        return html
    
    def _generate_json(self) -> str:
        """Generate a JSON report."""
        report_data = {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "results": self.results.to_dict()
        }
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_csv(self) -> str:
        """Generate a CSV report."""
        lines = []
        
        # Header
        lines.append("test_name,status,duration,error")
        
        # Test results
        for result in self.results.test_results:
            name = result.get('test_name', 'Unnamed Test').replace(',', ' ')
            status = "PASS" if result.get('passed', False) else "FAIL"
            duration = result.get('duration', 0)
            error = result.get('error', '').replace(',', ' ').replace('\n', ' ')
            
            lines.append(f"{name},{status},{duration},{error}")
        
        return "\n".join(lines)
    
    def _generate_text(self) -> str:
        """Generate a plain text report."""
        lines = []
        
        # Title and summary
        lines.append(self.title)
        lines.append("=" * len(self.title))
        lines.append("")
        lines.append(f"Generated at: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary statistics
        lines.append("SUMMARY")
        lines.append("-------")
        lines.append(f"Suite Name: {self.results.suite_name}")
        lines.append(f"Total Tests: {self.results.total_tests}")
        lines.append(f"Passed: {self.results.passed_tests}")
        lines.append(f"Failed: {self.results.failed_tests}")
        lines.append(f"Pass Rate: {self.results.pass_rate:.2%}")
        lines.append(f"Duration: {self.results.duration:.3f} seconds")
        lines.append("")
        
        # Failed tests (if any)
        if self.results.failed_tests > 0:
            lines.append("FAILED TESTS")
            lines.append("-----------")
            
            for result in self.results.get_failed_tests():
                name = result.get('test_name', 'Unnamed Test')
                error = result.get('error', 'No error message')
                lines.append(f"{name}: {error}")
            
            lines.append("")
        
        # All test results
        lines.append("TEST RESULTS")
        lines.append("------------")
        
        for result in self.results.test_results:
            name = result.get('test_name', 'Unnamed Test')
            status = "PASS" if result.get('passed', False) else "FAIL"
            duration = result.get('duration', 0)
            lines.append(f"{name}: {status} ({duration:.3f}s)")
        
        return "\n".join(lines)
    
    def save(self, file_path: str) -> None:
        """
        Save the report to a file.
        
        Args:
            file_path: Path to save the report
        """
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'w') as f:
            f.write(self.content)
    
    def __str__(self) -> str:
        """Return the report content as a string."""
        return self.content 