"""
Simple example demonstrating basic usage of LlamaTest.

This example creates a test suite with a few simple test cases
for a basic calculator function.
"""
from llamatest import TestCase, TestSuite, TestResult, assert_equal


def add(a, b):
    """Simple function to test: adds two numbers."""
    return a + b


def multiply(a, b):
    """Simple function to test: multiplies two numbers."""
    return a * b


def main():
    """Run a simple test suite against our functions."""
    # Create test cases
    test_case1 = TestCase(
        input=(5, 3),
        expected_output=8,
        name="Addition test 1"
    )
    
    test_case2 = TestCase(
        input=(-2, 7),
        expected_output=5,
        name="Addition test 2"
    )
    
    test_case3 = TestCase(
        input=(4, 6),
        expected_output=24,
        name="Multiplication test 1"
    )
    
    # Create a test suite for addition
    addition_suite = TestSuite(
        name="Addition Tests",
        description="Tests for the add function"
    )
    addition_suite.add_tests([test_case1, test_case2])
    
    # Create a test suite for multiplication
    multiplication_suite = TestSuite(
        name="Multiplication Tests",
        description="Tests for the multiply function"
    )
    multiplication_suite.add_test(test_case3)
    
    # Run the addition test suite
    print("Running addition tests...")
    addition_result = addition_suite.run(function=add)
    print(f"Addition tests: {addition_result.passed_tests}/{addition_result.total_tests} passed")
    
    # Run the multiplication test suite
    print("\nRunning multiplication tests...")
    multiplication_result = multiplication_suite.run(function=multiply)
    print(f"Multiplication tests: {multiplication_result.passed_tests}/{multiplication_result.total_tests} passed")
    
    # Generate and print reports
    print("\n=== Addition Test Results ===")
    report = addition_result.generate_report()
    print(report)
    
    print("\n=== Multiplication Test Results ===")
    report = multiplication_result.generate_report()
    print(report)


if __name__ == "__main__":
    main() 