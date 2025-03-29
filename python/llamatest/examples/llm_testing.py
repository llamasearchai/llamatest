"""
Example of testing LLMs using LlamaTest.

This example demonstrates how to test language model outputs using
text similarity metrics rather than exact matching.
"""
from llamatest import (
    TestCase, 
    TestSuite, 
    assert_text_similarity, 
    llm_output_evaluation,
    metrics
)


# This would be your actual LLM implementation
def mock_llm(prompt):
    """Mock LLM that returns predefined responses for demo purposes."""
    responses = {
        "What is the capital of France?": "Paris is the capital of France.",
        "Summarize climate change briefly": "Climate change refers to long-term shifts in temperatures and weather patterns, primarily caused by human activities, especially the burning of fossil fuels.",
        "How does photosynthesis work?": "Photosynthesis is the process where plants convert light energy into chemical energy. Plants use sunlight, water and carbon dioxide to produce oxygen and glucose.",
    }
    
    # Return closest predefined response or a default
    for key, response in responses.items():
        if key.lower() in prompt.lower():
            return response
    
    return "I don't have a specific answer for that question."


def evaluate_llm_output(expected, actual):
    """
    Custom evaluation function for LLM outputs.
    Uses text similarity rather than exact matching.
    """
    # Comprehensive evaluation with multiple metrics
    eval_results = llm_output_evaluation(
        expected, 
        actual, 
        metrics=["similarity", "content"]
    )
    
    # Consider the test passed if the overall score is above threshold
    passed = eval_results["overall_score"] >= 0.6
    
    # Return detailed results for reporting
    return {
        "passed": passed,
        "similarity": eval_results["text_similarity"],
        "keyword_presence": eval_results.get("keyword_presence", 0),
        "overall_score": eval_results["overall_score"],
    }


def main():
    """Run LLM tests with text similarity evaluation."""
    # Create test cases
    test_cases = [
        TestCase(
            input="What is the capital of France?",
            expected_output="The capital of France is Paris.",
            name="Capital question test"
        ),
        TestCase(
            input="Summarize climate change briefly",
            expected_output="Climate change is the long-term alteration of temperature and weather patterns caused primarily by human activities like burning fossil fuels.",
            name="Climate change summary test"
        ),
        TestCase(
            input="How does photosynthesis work?",
            expected_output="Photosynthesis is a process used by plants to convert light energy into chemical energy stored in glucose. Plants take in carbon dioxide and water and produce oxygen and sugar.",
            name="Photosynthesis explanation test"
        ),
    ]
    
    # Create test suite
    llm_suite = TestSuite(
        name="LLM Response Tests",
        description="Tests for language model responses using similarity metrics"
    )
    llm_suite.add_tests(test_cases)
    
    # Run the tests with custom evaluation
    print("Running LLM tests with similarity metrics...")
    results = llm_suite.run(
        function=mock_llm,
        eval_func=evaluate_llm_output
    )
    
    # Report results
    print(f"\nTests passed: {results.passed_tests}/{results.total_tests}")
    print(f"Pass rate: {results.pass_rate:.2%}")
    
    # Print detailed report
    report = results.generate_report()
    print("\n=== LLM Test Report ===")
    print(report)
    
    # Display detailed metrics for each test
    print("\n=== Detailed Metrics ===")
    for result in results.test_results:
        print(f"\nTest: {result['test_name']}")
        print(f"Expected: {result['expected_output']}")
        print(f"Actual: {result['actual_output']}")
        
        if isinstance(result.get('error'), dict):
            metrics = result['error']  # Our custom evaluator returns metrics here
            print(f"Text Similarity: {metrics.get('similarity', 0):.2f}")
            print(f"Keyword Presence: {metrics.get('keyword_presence', 0):.2f}")
            print(f"Overall Score: {metrics.get('overall_score', 0):.2f}")
        
        print(f"Passed: {result['passed']}")


if __name__ == "__main__":
    main() 