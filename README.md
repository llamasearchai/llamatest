# LlamaTest

A testing framework for AI and LLM applications. LlamaTest provides a comprehensive set of tools for creating, running, and analyzing tests for AI models, with a focus on language models and other generative AI.

## Features

- **Test Cases and Suites**: Create structured tests for your AI models and organize them into test suites
- **Specialized Assertions**: AI-specific assertions for text similarity, semantic equivalence, and model performance
- **Evaluation Metrics**: Comprehensive metrics for evaluating model outputs 
- **Datasets**: Tools for loading, managing, and transforming test datasets
- **Test Generators**: Generate test cases from templates, existing examples, or parameter combinations

## Installation

### Basic Installation

```bash
pip install llamatest
```

### With Optional Dependencies

```bash
# For NLP-specific features
pip install llamatest[nlp]

# For development
pip install llamatest[dev]

# For documentation
pip install llamatest[docs]
```

## Quick Start

### Creating a Simple Test

```python
from llamatest import TestCase, TestSuite, assert_text_similarity

# Create a test case for an LLM
test_case = TestCase(
    input="What is the capital of France?",
    expected_output="Paris",
    name="Capital of France test"
)

# Create a test suite
suite = TestSuite(name="Geography Questions")
suite.add_test(test_case)

# Run the test with a model
def my_model(input_text):
    # Replace with your actual model call
    return "Paris is the capital of France"

result = suite.run(function=my_model)

# Print the test results
print(f"Tests passed: {result.passed_tests}/{result.total_tests}")
```

### Using Text Similarity for Evaluation

```python
from llamatest import TestCase, assert_text_similarity

test_case = TestCase(
    input="Summarize this paragraph about climate change.",
    expected_output="Climate change poses significant global risks.",
    name="Summary test"
)

def evaluate_result(actual_output, expected_output):
    # Use similarity instead of exact matching
    return assert_text_similarity(
        expected_output, 
        actual_output,
        threshold=0.7,
        method="cosine"
    )

# Passing a custom evaluation function
result = test_case.run(function=my_model, eval_func=evaluate_result)
```

### Generating Test Cases from Templates

```python
from llamatest import TemplateGenerator, random_int, random_choice

# Create a template generator
generator = TemplateGenerator(
    input_template="What is {number1} {operation} {number2}?",
    expected_output_template="{answer}",
    providers={
        "number1": random_int(1, 100),
        "number2": random_int(1, 100),
        "operation": random_choice(["+", "-", "*", "/"]),
        "answer": lambda: eval(f"{values['number1']} {values['operation']} {values['number2']}")
    }
)

# Generate 10 test cases
test_cases = generator.generate(count=10)

# Add to a test suite
suite = TestSuite(name="Arithmetic Tests")
suite.add_tests(test_cases)
```

## Advanced Usage

### Working with Datasets

```python
from llamatest import Dataset, TextClassificationDataset

# Load a dataset from a CSV file
dataset = Dataset.load("test_data.csv")

# Split into training and testing
train_dataset, test_dataset = dataset.split(ratio=0.8, seed=42)

# Convert to test cases
test_cases = test_dataset.to_test_cases()
```

### Creating Custom Metrics

```python
from llamatest import MetricRegistry

# Define a custom metric
def semantic_precision(expected_outputs, actual_outputs, threshold=0.8):
    # Implementation of custom semantic precision metric
    ...
    return precision_score

# Register the metric
MetricRegistry.register(
    "semantic_precision",
    semantic_precision,
    "Precision score based on semantic similarity"
)

# Use the metric in assertions
assert_model_performance(
    metrics={"semantic_precision": 0.85},
    thresholds={"semantic_precision": 0.8}
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
# Updated in commit 1 - 2025-04-04 17:30:29

# Updated in commit 9 - 2025-04-04 17:30:29

# Updated in commit 17 - 2025-04-04 17:30:29

# Updated in commit 25 - 2025-04-04 17:30:30

# Updated in commit 1 - 2025-04-05 14:35:26

# Updated in commit 9 - 2025-04-05 14:35:27

# Updated in commit 17 - 2025-04-05 14:35:27

# Updated in commit 25 - 2025-04-05 14:35:27

# Updated in commit 1 - 2025-04-05 15:21:56

# Updated in commit 9 - 2025-04-05 15:21:56

# Updated in commit 17 - 2025-04-05 15:21:56

# Updated in commit 25 - 2025-04-05 15:21:56

# Updated in commit 1 - 2025-04-05 15:56:15

# Updated in commit 9 - 2025-04-05 15:56:15

# Updated in commit 17 - 2025-04-05 15:56:15

# Updated in commit 25 - 2025-04-05 15:56:16

# Updated in commit 1 - 2025-04-05 17:01:38

# Updated in commit 9 - 2025-04-05 17:01:38

# Updated in commit 17 - 2025-04-05 17:01:39

# Updated in commit 25 - 2025-04-05 17:01:39

# Updated in commit 1 - 2025-04-05 17:33:39

# Updated in commit 9 - 2025-04-05 17:33:39

# Updated in commit 17 - 2025-04-05 17:33:39

# Updated in commit 25 - 2025-04-05 17:33:39

# Updated in commit 1 - 2025-04-05 18:20:15

# Updated in commit 9 - 2025-04-05 18:20:15

# Updated in commit 17 - 2025-04-05 18:20:16
