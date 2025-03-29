"""
Example demonstrating test generation capabilities.

This example shows how to generate test cases using different generators:
- Template-based generation
- Data augmentation
- Combination generation
- Permutation generation
"""
from llamatest import (
    TestSuite,
    TemplateGenerator,
    DataAugmentationGenerator,
    CombinationGenerator,
    PermutationGenerator,
    random_int,
    random_choice,
    random_string,
    synonym_replacement,
    add_typos,
    random_insertion,
    json_permutations
)


def calculate(expression):
    """Simple calculator function to evaluate a string expression."""
    try:
        return eval(expression)
    except Exception as e:
        return f"Error: {str(e)}"


def classify_sentiment(text):
    """Mock sentiment classifier for demonstration."""
    positive_words = ["good", "great", "excellent", "happy", "positive", "wonderful"]
    negative_words = ["bad", "terrible", "awful", "sad", "negative", "horrible"]
    
    pos_count = sum(word in text.lower() for word in positive_words)
    neg_count = sum(word in text.lower() for word in negative_words)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def sort_items(items):
    """Function that sorts a list of items."""
    return sorted(items)


def main():
    """Demonstrate various test generation approaches."""
    print("=== LlamaTest Generator Examples ===\n")
    
    # 1. Template-based test generation
    print("1. Template-based Test Generation")
    template_gen = TemplateGenerator(
        input_template="{num1} {operator} {num2}",
        expected_output_template="{result}",
        providers={
            "num1": random_int(1, 100),
            "num2": random_int(1, 100),
            "operator": random_choice(["+", "-", "*", "/"]),
            "result": lambda: None  # Will be filled in later
        },
        name="Math Expression Generator"
    )
    
    # Generate the test cases
    template_tests = template_gen.generate(count=5)
    
    # Fill in the expected results based on the inputs
    for test in template_tests:
        # Extract values from the test metadata
        values = test.metadata.get("template_values", {})
        num1 = values.get("num1")
        num2 = values.get("num2")
        operator = values.get("operator")
        
        # Calculate the expected result
        if operator == "/" and num2 == 0:
            # Avoid division by zero
            test.expected_output = "Error: division by zero"
        else:
            expression = f"{num1} {operator} {num2}"
            test.expected_output = eval(expression)
    
    # Create and run the test suite
    template_suite = TestSuite(name="Template Generated Math Tests")
    template_suite.add_tests(template_tests)
    
    print(f"Generated {len(template_tests)} tests from template")
    template_results = template_suite.run(function=calculate)
    print(f"Results: {template_results.passed_tests}/{template_results.total_tests} passed")
    
    # 2. Data augmentation test generation
    print("\n2. Data Augmentation Test Generation")
    
    # Base examples for sentiment classification
    base_examples = [
        {"input": "I had a great day today.", "expected_output": "positive"},
        {"input": "This movie was terrible, I hated it.", "expected_output": "negative"},
        {"input": "The weather is neither good nor bad.", "expected_output": "neutral"}
    ]
    
    # Create generator with text transformations
    augmentation_gen = DataAugmentationGenerator(
        base_examples=base_examples,
        transformations=[
            synonym_replacement,
            add_typos,
            random_insertion
        ],
        preserve_output=True,  # Keep the same expected outputs
        name="Sentiment Augmentation Generator"
    )
    
    # Generate augmented test cases
    augmented_tests = augmentation_gen.generate(count=6)
    
    # Create and run the test suite
    augmentation_suite = TestSuite(name="Augmented Sentiment Tests")
    augmentation_suite.add_tests(augmented_tests)
    
    print(f"Generated {len(augmented_tests)} tests via data augmentation")
    aug_results = augmentation_suite.run(function=classify_sentiment)
    print(f"Results: {aug_results.passed_tests}/{aug_results.total_tests} passed")
    
    # 3. Combination test generation
    print("\n3. Combination Test Generation")
    
    # Define parameters for combinations
    parameters = {
        "product": ["ProductA", "ProductB", "ProductC"],
        "quantity": [1, 10, 100],
        "discount": [0, 0.1, 0.5]
    }
    
    # Define input and output formatters
    def input_formatter(params):
        return f"Calculate price for {params['quantity']} units of {params['product']} with {params['discount']*100}% discount"
    
    def output_formatter(params):
        # Mock price calculation
        base_prices = {"ProductA": 10, "ProductB": 20, "ProductC": 30}
        price = base_prices[params["product"]] * params["quantity"] * (1 - params["discount"])
        return price
    
    # Create combination generator
    combo_gen = CombinationGenerator(
        parameters=parameters,
        input_formatter=input_formatter,
        output_formatter=output_formatter,
        name="Product Price Generator"
    )
    
    # Generate all combinations (or specify a count)
    combo_tests = combo_gen.generate(count=10)
    
    print(f"Generated {len(combo_tests)} tests from parameter combinations")
    
    # 4. Permutation test generation
    print("\n4. Permutation Test Generation")
    
    # Base input to permute
    base_input = [5, 2, 9, 1, 7]
    
    # Create permutation generator
    perm_gen = PermutationGenerator(
        base_input=base_input,
        permutation_function=lambda x: json_permutations(x, max_count=10),
        expected_output=sorted(base_input),  # All permutations should sort to the same result
        name="Sort Permutation Generator"
    )
    
    # Generate permutations
    perm_tests = perm_gen.generate(count=5)
    
    # Create and run the test suite
    perm_suite = TestSuite(name="Permutation Sort Tests")
    perm_suite.add_tests(perm_tests)
    
    print(f"Generated {len(perm_tests)} tests using permutations")
    perm_results = perm_suite.run(function=sort_items)
    print(f"Results: {perm_results.passed_tests}/{perm_results.total_tests} passed")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Template Generated Tests: {template_results.passed_tests}/{template_results.total_tests} passed")
    print(f"Augmentation Generated Tests: {aug_results.passed_tests}/{aug_results.total_tests} passed") 
    print(f"Permutation Generated Tests: {perm_results.passed_tests}/{perm_results.total_tests} passed")


if __name__ == "__main__":
    main() 