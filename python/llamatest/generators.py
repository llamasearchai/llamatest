"""
Generators module for programmatically generating test cases.

This module provides utilities for generating test cases for AI models,
including template-based generation, data augmentation, and synthetic
test case creation.
"""
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
import random
import string
import uuid
import itertools
import re
from datetime import datetime, timedelta

from .case import TestCase


class TestGenerator:
    """
    Base class for test case generators.
    
    TestGenerator provides common functionality for different types
    of test case generators.
    """
    
    def __init__(self, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new test generator.
        
        Args:
            name: Optional name for the generator
            metadata: Optional metadata for the generator
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"generator-{self.id[:8]}"
        self.metadata = metadata or {}
        self.created_at = datetime.now()
    
    def generate(self, count: int = 10) -> List[TestCase]:
        """
        Generate test cases.
        
        This method should be overridden by subclasses to implement
        specific generation logic.
        
        Args:
            count: Number of test cases to generate
            
        Returns:
            List of generated TestCase objects
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("Subclasses must implement generate method")


class TemplateGenerator(TestGenerator):
    """
    Generator that creates test cases from templates.
    
    Templates can contain placeholders that are replaced with
    values from specified providers or random values.
    """
    
    def __init__(
        self, 
        input_template: str,
        expected_output_template: Optional[str] = None,
        providers: Dict[str, Union[List[Any], Callable[[], Any]]] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a template generator.
        
        Args:
            input_template: Template string for generating inputs
            expected_output_template: Template string for generating expected outputs
            providers: Dictionary mapping placeholder names to value providers
            name: Optional name for the generator
            metadata: Optional metadata for the generator
        """
        super().__init__(name, metadata)
        self.input_template = input_template
        self.expected_output_template = expected_output_template
        self.providers = providers or {}
    
    def _fill_template(self, template: str, values: Dict[str, Any]) -> str:
        """
        Fill a template with provided values.
        
        Args:
            template: Template string with {placeholders}
            values: Dictionary of values to use for placeholders
            
        Returns:
            Filled template string
        """
        return template.format(**values)
    
    def _generate_values(self) -> Dict[str, Any]:
        """
        Generate a set of values for template placeholders.
        
        Returns:
            Dictionary of placeholder names to generated values
        """
        values = {}
        
        for key, provider in self.providers.items():
            if isinstance(provider, list):
                # If provider is a list, randomly select a value
                values[key] = random.choice(provider)
            elif callable(provider):
                # If provider is a function, call it to get a value
                values[key] = provider()
            else:
                # If provider is a single value, use it directly
                values[key] = provider
        
        return values
    
    def generate(self, count: int = 10) -> List[TestCase]:
        """
        Generate test cases using the template.
        
        Args:
            count: Number of test cases to generate
            
        Returns:
            List of generated TestCase objects
        """
        test_cases = []
        
        for i in range(count):
            # Generate random values for placeholders
            values = self._generate_values()
            
            # Generate input from template
            input_text = self._fill_template(self.input_template, values)
            
            # Generate expected output from template if provided
            expected_output = None
            if self.expected_output_template:
                expected_output = self._fill_template(self.expected_output_template, values)
            
            # Create test case
            test_case = TestCase(
                input=input_text,
                expected_output=expected_output,
                name=f"{self.name}_case_{i+1}",
                metadata={"generator_id": self.id, "template_values": values}
            )
            
            test_cases.append(test_case)
        
        return test_cases


class DataAugmentationGenerator(TestGenerator):
    """
    Generator that creates test cases by augmenting existing examples.
    
    Data augmentation applies transformations to existing examples
    to create new, similar test cases.
    """
    
    def __init__(
        self,
        base_examples: List[Dict[str, Any]],
        transformations: List[Callable[[str], str]],
        preserve_output: bool = True,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a data augmentation generator.
        
        Args:
            base_examples: List of base examples to augment
            transformations: List of transformation functions
            preserve_output: Whether to preserve the expected output
            name: Optional name for the generator
            metadata: Optional metadata for the generator
        """
        super().__init__(name, metadata)
        self.base_examples = base_examples
        self.transformations = transformations
        self.preserve_output = preserve_output
    
    def generate(self, count: Optional[int] = None) -> List[TestCase]:
        """
        Generate test cases by augmenting base examples.
        
        Args:
            count: Optional target number of test cases to generate
                  (may generate fewer if there are not enough combinations)
            
        Returns:
            List of generated TestCase objects
        """
        test_cases = []
        
        # If count is not specified, apply each transformation to each example
        if count is None:
            for i, example in enumerate(self.base_examples):
                input_text = example.get("input", "")
                expected_output = example.get("expected_output")
                
                for j, transform in enumerate(self.transformations):
                    # Apply transformation to input
                    augmented_input = transform(input_text)
                    
                    # Create test case
                    test_case = TestCase(
                        input=augmented_input,
                        expected_output=expected_output if self.preserve_output else None,
                        name=f"{self.name}_aug_{i+1}_{j+1}",
                        metadata={
                            "generator_id": self.id,
                            "base_example_id": i,
                            "transformation_id": j,
                            "original_input": input_text
                        }
                    )
                    
                    test_cases.append(test_case)
        else:
            # If count is specified, randomly sample transformations and examples
            for i in range(count):
                example = random.choice(self.base_examples)
                transform = random.choice(self.transformations)
                
                input_text = example.get("input", "")
                expected_output = example.get("expected_output")
                
                # Apply transformation to input
                augmented_input = transform(input_text)
                
                # Create test case
                test_case = TestCase(
                    input=augmented_input,
                    expected_output=expected_output if self.preserve_output else None,
                    name=f"{self.name}_aug_{i+1}",
                    metadata={
                        "generator_id": self.id,
                        "original_input": input_text
                    }
                )
                
                test_cases.append(test_case)
        
        return test_cases


class CombinationGenerator(TestGenerator):
    """
    Generator that creates test cases from combinations of values.
    
    Generates test cases by taking the cartesian product of specified
    parameter values and using these combinations to create inputs.
    """
    
    def __init__(
        self,
        parameters: Dict[str, List[Any]],
        input_formatter: Callable[[Dict[str, Any]], Any],
        output_formatter: Optional[Callable[[Dict[str, Any]], Any]] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a combination generator.
        
        Args:
            parameters: Dictionary mapping parameter names to possible values
            input_formatter: Function to format parameter combinations into inputs
            output_formatter: Optional function to format parameter combinations into expected outputs
            name: Optional name for the generator
            metadata: Optional metadata for the generator
        """
        super().__init__(name, metadata)
        self.parameters = parameters
        self.input_formatter = input_formatter
        self.output_formatter = output_formatter
    
    def generate(self, count: Optional[int] = None) -> List[TestCase]:
        """
        Generate test cases from parameter combinations.
        
        Args:
            count: Optional maximum number of test cases to generate
                  (if None, generates all combinations)
            
        Returns:
            List of generated TestCase objects
        """
        # Create all possible combinations of parameter values
        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())
        combinations = list(itertools.product(*param_values))
        
        # If count is specified, randomly sample combinations
        if count is not None and count < len(combinations):
            combinations = random.sample(combinations, count)
        
        test_cases = []
        
        for i, combination in enumerate(combinations):
            # Create parameter dictionary for this combination
            params = {name: value for name, value in zip(param_names, combination)}
            
            # Format input and output
            input_value = self.input_formatter(params)
            expected_output = None
            if self.output_formatter:
                expected_output = self.output_formatter(params)
            
            # Create test case
            test_case = TestCase(
                input=input_value,
                expected_output=expected_output,
                name=f"{self.name}_combo_{i+1}",
                metadata={
                    "generator_id": self.id,
                    "parameters": params
                }
            )
            
            test_cases.append(test_case)
        
        return test_cases


class PermutationGenerator(TestGenerator):
    """
    Generator that creates test cases by permuting elements of a base input.
    
    Useful for testing robustness to different orderings of the same input.
    """
    
    def __init__(
        self,
        base_input: Any,
        permutation_function: Callable[[Any], List[Any]],
        expected_output: Any = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a permutation generator.
        
        Args:
            base_input: Base input to permute
            permutation_function: Function that generates permutations of the input
            expected_output: Expected output for all permutations
            name: Optional name for the generator
            metadata: Optional metadata for the generator
        """
        super().__init__(name, metadata)
        self.base_input = base_input
        self.permutation_function = permutation_function
        self.expected_output = expected_output
    
    def generate(self, count: Optional[int] = None) -> List[TestCase]:
        """
        Generate test cases by permuting the base input.
        
        Args:
            count: Optional maximum number of test cases to generate
                  (if None, generates all permutations)
            
        Returns:
            List of generated TestCase objects
        """
        # Generate permutations
        permutations = self.permutation_function(self.base_input)
        
        # If count is specified, randomly sample permutations
        if count is not None and count < len(permutations):
            permutations = random.sample(permutations, count)
        
        test_cases = []
        
        for i, permutation in enumerate(permutations):
            # Create test case
            test_case = TestCase(
                input=permutation,
                expected_output=self.expected_output,
                name=f"{self.name}_perm_{i+1}",
                metadata={
                    "generator_id": self.id,
                    "base_input": self.base_input
                }
            )
            
            test_cases.append(test_case)
        
        return test_cases


# Common transformation functions for text augmentation

def synonym_replacement(text: str, replacement_ratio: float = 0.1) -> str:
    """Replace random words with synonyms (simplified implementation)."""
    words = text.split()
    num_to_replace = max(1, int(len(words) * replacement_ratio))
    
    # For simplicity, use a predefined list of replacements
    # In a real implementation, use a proper synonym dictionary
    replacements = {
        "good": ["great", "excellent", "fine", "positive"],
        "bad": ["poor", "terrible", "awful", "negative"],
        "large": ["big", "huge", "sizable", "massive"],
        "small": ["tiny", "little", "miniature", "compact"],
        "happy": ["glad", "joyful", "pleased", "content"],
        "sad": ["unhappy", "down", "depressed", "gloomy"]
    }
    
    indices = random.sample(range(len(words)), min(num_to_replace, len(words)))
    
    for idx in indices:
        word = words[idx].lower().strip(".,!?;:'\"")
        if word in replacements:
            words[idx] = random.choice(replacements[word])
    
    return " ".join(words)


def random_insertion(text: str, insert_ratio: float = 0.1) -> str:
    """Insert random common words into the text."""
    common_words = ["the", "a", "an", "this", "that", "and", "or", "but", "very", "really"]
    
    words = text.split()
    num_to_insert = max(1, int(len(words) * insert_ratio))
    
    for _ in range(num_to_insert):
        word = random.choice(common_words)
        position = random.randint(0, len(words))
        words.insert(position, word)
    
    return " ".join(words)


def random_swap(text: str, swap_ratio: float = 0.1) -> str:
    """Randomly swap pairs of words."""
    words = text.split()
    num_swaps = max(1, int(len(words) * swap_ratio))
    
    for _ in range(num_swaps):
        if len(words) >= 2:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
    
    return " ".join(words)


def random_deletion(text: str, delete_ratio: float = 0.1) -> str:
    """Randomly delete words from the text."""
    words = text.split()
    num_to_delete = max(1, int(len(words) * delete_ratio))
    
    # Ensure we don't delete too many words
    if len(words) - num_to_delete < 1:
        num_to_delete = len(words) - 1
    
    indices_to_delete = random.sample(range(len(words)), num_to_delete)
    new_words = [word for i, word in enumerate(words) if i not in indices_to_delete]
    
    return " ".join(new_words)


def add_typos(text: str, typo_ratio: float = 0.1) -> str:
    """Add random typos to the text."""
    words = text.split()
    num_typos = max(1, int(len(words) * typo_ratio))
    
    for _ in range(num_typos):
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        
        if len(word) <= 1:
            continue
        
        # Choose a random typo type
        typo_type = random.choice(["swap", "delete", "replace", "insert"])
        
        if typo_type == "swap" and len(word) >= 2:
            # Swap two adjacent characters
            char_idx = random.randint(0, len(word) - 2)
            words[idx] = word[:char_idx] + word[char_idx+1] + word[char_idx] + word[char_idx+2:]
        
        elif typo_type == "delete" and len(word) >= 2:
            # Delete a random character
            char_idx = random.randint(0, len(word) - 1)
            words[idx] = word[:char_idx] + word[char_idx+1:]
        
        elif typo_type == "replace":
            # Replace a random character with a nearby key
            char_idx = random.randint(0, len(word) - 1)
            nearby_keys = {
                'a': 'sqz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx',
                'e': 'wsrdf', 'f': 'dcvgtr', 'g': 'ftyhbv', 'h': 'gyujnb',
                'i': 'ujklo', 'j': 'huikm', 'k': 'jiolm', 'l': 'kop',
                'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
                'q': 'wa', 'r': 'edft', 's': 'qazxcdew', 't': 'rfgy',
                'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
                'y': 'tghu', 'z': 'asx'
            }
            char = word[char_idx].lower()
            if char in nearby_keys:
                replacement = random.choice(nearby_keys[char])
                words[idx] = word[:char_idx] + replacement + word[char_idx+1:]
        
        elif typo_type == "insert":
            # Insert a random character
            char_idx = random.randint(0, len(word))
            random_char = random.choice(string.ascii_lowercase)
            words[idx] = word[:char_idx] + random_char + word[char_idx:]
    
    return " ".join(words)


def change_case(text: str) -> str:
    """Change the case of the text (lower to upper or vice versa)."""
    if text.islower():
        return text.upper()
    if text.isupper():
        return text.lower()
    
    words = text.split()
    for i in range(len(words)):
        # Randomly change case of some words
        if random.random() < 0.5:
            words[i] = words[i].upper() if words[i].islower() else words[i].lower()
    
    return " ".join(words)


def add_noise(text: str, noise_ratio: float = 0.05) -> str:
    """Add random punctuation or special characters as noise."""
    noise_chars = ".,!?;:'\"()[]{}#$%^&*+-=~`"
    words = text.split()
    num_noise = max(1, int(len(words) * noise_ratio))
    
    for _ in range(num_noise):
        idx = random.randint(0, len(words) - 1)
        noise = random.choice(noise_chars)
        position = random.choice(["before", "after", "both"])
        
        if position == "before":
            words[idx] = noise + words[idx]
        elif position == "after":
            words[idx] = words[idx] + noise
        else:
            words[idx] = noise + words[idx] + noise
    
    return " ".join(words)


def rephrase_text(text: str) -> str:
    """
    Simple text rephrasing (placeholder implementation).
    
    Note: A real implementation would use NLP techniques or language models
    for more sophisticated rephrasing.
    """
    # Simple patterns for demonstration purposes only
    patterns = [
        # Question transformations
        (r"^What is", "Can you tell me what is"),
        (r"^How do", "Could you explain how do"),
        # Statement transformations
        (r"^I think", "In my opinion,"),
        (r"^It is", "I believe it is"),
        # Add/remove contractions
        (r"can't", "cannot"),
        (r"don't", "do not"),
        (r"cannot", "can't"),
        (r"do not", "don't"),
        # Passive/active voice (very simplified)
        (r"(\w+) was (\w+ed) by", r"\2 \1"),
    ]
    
    # Apply a random transformation
    pattern, replacement = random.choice(patterns)
    result = re.sub(pattern, replacement, text)
    
    # If no transformation was applied, add a filler word
    if result == text:
        fillers = ["actually", "basically", "certainly", "definitely", "essentially"]
        words = text.split()
        if words:
            position = random.randint(0, len(words))
            words.insert(position, random.choice(fillers))
            result = " ".join(words)
    
    return result


# Utility functions for creating permutations

def list_permutations(items: List[Any], max_count: Optional[int] = None) -> List[List[Any]]:
    """
    Generate permutations of a list of items.
    
    Args:
        items: List of items to permute
        max_count: Maximum number of permutations to generate
        
    Returns:
        List of permutations
    """
    import itertools
    
    perms = list(itertools.permutations(items))
    
    if max_count and len(perms) > max_count:
        perms = random.sample(perms, max_count)
    
    return [list(p) for p in perms]


def dict_permutations(d: Dict[str, Any], max_count: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate permutations of a dictionary by reordering keys.
    
    Args:
        d: Dictionary to permute
        max_count: Maximum number of permutations to generate
        
    Returns:
        List of permutations of the dictionary
    """
    keys = list(d.keys())
    key_perms = list_permutations(keys, max_count)
    
    perms = []
    for key_perm in key_perms:
        perm_dict = {k: d[k] for k in key_perm}
        perms.append(perm_dict)
    
    return perms


def json_permutations(json_obj: Union[Dict, List], max_count: Optional[int] = None) -> List[Union[Dict, List]]:
    """
    Generate permutations of a JSON object.
    
    Args:
        json_obj: JSON object (dict or list) to permute
        max_count: Maximum number of permutations to generate
        
    Returns:
        List of permutations of the JSON object
    """
    if isinstance(json_obj, dict):
        return dict_permutations(json_obj, max_count)
    elif isinstance(json_obj, list):
        return list_permutations(json_obj, max_count)
    else:
        return [json_obj]  # No permutations for scalar values


# Value providers for template generators

def random_int(min_val: int = 0, max_val: int = 100) -> Callable[[], int]:
    """Return a function that generates random integers in a range."""
    return lambda: random.randint(min_val, max_val)


def random_float(min_val: float = 0.0, max_val: float = 1.0) -> Callable[[], float]:
    """Return a function that generates random floats in a range."""
    return lambda: random.uniform(min_val, max_val)


def random_string(length: int = 10, chars: str = string.ascii_letters) -> Callable[[], str]:
    """Return a function that generates random strings."""
    return lambda: ''.join(random.choice(chars) for _ in range(length))


def random_date(start_date: datetime = None, end_date: datetime = None) -> Callable[[], str]:
    """Return a function that generates random dates in ISO format."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    delta = end_date - start_date
    delta_days = delta.days
    
    def date_generator():
        random_days = random.randint(0, delta_days)
        random_date = start_date + timedelta(days=random_days)
        return random_date.isoformat()[:10]  # YYYY-MM-DD format
    
    return date_generator


def random_choice(choices: List[Any]) -> Callable[[], Any]:
    """Return a function that randomly selects from choices."""
    return lambda: random.choice(choices)


def sequential_provider(values: List[Any]) -> Callable[[], Any]:
    """Return a function that sequentially returns values from a list."""
    index = 0
    
    def provider():
        nonlocal index
        value = values[index % len(values)]
        index += 1
        return value
    
    return provider 