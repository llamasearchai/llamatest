"""
LlamaTest - A testing framework for AI and LLM applications.

This module provides tools for creating, running, and analyzing tests
for AI models, with a focus on language models and other generative AI.
"""

__version__ = "0.1.0"

# Import main classes to expose at package level
from .case import TestCase
from .suite import TestSuite
from .result import TestResult
from .report import Report, ReportFormat
from .metrics import (
    accuracy,
    precision_recall_f1,
    confusion_matrix,
    text_similarity,
    llm_output_evaluation,
    regression_metrics,
    MetricRegistry
)
from .assertions import (
    assert_equal,
    assert_text_contains,
    assert_text_similarity,
    assert_model_performance,
    assert_json_structure
)
from .datasets import (
    Dataset,
    TextClassificationDataset,
    QADataset,
    load_csv_dataset,
    load_jsonl_dataset
)
from .generators import (
    TestGenerator,
    TemplateGenerator,
    DataAugmentationGenerator,
    CombinationGenerator,
    PermutationGenerator
)

# Define what's available via "from llamatest import *"
__all__ = [
    # Core testing
    'TestCase',
    'TestSuite',
    'TestResult',
    'Report',
    'ReportFormat',
    
    # Metrics
    'accuracy',
    'precision_recall_f1',
    'confusion_matrix',
    'text_similarity',
    'llm_output_evaluation',
    'regression_metrics',
    'MetricRegistry',
    
    # Assertions
    'assert_equal',
    'assert_text_contains',
    'assert_text_similarity',
    'assert_model_performance',
    'assert_json_structure',
    
    # Datasets
    'Dataset',
    'TextClassificationDataset',
    'QADataset',
    'load_csv_dataset',
    'load_jsonl_dataset',
    
    # Generators
    'TestGenerator',
    'TemplateGenerator',
    'DataAugmentationGenerator',
    'CombinationGenerator',
    'PermutationGenerator'
] 