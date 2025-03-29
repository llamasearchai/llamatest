"""
Datasets module for loading and managing test datasets.

This module provides utilities for working with test datasets,
including loading from various sources, dataset transformations,
and specialized datasets for AI testing.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Iterator, Tuple
import os
import json
import csv
import random
import hashlib
import datetime


class Dataset:
    """
    Base class for test datasets.
    
    A Dataset represents a collection of input-output pairs for testing,
    with support for filtering, mapping, and batching operations.
    """
    
    def __init__(self, data: List[Dict[str, Any]], name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a dataset with input-output pairs.
        
        Args:
            data: List of dictionaries containing at least 'input' and 'expected_output' keys
            name: Optional name for the dataset
            metadata: Optional metadata for the dataset
        """
        self.data = data
        self.name = name or "unnamed_dataset"
        self.metadata = metadata or {}
        
        # Ensure all items have the required keys
        for i, item in enumerate(self.data):
            if 'input' not in item:
                raise ValueError(f"Item at index {i} is missing the 'input' key")
            if 'expected_output' not in item:
                raise ValueError(f"Item at index {i} is missing the 'expected_output' key")
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get an item or slice from the dataset."""
        return self.data[idx]
    
    def filter(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> 'Dataset':
        """
        Create a new dataset with only items that pass the filter function.
        
        Args:
            filter_fn: Function that takes a dataset item and returns True to keep it
            
        Returns:
            New filtered dataset
        """
        filtered_data = [item for item in self.data if filter_fn(item)]
        return Dataset(
            filtered_data,
            name=f"{self.name}_filtered",
            metadata={**self.metadata, 'filtered_from': self.name, 'original_size': len(self.data)}
        )
    
    def map(self, map_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> 'Dataset':
        """
        Create a new dataset with items transformed by the map function.
        
        Args:
            map_fn: Function that takes a dataset item and returns a transformed item
            
        Returns:
            New mapped dataset
        """
        mapped_data = [map_fn(item) for item in self.data]
        return Dataset(
            mapped_data,
            name=f"{self.name}_mapped",
            metadata={**self.metadata, 'mapped_from': self.name}
        )
    
    def batch(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """
        Create batches of the dataset with a specified size.
        
        Args:
            batch_size: Number of items per batch
            
        Returns:
            Iterator yielding batches
        """
        for i in range(0, len(self.data), batch_size):
            yield self.data[i:i + batch_size]
    
    def sample(self, n: int, seed: Optional[int] = None) -> 'Dataset':
        """
        Create a new dataset with a random sample of items.
        
        Args:
            n: Number of items to sample
            seed: Optional random seed for reproducibility
            
        Returns:
            New sampled dataset
        """
        if n > len(self.data):
            raise ValueError(f"Cannot sample {n} items from dataset with {len(self.data)} items")
        
        if seed is not None:
            random.seed(seed)
        
        sampled_data = random.sample(self.data, n)
        return Dataset(
            sampled_data,
            name=f"{self.name}_sampled",
            metadata={
                **self.metadata,
                'sampled_from': self.name,
                'sample_size': n,
                'sample_seed': seed,
                'original_size': len(self.data)
            }
        )
    
    def split(self, ratio: float = 0.8, seed: Optional[int] = None) -> Tuple['Dataset', 'Dataset']:
        """
        Split the dataset into two parts based on the given ratio.
        
        Args:
            ratio: Proportion to include in the first dataset (0.0 to 1.0)
            seed: Optional random seed for reproducibility
            
        Returns:
            Tuple of (first_dataset, second_dataset)
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Split ratio must be between 0.0 and 1.0")
        
        if seed is not None:
            random.seed(seed)
        
        shuffled_data = random.sample(self.data, len(self.data))
        split_idx = int(len(shuffled_data) * ratio)
        
        first_data = shuffled_data[:split_idx]
        second_data = shuffled_data[split_idx:]
        
        first_dataset = Dataset(
            first_data,
            name=f"{self.name}_split1",
            metadata={
                **self.metadata,
                'split_from': self.name,
                'split_ratio': ratio,
                'split_seed': seed,
                'split_part': 1,
                'original_size': len(self.data)
            }
        )
        
        second_dataset = Dataset(
            second_data,
            name=f"{self.name}_split2",
            metadata={
                **self.metadata,
                'split_from': self.name,
                'split_ratio': ratio,
                'split_seed': seed,
                'split_part': 2,
                'original_size': len(self.data)
            }
        )
        
        return first_dataset, second_dataset
    
    def save(self, file_path: str, format: str = 'json') -> None:
        """
        Save the dataset to a file.
        
        Args:
            file_path: Path to save the dataset
            format: Format to save in ('json' or 'csv')
            
        Raises:
            ValueError: If format is not supported
        """
        if format.lower() == 'json':
            save_data = {
                'name': self.name,
                'metadata': self.metadata,
                'created_at': datetime.datetime.now().isoformat(),
                'data': self.data
            }
            
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
        
        elif format.lower() == 'csv':
            if not self.data:
                raise ValueError("Cannot save empty dataset to CSV")
            
            # Get all possible keys from all items
            all_keys = set()
            for item in self.data:
                all_keys.update(item.keys())
            
            # Ensure input and expected_output are first
            all_keys.discard('input')
            all_keys.discard('expected_output')
            fieldnames = ['input', 'expected_output'] + sorted(all_keys)
            
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.data)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, file_path: str, format: Optional[str] = None) -> 'Dataset':
        """
        Load a dataset from a file.
        
        Args:
            file_path: Path to load the dataset from
            format: Optional format override ('json' or 'csv')
            
        Returns:
            Loaded dataset
            
        Raises:
            ValueError: If format is not supported
        """
        if format is None:
            # Infer format from file extension
            _, ext = os.path.splitext(file_path)
            format = ext.lstrip('.').lower()
        
        if format.lower() == 'json':
            with open(file_path, 'r') as f:
                file_data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(file_data, dict) and 'data' in file_data:
                # Our format with metadata
                return cls(
                    file_data['data'],
                    name=file_data.get('name', 'loaded_dataset'),
                    metadata=file_data.get('metadata', {})
                )
            elif isinstance(file_data, list):
                # Simple list of items
                return cls(file_data, name=os.path.basename(file_path))
            else:
                raise ValueError(f"Unsupported JSON format in {file_path}")
        
        elif format.lower() == 'csv':
            with open(file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            
            # Convert types where possible
            for item in data:
                for key, value in item.items():
                    # Try to convert to appropriate types
                    if value.lower() == 'true':
                        item[key] = True
                    elif value.lower() == 'false':
                        item[key] = False
                    elif value.isdigit():
                        item[key] = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        item[key] = float(value)
            
            return cls(data, name=os.path.basename(file_path))
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def to_test_cases(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Convert dataset items to test case definitions.
        
        Args:
            **kwargs: Additional parameters to include in each test case
            
        Returns:
            List of test case definitions
        """
        test_cases = []
        
        for i, item in enumerate(self.data):
            test_case = {
                'input': item['input'],
                'expected_output': item['expected_output'],
                'name': item.get('name', f"test_case_{i}"),
                **kwargs
            }
            
            # Copy over any additional fields from the dataset item
            for key, value in item.items():
                if key not in test_case and key not in ('input', 'expected_output'):
                    test_case[key] = value
            
            test_cases.append(test_case)
        
        return test_cases


class TextClassificationDataset(Dataset):
    """Dataset specifically for text classification tasks."""
    
    def __init__(self, data: List[Dict[str, Any]], name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a text classification dataset.
        
        Args:
            data: List of dictionaries with 'text' and 'label' keys
            name: Optional name for the dataset
            metadata: Optional metadata for the dataset
        """
        # Convert to standard format if needed
        standardized_data = []
        for item in data:
            if 'input' not in item or 'expected_output' not in item:
                # Convert to standard format
                standardized_item = {
                    'input': item.get('text', ''),
                    'expected_output': item.get('label', ''),
                }
                
                # Copy any additional fields
                for key, value in item.items():
                    if key not in ('text', 'label'):
                        standardized_item[key] = value
                
                standardized_data.append(standardized_item)
            else:
                standardized_data.append(item)
        
        super().__init__(standardized_data, name, metadata)
    
    def get_labels(self) -> List[str]:
        """Get the unique labels in the dataset."""
        return sorted(list(set(item['expected_output'] for item in self.data)))
    
    def stratified_split(self, ratio: float = 0.8, seed: Optional[int] = None) -> Tuple['TextClassificationDataset', 'TextClassificationDataset']:
        """
        Split the dataset while preserving class distribution.
        
        Args:
            ratio: Proportion to include in the first dataset (0.0 to 1.0)
            seed: Optional random seed for reproducibility
            
        Returns:
            Tuple of (first_dataset, second_dataset)
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Split ratio must be between 0.0 and 1.0")
        
        if seed is not None:
            random.seed(seed)
        
        # Group by label
        label_groups = {}
        for item in self.data:
            label = item['expected_output']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)
        
        first_data = []
        second_data = []
        
        # Split each label group according to ratio
        for label, items in label_groups.items():
            shuffled_items = random.sample(items, len(items))
            split_idx = int(len(shuffled_items) * ratio)
            
            first_data.extend(shuffled_items[:split_idx])
            second_data.extend(shuffled_items[split_idx:])
        
        # Shuffle again to avoid having all samples of one class together
        random.shuffle(first_data)
        random.shuffle(second_data)
        
        # Create new datasets
        meta = {
            **self.metadata,
            'split_from': self.name,
            'split_ratio': ratio,
            'split_seed': seed,
            'original_size': len(self.data),
            'stratified': True
        }
        
        first_dataset = TextClassificationDataset(
            first_data,
            name=f"{self.name}_split1",
            metadata={**meta, 'split_part': 1}
        )
        
        second_dataset = TextClassificationDataset(
            second_data,
            name=f"{self.name}_split2",
            metadata={**meta, 'split_part': 2}
        )
        
        return first_dataset, second_dataset


class QADataset(Dataset):
    """Dataset specifically for question-answering tasks."""
    
    def __init__(self, data: List[Dict[str, Any]], name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a question-answering dataset.
        
        Args:
            data: List of dictionaries with 'question' and 'answer' keys
            name: Optional name for the dataset
            metadata: Optional metadata for the dataset
        """
        # Convert to standard format if needed
        standardized_data = []
        for item in data:
            if 'input' not in item or 'expected_output' not in item:
                # Convert to standard format
                standardized_item = {
                    'input': item.get('question', ''),
                    'expected_output': item.get('answer', ''),
                }
                
                # Include context if available
                if 'context' in item:
                    standardized_item['context'] = item['context']
                
                # Copy any additional fields
                for key, value in item.items():
                    if key not in ('question', 'answer'):
                        standardized_item[key] = value
                
                standardized_data.append(standardized_item)
            else:
                standardized_data.append(item)
        
        super().__init__(standardized_data, name, metadata)


def load_csv_dataset(file_path: str, input_col: str = 'input', output_col: str = 'expected_output', 
                    name: Optional[str] = None) -> Dataset:
    """
    Load a dataset from a CSV file with custom column names.
    
    Args:
        file_path: Path to the CSV file
        input_col: Name of the input column
        output_col: Name of the expected output column
        name: Optional name for the dataset
        
    Returns:
        Loaded dataset
    """
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Rename columns to standard format
    standardized_data = []
    for item in data:
        standardized_item = {
            'input': item.get(input_col, ''),
            'expected_output': item.get(output_col, '')
        }
        
        # Copy other columns
        for key, value in item.items():
            if key not in (input_col, output_col):
                standardized_item[key] = value
        
        standardized_data.append(standardized_item)
    
    return Dataset(
        standardized_data,
        name=name or os.path.basename(file_path),
        metadata={'source_file': file_path}
    )


def create_regression_dataset(size: int, func: Callable[[float], float], 
                             x_range: Tuple[float, float] = (0, 1),
                             noise: float = 0.0, seed: Optional[int] = None) -> Dataset:
    """
    Create a synthetic regression dataset.
    
    Args:
        size: Number of data points
        func: Function that maps input to output
        x_range: Range of input values (min, max)
        noise: Standard deviation of Gaussian noise to add
        seed: Random seed for reproducibility
        
    Returns:
        Dataset with synthetic regression data
    """
    if seed is not None:
        random.seed(seed)
    
    x_min, x_max = x_range
    data = []
    
    for _ in range(size):
        x = random.uniform(x_min, x_max)
        y = func(x)
        
        if noise > 0:
            y += random.gauss(0, noise)
        
        data.append({
            'input': x,
            'expected_output': y
        })
    
    return Dataset(
        data,
        name=f"regression_dataset_{size}",
        metadata={
            'synthetic': True,
            'size': size,
            'x_range': x_range,
            'noise': noise,
            'seed': seed
        }
    )


def create_classification_dataset(size: int, num_classes: int, num_features: int = 2,
                                 class_sep: float = 1.0, noise: float = 0.0,
                                 seed: Optional[int] = None) -> Dataset:
    """
    Create a synthetic classification dataset.
    
    Args:
        size: Number of data points
        num_classes: Number of classes
        num_features: Number of input features
        class_sep: Separation between classes
        noise: Standard deviation of Gaussian noise to add
        seed: Random seed for reproducibility
        
    Returns:
        Dataset with synthetic classification data
    """
    if seed is not None:
        random.seed(seed)
    
    data = []
    
    # Generate class centers
    centers = []
    for _ in range(num_classes):
        center = [random.uniform(-10, 10) for _ in range(num_features)]
        centers.append(center)
    
    # Generate data points
    for _ in range(size):
        # Randomly select a class
        class_idx = random.randint(0, num_classes - 1)
        center = centers[class_idx]
        
        # Generate features with separation and noise
        features = []
        for i in range(num_features):
            feature = center[i] + random.gauss(0, noise) + class_sep
            features.append(feature)
        
        data.append({
            'input': features,
            'expected_output': class_idx
        })
    
    return Dataset(
        data,
        name=f"classification_dataset_{size}_{num_classes}",
        metadata={
            'synthetic': True,
            'size': size,
            'num_classes': num_classes,
            'num_features': num_features,
            'class_sep': class_sep,
            'noise': noise,
            'seed': seed
        }
    )


def load_jsonl_dataset(file_path: str, input_field: str = 'input', 
                      output_field: str = 'expected_output', name: Optional[str] = None) -> Dataset:
    """
    Load a dataset from a JSONL (JSON Lines) file.
    
    Args:
        file_path: Path to the JSONL file
        input_field: Name of the input field
        output_field: Name of the expected output field
        name: Optional name for the dataset
        
    Returns:
        Loaded dataset
    """
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                
                standardized_item = {
                    'input': item.get(input_field, ''),
                    'expected_output': item.get(output_field, '')
                }
                
                # Copy other fields
                for key, value in item.items():
                    if key not in (input_field, output_field):
                        standardized_item[key] = value
                
                data.append(standardized_item)
    
    return Dataset(
        data,
        name=name or os.path.basename(file_path),
        metadata={'source_file': file_path}
    )


def dataset_hash(dataset: Dataset) -> str:
    """
    Compute a hash of the dataset contents.
    
    Args:
        dataset: Dataset to hash
        
    Returns:
        SHA-256 hash of the dataset
    """
    # Convert dataset to a consistent JSON string
    data_str = json.dumps(dataset.data, sort_keys=True)
    
    # Compute SHA-256 hash
    return hashlib.sha256(data_str.encode()).hexdigest()


def combine_datasets(datasets: List[Dataset], name: Optional[str] = None) -> Dataset:
    """
    Combine multiple datasets into one.
    
    Args:
        datasets: List of datasets to combine
        name: Optional name for the combined dataset
        
    Returns:
        Combined dataset
    """
    if not datasets:
        raise ValueError("Cannot combine empty list of datasets")
    
    combined_data = []
    component_names = []
    
    for dataset in datasets:
        combined_data.extend(dataset.data)
        component_names.append(dataset.name)
    
    metadata = {
        'combined_from': component_names,
        'component_sizes': [len(dataset) for dataset in datasets]
    }
    
    return Dataset(
        combined_data,
        name=name or f"combined_dataset_{len(combined_data)}",
        metadata=metadata
    ) 