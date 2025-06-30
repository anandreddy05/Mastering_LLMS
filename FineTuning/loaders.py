from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from items import Item
import os
import shutil

CHUNK_SIZE = 1000
MIN_PRICE = 0.5
MAX_PRICE = 999.49 # Taken to reduce the skewness as the data.

class ItemLoader:
    def __init__(self, name):
        self.name = name
        self.dataset = None

    def clear_dataset_cache(self):
        """Clear the Hugging Face dataset cache for this specific dataset"""
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
            dataset_cache_path = os.path.join(cache_dir, "McAuley-Lab___amazon-reviews-2023")
            if os.path.exists(dataset_cache_path):
                print(f"Clearing cache at {dataset_cache_path}")
                shutil.rmtree(dataset_cache_path)
                return True
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")
        return False

    def check_available_configs(self):
        """Check what configurations are actually available for this dataset"""
        try:
            configs = get_dataset_config_names("McAuley-Lab/Amazon-Reviews-2023")
            print(f"Available configurations: {len(configs)} total")
            
            # Look for configs that match our dataset name
            matching_configs = []
            for config in configs:
                if self.name.lower() in config.lower():
                    matching_configs.append(config)
            
            if matching_configs:
                print(f"Matching configurations for '{self.name}': {matching_configs}")
                return matching_configs
            else:
                print(f"No exact matches found for '{self.name}'")
                print("Available configurations (first 10):", configs[:10])
                return []
        except Exception as e:
            print(f"Error checking configurations: {e}")
            return []

    def try_load_dataset(self, config_name, force_download=False):
        """Attempt to load dataset with given configuration"""
        try:
            print(f"Attempting to load with config: {config_name}")
            
            load_params = {
                "path": "McAuley-Lab/Amazon-Reviews-2023",
                "name": config_name,
                "split": "full",
                "trust_remote_code": True
            }
            
            if force_download:
                load_params["download_mode"] = "force_redownload"
                load_params["cache_dir"] = None
            
            dataset = load_dataset(**load_params)
            print(f"✓ Successfully loaded dataset with config: {config_name}")
            print(f"Dataset size: {len(dataset):,} samples")
            return dataset
            
        except Exception as e:
            print(f"✗ Failed to load with config '{config_name}': {str(e)}")
            return None

    def from_datapoint(self, datapoint):
        """
        Try to create an Item from this datapoint
        Return the Item if successful, or None if it shouldn't be included
        """
        try:
            price_str = datapoint.get('price', '')
            if price_str:
                price = float(price_str)
                if MIN_PRICE <= price <= MAX_PRICE:
                    item = Item(datapoint, price)
                    return item if item.include else None
        except (ValueError, TypeError, AttributeError):
            return None

    def from_chunk(self, chunk):
        """
        Create a list of Items from this chunk of elements from the Dataset
        """
        batch = []
        for datapoint in chunk:
            result = self.from_datapoint(datapoint)
            if result:
                batch.append(result)
        return batch

    def chunk_generator(self):
        """
        Iterate over the Dataset, yielding chunks of datapoints at a time
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers):
        """
        Use concurrent.futures to farm out the work to process chunks of datapoints -
        This speeds up processing significantly, but will tie up your computer while it's doing so!
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)
        for result in results:
            result.category = self.name
        return results
            
    def load(self, workers=8):
        """
        Load in this dataset; the workers parameter specifies how many processes
        should work on loading and scrubbing the data
        """
        start = datetime.now()
        print(f"Loading dataset {self.name}", flush=True)
        
        # The configurations are confirmed to exist, so let's try different loading strategies
        config_name = f"raw_meta_{self.name}"
        
        # Strategy 1: Try with force redownload first
        try:
            print(f"Attempting to load {config_name} with force redownload...")
            self.dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023", 
                config_name,
                split="full", 
                trust_remote_code=True,
                download_mode="force_redownload"
            )
            print(f"✓ Successfully loaded {config_name}")
            
        except Exception as e1:
            print(f"Force redownload failed: {e1}")
            
            # Strategy 2: Clear cache and try again
            try:
                print("Clearing cache and retrying...")
                self.clear_dataset_cache()
                self.dataset = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023", 
                    config_name,
                    split="full", 
                    trust_remote_code=True,
                    cache_dir=None  # Don't use cache
                )
                print(f"✓ Successfully loaded {config_name} after cache clear")
                
            except Exception as e2:
                print(f"Cache clear retry failed: {e2}")
                
                # Strategy 3: Try with streaming and then convert
                try:
                    print("Trying streaming mode...")
                    streaming_dataset = load_dataset(
                        "McAuley-Lab/Amazon-Reviews-2023", 
                        config_name,
                        split="full", 
                        trust_remote_code=True,
                        streaming=True
                    )
                    # Convert streaming dataset to regular dataset
                    print("Converting streaming dataset to regular dataset...")
                    self.dataset = streaming_dataset.to_iterable_dataset()
                    print(f"✓ Successfully loaded {config_name} via streaming")
                    
                except Exception as e3:
                    print(f"All loading strategies failed for {self.name}")
                    print(f"Final error: {e3}")
                    raise FileNotFoundError(f"Could not load dataset {self.name} with any method")
        
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(f"Completed {self.name} with {len(results):,} datapoints in {(finish-start).total_seconds()/60:.1f} mins", flush=True)
        return results

