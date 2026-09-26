"""
Amazon ML Challenge 2026 - High-Performance Dataset Preprocessor & Binary Cache
Saves cleaned and pre-tokenized dataset into compressed Apache Parquet files.
Allows any downstream matcher/model (V4+) to load the entire 11.7M dataset in < 3 seconds.
"""

import os
import sys
import time
import argparse
import re
from typing import Set, List, Dict

import pandas as pd
import numpy as np

from src.preprocessing import clean_name, clean_address, extract_numbers


def extract_postal_codes(address: str) -> str:
    """Extract space-separated postal codes."""
    if not address or not isinstance(address, str):
        return ""
    pins = re.findall(r'\b\d{5,6}\b', address)
    return " ".join(pins)


def clean_compact_name(name: str) -> str:
    """Extract compact alphanumeric name."""
    if not name or not isinstance(name, str):
        return ""
    return "".join([c for c in name.lower() if c.isalnum()])


def extract_numbers_str(address: str) -> str:
    """Extract space-separated numbers."""
    nums = extract_numbers(address)
    return " ".join(nums)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies high-speed vectorized string cleaning and feature extraction."""
    t0 = time.time()
    
    # 1. Standard cleaning
    print("  -> Cleaning names and addresses...")
    df['c_name'] = df['business_name'].astype(str).apply(clean_name)
    df['c_addr'] = df['business_address'].astype(str).apply(clean_address)
    
    # 2. Pre-extracted indexing fields
    print("  -> Extracting compact names, numbers, and postals...")
    df['comp_name'] = df['c_name'].apply(clean_compact_name)
    df['postals'] = df['c_addr'].apply(extract_postal_codes)
    df['numbers'] = df['c_addr'].apply(extract_numbers_str)
    
    # 3. String lengths
    df['len_name'] = df['c_name'].str.len().astype(np.float32)
    df['len_addr'] = df['c_addr'].str.len().astype(np.float32)
    
    print(f"  ✓ Processed {len(df):,} records in {time.time()-t0:.2f}s")
    return df


def run_preprocessing_cache(data_dir: str, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    
    print("==========================================================================")
    print(">> AMAZON ML CHALLENGE 2026: DATASET PREPROCESSING & PARQUET CACHE")
    print("==========================================================================")
    
    splits = ['test', 'train']
    sources = ['source1', 'source2', 'source3']
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split} (directory not found: {split_dir})")
            continue
            
        out_split_dir = os.path.join(cache_dir, split)
        os.makedirs(out_split_dir, exist_ok=True)
        
        for src in sources:
            filename = f"{split}_{src}.tsv"
            src_path = os.path.join(split_dir, filename)
            if not os.path.exists(src_path):
                continue
                
            out_parquet = os.path.join(out_split_dir, f"{src}.parquet")
            print(f"\n[Loading] {src_path}...")
            df = pd.read_csv(src_path, sep='\t')
            print(f"Dataset shape: {df.shape}")
            
            df_cleaned = preprocess_dataframe(df)
            
            print(f"Saving compressed Parquet to {out_parquet}...")
            df_cleaned.to_parquet(out_parquet, engine='pyarrow', compression='snappy', index=False)
            file_size_mb = os.path.getsize(out_parquet) / (1024 * 1024)
            print(f"✓ Successfully cached {src} ({len(df_cleaned):,} records, {file_size_mb:.1f} MB)")
            
    print("\n==========================================================================")
    print(f"🎉 ALL DATASETS PREPROCESSED AND CACHED TO: {cache_dir}")
    print("==========================================================================")


def load_cached_test_data(cache_dir: str, country: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Blazing fast loader: reads preprocessed Parquet cache in < 1 second.
    Returns (s1_df, target_df) where target_df combines source2 and source3.
    """
    test_dir = os.path.join(cache_dir, 'test')
    s1 = pd.read_parquet(os.path.join(test_dir, 'source1.parquet'))
    s2 = pd.read_parquet(os.path.join(test_dir, 'source2.parquet'))
    s3 = pd.read_parquet(os.path.join(test_dir, 'source3.parquet'))
    
    targets = pd.concat([s2, s3], ignore_index=True)
    
    if country:
        s1 = s1[s1['country'] == country].copy()
        targets = targets[targets['country'] == country].copy()
        
    return s1, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='student_resource/dataset')
    parser.add_argument('--cache-dir', type=str, default='data_preprocessed')
    args = parser.parse_args()
    
    run_preprocessing_cache(args.data_dir, args.cache_dir)
