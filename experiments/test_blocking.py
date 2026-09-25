"""
Benchmark and evaluate candidate generation (blocking) recall on validation slice.
"""

import time
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import clean_name, clean_address, extract_numbers
from src.metrics import compute_macro_f05


def run_blocking_benchmark(sample_size=10000, country='US'):
    print(f"Loading data for {country} (Sample Size: {sample_size:,} S1 entities)...")
    
    # Load raw sources
    s1_all = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t')
    s2_all = pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t')
    s3_all = pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t')
    gt_all = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t')
    
    # Filter country
    s1_df = s1_all[s1_all['country'] == country].head(sample_size).copy()
    s1_ids = set(s1_df['entity_id'])
    
    # Ground truth for this sample
    gt_df = gt_all[gt_all['source1_entity_id'].isin(s1_ids)].copy()
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna('')
    gt_map = {
        row['source1_entity_id']: [m.strip() for m in row['matched_entity_ids'].split(',') if m.strip()]
        for _, row in gt_df.iterrows()
    }
    
    # Target entities to retrieve (S2 and S3 for this country)
    s2_df = s2_all[s2_all['country'] == country].copy()
    s3_df = s3_all[s3_all['country'] == country].copy()
    
    print(f"Loaded S1 sample: {len(s1_df):,}, S2 candidate pool: {len(s2_df):,}, S3 candidate pool: {len(s3_df):,}")
    
    total_true_matches = sum(len(v) for v in gt_map.values())
    print(f"Total True Matches in Sample: {total_true_matches:,}")
    
    start_t = time.time()
    
    # 1. Clean names & addresses
    print("Normalizing strings...")
    s1_df['c_name'] = s1_df['business_name'].apply(clean_name)
    s1_df['c_addr'] = s1_df['business_address'].apply(clean_address)
    s1_df['full_text'] = s1_df['c_name'] + " " + s1_df['c_addr']
    
    s2_df['c_name'] = s2_df['business_name'].apply(clean_name)
    s2_df['c_addr'] = s2_df['business_address'].apply(clean_address)
    s2_df['full_text'] = s2_df['c_name'] + " " + s2_df['c_addr']
    
    s3_df['c_name'] = s3_df['business_name'].apply(clean_name)
    s3_df['c_addr'] = s3_df['business_address'].apply(clean_address)
    s3_df['full_text'] = s3_df['c_name'] + " " + s3_df['c_addr']
    
    # Combined target pool
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    target_ids = targets_df['entity_id'].values
    target_texts = targets_df['full_text'].values
    
    print(f"Preprocessing took {time.time() - start_t:.2f}s")
    
    # 2. Build Inverted Index on 3-char n-grams and word tokens
    print("Building TF-IDF / N-gram Index...")
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=100000)
    target_mat = vec.fit_transform(target_texts)
    s1_mat = vec.transform(s1_df['full_text'].values)
    
    print("Querying nearest candidates...")
    # Batch matrix multiplication for cosine similarity
    batch_size = 500
    top_k = 20
    candidates_dict = defaultdict(set)
    
    for i in range(0, len(s1_df), batch_size):
        batch_s1_mat = s1_mat[i:i+batch_size]
        sim_mat = batch_s1_mat.dot(target_mat.T)
        
        for local_idx in range(batch_s1_mat.shape[0]):
            global_idx = i + local_idx
            s1_id = s1_df.iloc[global_idx]['entity_id']
            
            # Extract top K indices
            row_data = sim_mat[local_idx].toarray()[0]
            if len(row_data) > 0:
                top_indices = np.argpartition(row_data, -top_k)[-top_k:]
                # Filter by min similarity threshold (e.g. > 0.15)
                for tidx in top_indices:
                    if row_data[tidx] > 0.15:
                        candidates_dict[s1_id].add(target_ids[tidx])
                        
    elapsed = time.time() - start_t
    print(f"Candidate Generation completed in {elapsed:.2f}s ({len(s1_df)/elapsed:.1f} S1/sec)")
    
    # 3. Evaluate Recall Ceiling
    captured_matches = 0
    total_candidates = sum(len(c) for c in candidates_dict.values())
    avg_cands = total_candidates / len(s1_df)
    
    for s1_id, true_list in gt_map.items():
        cand_set = candidates_dict[s1_id]
        for m in true_list:
            if m in cand_set:
                captured_matches += 1
                
    recall_ceiling = (captured_matches / total_true_matches) * 100 if total_true_matches > 0 else 0
    print("\n==========================================")
    print(f"📊 BLOCKING BENCHMARK RESULTS ({country})")
    print("==========================================")
    print(f"Sample Size: {len(s1_df):,} S1 Entities")
    print(f"Total True Matches: {total_true_matches:,}")
    print(f"Captured Matches: {captured_matches:,}")
    print(f"Recall Ceiling: {recall_ceiling:.2f}%")
    print(f"Avg Candidates / S1: {avg_cands:.2f}")
    print("==========================================\n")


if __name__ == "__main__":
    run_blocking_benchmark(sample_size=3000, country='US')
