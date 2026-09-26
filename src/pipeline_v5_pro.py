"""
Amazon ML Challenge 2026 - Pipeline V5 Pro (High Precision Blocker Ensemble)
"""

import time
import os
import sys
import argparse
import re
import gc
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from rapidfuzz import fuzz, distance
import networkx as nx

try:
    from train_hard_negative_ensemble import extract_v5_features, extract_brand_root, clean_compact_name, get_char_ngrams, MultiChannelBlockerV5, FEATURE_NAMES_V5
except ImportError:
    from src.train_hard_negative_ensemble import extract_v5_features, extract_brand_root, clean_compact_name, get_char_ngrams, MultiChannelBlockerV5, FEATURE_NAMES_V5


def run_pipeline_v5_pro(cache_dir: str = 'data_preprocessed', model_dir: str = 'models', output_dir: str = 'output_v5', threshold: float = 0.65):
    os.makedirs(output_dir, exist_ok=True)
    
    print("==========================================================================")
    print(">> AMAZON ML CHALLENGE 2026: PIPELINE V5 PRO (HARD-NEGATIVE ENSEMBLE)")
    print("==========================================================================")
    
    # 1. Load Dual Models
    lgb_path = os.path.join(model_dir, 'lgb_v5_hard.txt')
    cb_path = os.path.join(model_dir, 'catboost_v5_hard.cbm')
    
    print(f">> Loading LightGBM Model: {lgb_path}")
    lgb_model = lgb.Booster(model_file=lgb_path)
    
    print(f">> Loading CatBoost Model: {cb_path}")
    cb_model = CatBoostClassifier()
    cb_model.load_model(cb_path)
    
    final_matching_path = os.path.join(output_dir, 'matching_results.tsv')
    final_cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    
    with open(final_matching_path, 'w', encoding='utf-8') as f_m, \
         open(final_cand_path, 'w', encoding='utf-8') as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        
    countries = ['US', 'France', 'India']
    total_start = time.time()
    batch_size = 10000
    test_dir = os.path.join(cache_dir, 'test')
    
    total_singletons = 0
    total_matches_written = 0
    
    for country in countries:
        c_start = time.time()
        print(f"\n==========================================================================")
        print(f">> [Step 1/3] Loading Partitioned Parquet & Building Inverted Index for {country}...")
        print(f"==========================================================================")
        
        t0 = time.time()
        c_s1 = pd.read_parquet(os.path.join(test_dir, 'source1.parquet'), filters=[('country', '==', country)])
        c_s2 = pd.read_parquet(os.path.join(test_dir, 'source2.parquet'), filters=[('country', '==', country)])
        c_s3 = pd.read_parquet(os.path.join(test_dir, 'source3.parquet'), filters=[('country', '==', country)])
        c_targets = pd.concat([c_s2, c_s3], ignore_index=True)
        del c_s2, c_s3
        gc.collect()
        print(f"✓ Loaded {len(c_s1):,} S1 queries and {len(c_targets):,} Target records for {country} in {time.time()-t0:.2f}s")
        
        blocker = MultiChannelBlockerV5()
        blocker.fit(
            entity_ids=c_targets['entity_id'].tolist(),
            names=c_targets['c_name'].tolist(),
            addresses=c_targets['c_addr'].tolist(),
            roots=c_targets['brand_root'].tolist(),
            comps=c_targets['comp_name'].tolist(),
            numbers=c_targets['numbers'].tolist(),
            postals=c_targets['postals'].tolist()
        )
        del c_targets
        gc.collect()
        
        print(f"Pre-extracting metadata for {len(c_s1):,} S1 entities in {country}...")
        s1_tuples = []
        for eid, name, addr, root, comp, nums_str, postals_str in zip(
            c_s1['entity_id'], c_s1['c_name'], c_s1['c_addr'], c_s1['brand_root'], c_s1['comp_name'], c_s1['numbers'], c_s1['postals']
        ):
            name_str = str(name) if pd.notna(name) else ""
            addr_str = str(addr) if pd.notna(addr) else ""
            root_str = str(root) if pd.notna(root) else ""
            comp_str = str(comp) if pd.notna(comp) else ""
            tokens = name_str.split()
            ngrams = frozenset(get_char_ngrams(name_str, n=3))
            nums = frozenset(str(nums_str).split()) if pd.notna(nums_str) and nums_str else frozenset()
            pins = frozenset(str(postals_str).split()) if pd.notna(postals_str) and postals_str else frozenset()
            s1_tuples.append((str(eid), name_str, addr_str, root_str, comp_str, tokens, ngrams, nums, pins))
            
        del c_s1
        gc.collect()
        
        total_s1 = len(s1_tuples)
        print(f"⚡ Streaming Dual Ensemble Inference for {country} ({total_s1:,} queries)...")
        
        with open(final_matching_path, 'a', encoding='utf-8') as f_m, \
             open(final_cand_path, 'a', encoding='utf-8') as f_c:
             
            for start_idx in range(0, total_s1, batch_size):
                b_t0 = time.time()
                batch = s1_tuples[start_idx:start_idx+batch_size]
                batch_feats = []
                batch_meta = []
                batch_cand_ids = defaultdict(list)
                
                for b_idx, s1_info in enumerate(batch):
                    s1_id, s1_name, s1_addr, s1_root, s1_comp, s1_tokens, s1_ngrams, s1_nums, s1_postals = s1_info
                    cand_indices = blocker.retrieve_candidates(s1_name, s1_addr, s1_root, s1_comp, s1_tokens, top_k=25)
                    
                    for c_idx in cand_indices:
                        cand_info = blocker.target_data[c_idx]
                        cand_eid = cand_info[0]
                        batch_cand_ids[s1_id].append(cand_eid)
                        
                        feat_vec = extract_v5_features(
                            s1_name, s1_addr, s1_root, s1_comp, s1_tokens, s1_ngrams, s1_nums, s1_postals,
                            cand_eid, cand_info[1], cand_info[2], cand_info[3], cand_info[4], cand_info[5], cand_info[6], cand_info[7], cand_info[8]
                        )
                        batch_feats.append(feat_vec)
                        batch_meta.append((b_idx, s1_id, cand_eid, c_idx))
                        
                # Scored matches
                s1_scored_matches = defaultdict(list)
                if batch_feats:
                    X_batch = np.array(batch_feats, dtype=np.float32)
                    p_lgb = lgb_model.predict(X_batch)
                    p_cb = cb_model.predict_proba(X_batch)[:, 1]
                    p_blend = (0.5 * p_lgb) + (0.5 * p_cb)
                    
                    for (b_idx, s1_id, cand_eid, c_idx), prob in zip(batch_meta, p_blend):
                        if prob >= threshold:
                            s1_info = batch[b_idx]
                            cand_info = blocker.target_data[c_idx]
                            
                            s1_postals = s1_info[8]
                            cand_postals = cand_info[8]
                            s1_nums = s1_info[7]
                            cand_nums = cand_info[7]
                            
                            # Negative Constraints
                            if s1_postals and cand_postals and len(s1_postals.intersection(cand_postals)) == 0 and fuzz.ratio(s1_info[1], cand_info[1]) < 90:
                                continue
                            if s1_nums and cand_nums and len(s1_nums.intersection(cand_nums)) == 0 and fuzz.ratio(s1_info[1], cand_info[1]) < 85:
                                continue
                                
                            s1_scored_matches[s1_id].append((cand_eid, prob))
                            
                # Write batch stream
                for b_idx, s1_info in enumerate(batch):
                    s1_id = s1_info[0]
                    cand_list = s1_scored_matches.get(s1_id, [])
                    cand_list.sort(key=lambda x: x[1], reverse=True)
                    
                    s2_matches = []
                    s3_matches = []
                    for ceid, _ in cand_list:
                        if ceid.startswith('S2-') and len(s2_matches) < 4:
                            s2_matches.append(ceid)
                        elif ceid.startswith('S3-') and len(s3_matches) < 4:
                            s3_matches.append(ceid)
                            
                    final_matches = s2_matches + s3_matches
                    cands_str = ",".join(batch_cand_ids.get(s1_id, []))
                    matches_str = ",".join(final_matches)
                    
                    if not matches_str:
                        total_singletons += 1
                    else:
                        total_matches_written += 1
                        
                    f_c.write(f"{s1_id}\t{cands_str}\n")
                    f_m.write(f"{s1_id}\t{matches_str}\n")
                    
                processed = min(start_idx + batch_size, total_s1)
                elapsed = time.time() - c_start
                rate = processed / max(elapsed, 0.001)
                b_time = time.time() - b_t0
                if (start_idx // batch_size) % 2 == 0 or processed == total_s1:
                    print(f"  [{country}] Processed {processed:,}/{total_s1:,} ({processed/total_s1*100:.1f}%) | Singletons: {total_singletons:,} | Speed: {rate:.1f} ent/s (batch {b_time:.2f}s)", flush=True)
                    
        del blocker, s1_tuples
        gc.collect()
        
        c_elapsed = time.time() - c_start
        c_rate = total_s1 / c_elapsed
        print(f"✓ Finished {country} ({total_s1:,} records) in {c_elapsed:.1f}s ({c_rate:.1f} ent/s)!")
        
    total_elapsed = time.time() - total_start
    print("\n==========================================================================")
    print(f"🎉 PIPELINE V5 PRO INFERENCE COMPLETE in {total_elapsed/60:.2f} minutes!")
    print(f"Total Singletons: {total_singletons:,} ({(total_singletons/1732544)*100:.2f}%)")
    print(f"Total Matches: {total_matches_written:,}")
    print(f"1. {final_matching_path}")
    print(f"2. {final_cand_path}")
    print("==========================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, default='data_preprocessed')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--output-dir', type=str, default='output_v5')
    parser.add_argument('--threshold', type=float, default=0.65)
    args = parser.parse_args()
    
    run_pipeline_v5_pro(args.cache_dir, args.model_dir, args.output_dir, args.threshold)
