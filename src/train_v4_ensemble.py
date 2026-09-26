"""
Amazon ML Challenge 2026 - Pipeline V4 40-D Ensemble Trainer (LightGBM + CatBoost)
Trains dual gradient boosted ensemble on 911,166 ground-truth pairs.
Features:
- Multi-Metric Levenshtein, Jaro-Winkler, WRatio, Token Set, Token Sort, Partial Ratio
- Brand Root Exact Equality & Length Discrepancy (Legal suffix stripped)
- Address street numbers Jaccard, count, mismatch penalty
- 5/6-digit postal code exact match, mismatch, presence
- Alphanumeric compact name equality & WRatio
- Character 3-gram Jaccard & sub-word overlap
- Locality / City token overlap
- Length differences & ratios
- Tripartite Source indicator (is_s3)
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
from rapidfuzz import fuzz, distance
from catboost import CatBoostClassifier

from src.preprocessing import clean_name, clean_address, extract_numbers
from src.metrics import compute_macro_f05


FEATURE_NAMES_V4 = [
    'name_ratio', 'name_partial_ratio', 'name_token_sort', 'name_token_set',
    'name_wratio', 'name_jaro_winkler', 'name_exact_match', 'name_len_diff', 'name_len_ratio',
    'name_first_token_match', 'name_last_token_match', 'name_char_jaccard_3gram',
    'brand_root_ratio', 'brand_root_exact', 'brand_root_token_set',
    'compact_ratio', 'compact_exact',
    'addr_ratio', 'addr_token_set', 'addr_token_sort', 'addr_partial_ratio',
    'addr_is_missing', 'addr_first_token_match', 'addr_jaro_winkler',
    'num_common_count', 'num_jaccard', 'num_mismatch_penalty',
    'pin_exact_match', 'pin_mismatch_penalty', 'pin_both_present',
    'name_addr_avg', 'brand_addr_avg', 'is_s3',
    'len_s1_name', 'len_cand_name', 'len_s1_addr', 'len_cand_addr'
]

LEGAL_SUFFIXES_REGEX = re.compile(
    r'\b(pvt ltd|private limited|ltd|llc|inc|incorporated|corp|corporation|co|company|sarl|sa|gmbh|enterprises|enterprise|services|service|store|stores|trading|agency|holdings|group)\b',
    re.IGNORECASE
)


def extract_brand_root(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    root = LEGAL_SUFFIXES_REGEX.sub('', name).strip()
    return " ".join(root.split())


def extract_postal_codes(address: str) -> Set[str]:
    if not address:
        return set()
    return set(re.findall(r'\b\d{5,6}\b', address))


def clean_compact_name(name: str) -> str:
    if not name:
        return ""
    return "".join([c for c in name.lower() if c.isalnum()])


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    if not text or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def extract_v4_features(s1_name, s1_addr, s1_root, s1_comp, s1_tokens, s1_ngrams, s1_nums, s1_postals,
                        cand_eid, cand_name, cand_addr, cand_root, cand_comp, cand_tokens, cand_ngrams, cand_nums, cand_postals):
    """Computes full 37-D feature vector."""
    len1_n, len2_n = float(len(s1_name)), float(len(cand_name))
    len1_a, len2_a = float(len(s1_addr)), float(len(cand_addr))
    
    # 1. Name Ratios
    nr = fuzz.ratio(s1_name, cand_name) / 100.0
    npr = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    nts = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    ntset = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    nwr = fuzz.WRatio(s1_name, cand_name) / 100.0
    njw = distance.JaroWinkler.similarity(s1_name, cand_name)
    nexact = 1.0 if s1_name == cand_name else 0.0
    nldiff = abs(len1_n - len2_n)
    nlratio = min(len1_n, len2_n) / max(len1_n, len2_n) if max(len1_n, len2_n) > 0 else 1.0
    
    nfirst = 1.0 if (s1_tokens and cand_tokens and s1_tokens[0] == cand_tokens[0]) else 0.0
    nlast = 1.0 if (s1_tokens and cand_tokens and s1_tokens[-1] == cand_tokens[-1]) else 0.0
    
    # 3-gram Jaccard
    if s1_ngrams and cand_ngrams:
        common_ng = len(s1_ngrams.intersection(cand_ngrams))
        ncjacc = common_ng / (len(s1_ngrams.union(cand_ngrams)) + 1e-5)
    else:
        ncjacc = 0.0
        
    # Brand Root Features
    br_r = fuzz.ratio(s1_root, cand_root) / 100.0 if (s1_root and cand_root) else nr
    br_exact = 1.0 if (s1_root and cand_root and s1_root == cand_root) else 0.0
    br_tset = fuzz.token_set_ratio(s1_root, cand_root) / 100.0 if (s1_root and cand_root) else ntset
    
    # Compact Features
    comp_r = fuzz.ratio(s1_comp, cand_comp) / 100.0 if (s1_comp and cand_comp) else 0.0
    comp_exact = 1.0 if (s1_comp and cand_comp and s1_comp == cand_comp) else 0.0
    
    # 2. Address Features
    addr_missing = 1.0 if (not s1_addr or not cand_addr) else 0.0
    if addr_missing:
        ar = atset = ats = apr = afirst = ajw = num_c = num_j = num_p = pin_match = pin_mis = pin_both = 0.0
    else:
        ar = fuzz.ratio(s1_addr, cand_addr) / 100.0
        atset = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        ats = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        apr = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
        afirst = 1.0 if (s1_addr[:6] == cand_addr[:6]) else 0.0
        ajw = distance.JaroWinkler.similarity(s1_addr, cand_addr)
        
        # Numbers
        if s1_nums and cand_nums:
            c_nums = s1_nums.intersection(cand_nums)
            num_c = float(len(c_nums))
            num_j = len(c_nums) / len(s1_nums.union(cand_nums))
            num_p = 1.0 if len(c_nums) > 0 else -1.0
        else:
            num_c = num_j = num_p = 0.0
            
        # Postals
        if s1_postals and cand_postals:
            c_pins = s1_postals.intersection(cand_postals)
            pin_match = 1.0 if len(c_pins) > 0 else 0.0
            pin_mis = -1.0 if len(c_pins) == 0 else 1.0
            pin_both = 1.0
        else:
            pin_match = pin_mis = pin_both = 0.0
            
    name_addr_avg = (nr + ar) / 2.0 if not addr_missing else nr
    brand_addr_avg = (br_r + ar) / 2.0 if not addr_missing else br_r
    is_s3 = 1.0 if cand_eid.startswith('S3-') else 0.0
    
    return np.array([
        nr, npr, nts, ntset, nwr, njw, nexact, nldiff, nlratio,
        nfirst, nlast, ncjacc,
        br_r, br_exact, br_tset,
        comp_r, comp_exact,
        ar, atset, ats, apr, addr_missing, afirst, ajw,
        num_c, num_j, num_p, pin_match, pin_mis, pin_both,
        name_addr_avg, brand_addr_avg, is_s3,
        len1_n, len2_n, len1_a, len2_a
    ], dtype=np.float32)


def train_v4_ensemble(data_dir: str, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    
    print("==========================================================================")
    print(">> AMAZON ML CHALLENGE 2026: TRAINING 40-D DUAL ENSEMBLE (LGBM + CATBOOST)")
    print("==========================================================================")
    
    # 1. Load Ground Truth first to determine required entities
    print("\n[Step 1/4] Loading Ground Truth & Sampling Pairs...")
    gt = pd.read_csv(os.path.join(data_dir, 'train/train_ground_truth.tsv'), sep='\t')
    
    pos_pairs = []
    for s1_id, matches_str in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
        if not matches_str or pd.isna(matches_str):
            continue
        matches = str(matches_str).split(',')
        for m_id in matches:
            if m_id.strip():
                pos_pairs.append((s1_id, m_id.strip()))
                
    print(f"Total Positive Ground Truth Pairs: {len(pos_pairs):,}")
    if len(pos_pairs) > 1000000:
        np.random.seed(42)
        idx_sample = np.random.choice(len(pos_pairs), size=1000000, replace=False)
        pos_pairs = [pos_pairs[i] for i in idx_sample]
        print(f"Sampled {len(pos_pairs):,} positive pairs for balanced ensemble training.")
    
    # Load Preprocessed Parquet Data
    print("Loading Preprocessed Parquet Data...")
    train_s1 = pd.read_parquet('data_preprocessed/train/source1.parquet')
    train_s2 = pd.read_parquet('data_preprocessed/train/source2.parquet')
    train_s3 = pd.read_parquet('data_preprocessed/train/source3.parquet')
    targets = pd.concat([train_s2, train_s3], ignore_index=True)
    del train_s2, train_s3
    gc.collect()
    
    # Sample 400k Negative Pairs
    np.random.seed(42)
    sample_s1 = np.random.choice(train_s1['entity_id'].values, size=min(len(train_s1), 400000), replace=False)
    sample_targets = np.random.choice(targets['entity_id'].values, size=min(len(targets), 400000), replace=False)
    neg_pairs = list(zip(sample_s1, sample_targets))
    
    needed_s1 = set(p[0] for p in pos_pairs) | set(p[0] for p in neg_pairs)
    needed_targets = set(p[1] for p in pos_pairs) | set(p[1] for p in neg_pairs)
    
    print(f"Filtering to {len(needed_s1):,} needed S1 records and {len(needed_targets):,} target records...")
    train_s1 = train_s1[train_s1['entity_id'].isin(needed_s1)]
    targets = targets[targets['entity_id'].isin(needed_targets)]
    
    s1_dict = {}
    for eid, name, addr, root, comp, nums, postals in zip(
        train_s1['entity_id'], train_s1['c_name'], train_s1['c_addr'], 
        train_s1['brand_root'], train_s1['comp_name'], train_s1['numbers'], train_s1['postals']
    ):
        name_str = str(name) if pd.notna(name) else ''
        addr_str = str(addr) if pd.notna(addr) else ''
        root_str = str(root) if pd.notna(root) else ''
        comp_str = str(comp) if pd.notna(comp) else ''
        tokens = name_str.split()
        ngrams = frozenset(get_char_ngrams(name_str, n=3))
        n_set = frozenset(str(nums).split()) if pd.notna(nums) and nums else frozenset()
        p_set = frozenset(str(postals).split()) if pd.notna(postals) and postals else frozenset()
        s1_dict[eid] = (name_str, addr_str, root_str, comp_str, tokens, ngrams, n_set, p_set)
        
    del train_s1
    gc.collect()
    
    t_dict = {}
    for eid, name, addr, root, comp, nums, postals in zip(
        targets['entity_id'], targets['c_name'], targets['c_addr'], 
        targets['brand_root'], targets['comp_name'], targets['numbers'], targets['postals']
    ):
        name_str = str(name) if pd.notna(name) else ''
        addr_str = str(addr) if pd.notna(addr) else ''
        root_str = str(root) if pd.notna(root) else ''
        comp_str = str(comp) if pd.notna(comp) else ''
        tokens = name_str.split()
        ngrams = frozenset(get_char_ngrams(name_str, n=3))
        n_set = frozenset(str(nums).split()) if pd.notna(nums) and nums else frozenset()
        p_set = frozenset(str(postals).split()) if pd.notna(postals) and postals else frozenset()
        t_dict[eid] = (name_str, addr_str, root_str, comp_str, tokens, ngrams, n_set, p_set)
        
    del targets
    gc.collect()
    
    # 2. Build Positives & Hard Negatives
    print("\n[Step 2/4] Extracting 40-D Features for 911k Positives + Negatives...")
    X_list = []
    y_list = []
    
    # Positives
    for s1_id, m_id in pos_pairs:
        if s1_id in s1_dict and m_id in t_dict:
            s1_info = s1_dict[s1_id]
            c_info = t_dict[m_id]
            feat = extract_v4_features(
                s1_info[0], s1_info[1], s1_info[2], s1_info[3], s1_info[4], s1_info[5], s1_info[6], s1_info[7],
                m_id, c_info[0], c_info[1], c_info[2], c_info[3], c_info[4], c_info[5], c_info[6], c_info[7]
            )
            X_list.append(feat)
            y_list.append(1)
            
    num_pos = len(y_list)
    print(f"Extracted {num_pos:,} positive pairs.")
    
    # Negatives
    for s1_id, rand_t_id in neg_pairs:
        if s1_id in s1_dict and rand_t_id in t_dict:
            s1_info = s1_dict[s1_id]
            c_info = t_dict[rand_t_id]
            feat = extract_v4_features(
                s1_info[0], s1_info[1], s1_info[2], s1_info[3], s1_info[4], s1_info[5], s1_info[6], s1_info[7],
                rand_t_id, c_info[0], c_info[1], c_info[2], c_info[3], c_info[4], c_info[5], c_info[6], c_info[7]
            )
            X_list.append(feat)
            y_list.append(0)
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Total Training Matrix: {X.shape}, Positives: {num_pos:,}, Negatives: {len(y)-num_pos:,}")
    
    # 3. Train LightGBM Booster
    print("\n[Step 3/4] Training Model 1: LightGBM (1,000 Trees, 4 Threads)...")
    lgb_train = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES_V4)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'num_threads': 4,
        'verbose': -1
    }
    t0 = time.time()
    lgb_model = lgb.train(params, lgb_train, num_boost_round=1000)
    lgb_path = os.path.join(model_dir, 'lgb_v4.txt')
    lgb_model.save_model(lgb_path)
    print(f"✓ LightGBM trained in {time.time()-t0:.1f}s -> Saved to {lgb_path}")
    
    # 4. Train CatBoost Booster
    print("\n[Step 4/4] Training Model 2: CatBoost Classifier (1,000 Trees, 4 Threads)...")
    t0 = time.time()
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        loss_function='Logloss',
        thread_count=4,
        verbose=100
    )
    cb_model.fit(X, y)
    cb_path = os.path.join(model_dir, 'catboost_v4.cbm')
    cb_model.save_model(cb_path)
    print(f"✓ CatBoost trained in {time.time()-t0:.1f}s -> Saved to {cb_path}")
    
    print("\n==========================================================================")
    print("🎉 DUAL ENSEMBLE TRAINING COMPLETE!")
    print(f"1. {lgb_path}")
    print(f"2. {cb_path}")
    print("==========================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='student_resource/dataset')
    parser.add_argument('--model-dir', type=str, default='models')
    args = parser.parse_args()
    
    train_v4_ensemble(args.data_dir, args.model_dir)
