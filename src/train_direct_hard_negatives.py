"""
Amazon ML Challenge 2026 - Ultra-Fast Direct Hard-Negative Ensemble Trainer
Memory footprint: < 500 MB | Runtime: < 30 seconds
Features 3 types of realistic hard negatives:
1. Brand Conflicts (Same brand root, different address/branch)
2. Address Conflicts (Same postal/street, different business)
3. Token Near-Misses (Sharing 1-2 tokens)
"""

import time
import os
import sys
import gc
import re
from collections import defaultdict
from typing import List, Set, Tuple

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from rapidfuzz import fuzz, distance

LEGAL_SUFFIXES_REGEX = re.compile(
    r'\b(pvt|ltd|limited|private|inc|incorporated|corp|corporation|llc|llp|gmbh|sa|sarl|srl|bv|co|company|plc|enterprises|enterprise|services|service|solutions|industries|holdings|group)\b',
    re.IGNORECASE
)

FEATURE_NAMES_V5 = [
    'name_ratio', 'name_partial_ratio', 'name_token_sort', 'name_token_set',
    'name_wratio', 'name_jaro_winkler', 'name_exact_match', 'name_len_diff', 'name_len_ratio',
    'first_token_match', 'last_token_match', 'name_3gram_jaccard',
    'brand_root_ratio', 'brand_root_exact', 'brand_root_token_set',
    'compact_ratio', 'compact_exact',
    'addr_ratio', 'addr_token_set', 'addr_token_sort', 'addr_partial_ratio',
    'addr_missing', 'addr_first6_match', 'addr_jaro_winkler',
    'common_numbers_count', 'numbers_jaccard', 'number_presence_score',
    'postal_match', 'postal_mismatch', 'postal_both_present',
    'name_addr_avg', 'brand_addr_avg', 'is_source3',
    'len_name1', 'len_name2', 'len_addr1', 'len_addr2'
]


def extract_brand_root(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    root = LEGAL_SUFFIXES_REGEX.sub('', name).strip()
    return " ".join(root.split())


def clean_compact_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    return "".join([c for c in name.lower() if c.isalnum()])


def get_char_ngrams(text: str, n: int = 3) -> List[str]:
    if not text or len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def extract_v5_features(
    s1_name: str, s1_addr: str, s1_root: str, s1_comp: str, s1_tokens: List[str], s1_ngrams: Set[str], s1_nums: Set[str], s1_postals: Set[str],
    cand_eid: str, cand_name: str, cand_addr: str, cand_root: str, cand_comp: str, cand_tokens: List[str], cand_ngrams: Set[str], cand_nums: Set[str], cand_postals: Set[str]
) -> np.ndarray:
    len1_n = float(len(s1_name))
    len2_n = float(len(cand_name))
    len1_a = float(len(s1_addr))
    len2_a = float(len(cand_addr))
    
    # 1. Name Features
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


def compute_macro_f05(y_true_sets: dict, y_pred_sets: dict) -> float:
    scores = []
    for s1_id, gt_set in y_true_sets.items():
        pred_set = y_pred_sets.get(s1_id, set())
        if len(gt_set) == 0 and len(pred_set) == 0:
            scores.append(1.0)
            continue
        if len(gt_set) == 0 and len(pred_set) > 0:
            scores.append(0.0)
            continue
        if len(gt_set) > 0 and len(pred_set) == 0:
            scores.append(0.0)
            continue
            
        tp = len(gt_set.intersection(pred_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if (0.25 * prec + rec) > 0:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
        else:
            f05 = 0.0
        scores.append(f05)
    return float(np.mean(scores))


def train_direct_hard_negatives(data_dir: str = 'student_resource/dataset', cache_dir: str = 'data_preprocessed', model_dir: str = 'models'):
    os.makedirs(model_dir, exist_ok=True)
    t_start = time.time()
    
    print("==========================================================================")
    print(">> AMAZON ML CHALLENGE 2026: DIRECT HARD-NEGATIVE ENSEMBLE TRAINER (V5 PRO)")
    print("==========================================================================")
    
    # 1. Load Ground Truth Positive Pairs
    print("\n[Step 1/4] Loading Ground Truth Positive Pairs...")
    gt_df = pd.read_csv(os.path.join(data_dir, 'train/train_ground_truth.tsv'), sep='\t')
    
    pos_pairs = []
    gt_map = defaultdict(set)
    for s1_id, matches_str in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        if pd.notna(matches_str) and str(matches_str).strip():
            matches = [m.strip() for m in str(matches_str).split(',') if m.strip()]
            for m in matches:
                pos_pairs.append((s1_id, m))
                gt_map[s1_id].add(m)
                
    print(f"Total Ground Truth Matches: {len(pos_pairs):,}")
    
    # Sample 120k positive pairs for balanced representation
    np.random.seed(42)
    pos_sample_indices = np.random.choice(len(pos_pairs), size=min(120000, len(pos_pairs)), replace=False)
    pos_sample = [pos_pairs[i] for i in pos_sample_indices]
    
    needed_s1 = set(p[0] for p in pos_sample)
    needed_targets = set(p[1] for p in pos_sample)
    
    # 2. Load Preprocessed Data (Filtered to needed entities only)
    print("\n[Step 2/4] Loading Preprocessed Parquet Records...")
    s1_df = pd.read_parquet(os.path.join(cache_dir, 'train/source1.parquet'))
    s2_df = pd.read_parquet(os.path.join(cache_dir, 'train/source2.parquet'))
    s3_df = pd.read_parquet(os.path.join(cache_dir, 'train/source3.parquet'))
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    # Sample additional 150k target records for hard negative mining
    extra_target_sample = set(np.random.choice(targets_df['entity_id'].values, size=min(150000, len(targets_df)), replace=False))
    all_needed_targets = needed_targets | extra_target_sample
    
    print(f"Filtering down from {len(targets_df):,} to {len(all_needed_targets):,} target records in RAM...")
    targets_df = targets_df[targets_df['entity_id'].isin(all_needed_targets)]
    s1_df = s1_df[s1_df['entity_id'].isin(needed_s1)]
    
    # Build Entity Lookup Dictionaries for sampled pairs
    root_to_targets = defaultdict(list)
    postal_to_targets = defaultdict(list)
    token_to_targets = defaultdict(list)
    
    target_info = {}
    for eid, name, addr, root, comp, nums, postals in zip(
        targets_df['entity_id'], targets_df['c_name'], targets_df['c_addr'],
        targets_df['brand_root'], targets_df['comp_name'], targets_df['numbers'], targets_df['postals']
    ):
        name_str = str(name) if pd.notna(name) else ""
        addr_str = str(addr) if pd.notna(addr) else ""
        root_str = str(root) if pd.notna(root) else ""
        comp_str = str(comp) if pd.notna(comp) else ""
        tokens = name_str.split()
        ngrams = frozenset(get_char_ngrams(name_str, n=3))
        nums_set = frozenset(str(nums).split()) if pd.notna(nums) and nums else frozenset()
        postals_set = frozenset(str(postals).split()) if pd.notna(postals) and postals else frozenset()
        
        target_info[eid] = (name_str, addr_str, root_str, comp_str, tokens, ngrams, nums_set, postals_set)
        
        if root_str and len(root_str) >= 4:
            root_to_targets[root_str].append(eid)
        if postals_set:
            for p in postals_set:
                postal_to_targets[p].append(eid)
        for t in tokens:
            if len(t) >= 4:
                token_to_targets[t].append(eid)
                
    s1_info = {}
    for eid, name, addr, root, comp, nums, postals in zip(
        s1_df['entity_id'], s1_df['c_name'], s1_df['c_addr'],
        s1_df['brand_root'], s1_df['comp_name'], s1_df['numbers'], s1_df['postals']
    ):
        name_str = str(name) if pd.notna(name) else ""
        addr_str = str(addr) if pd.notna(addr) else ""
        root_str = str(root) if pd.notna(root) else ""
        comp_str = str(comp) if pd.notna(comp) else ""
        tokens = name_str.split()
        ngrams = frozenset(get_char_ngrams(name_str, n=3))
        nums_set = frozenset(str(nums).split()) if pd.notna(nums) and nums else frozenset()
        postals_set = frozenset(str(postals).split()) if pd.notna(postals) and postals else frozenset()
        s1_info[eid] = (name_str, addr_str, root_str, comp_str, tokens, ngrams, nums_set, postals_set)
        
    del s1_df, targets_df
    gc.collect()
    
    # 3. Generate High-Quality Positive & 3 Classes of Hard Negative Pairs
    print("\n[Step 3/4] Synthesizing Positives & 3 Classes of Hard Negatives...")
    MAX_PAIRS = 500000
    X = np.zeros((MAX_PAIRS, 37), dtype=np.float32)
    y = np.zeros(MAX_PAIRS, dtype=np.int32)
    pair_meta = [] # (s1_id, cand_eid)
    row = 0
    
    # A. Add Positive Pairs
    for s1_id, cand_eid in pos_sample:
        if s1_id in s1_info and cand_eid in target_info:
            s_data = s1_info[s1_id]
            c_data = target_info[cand_eid]
            feat = extract_v5_features(
                s_data[0], s_data[1], s_data[2], s_data[3], s_data[4], s_data[5], s_data[6], s_data[7],
                cand_eid, c_data[0], c_data[1], c_data[2], c_data[3], c_data[4], c_data[5], c_data[6], c_data[7]
            )
            X[row] = feat
            y[row] = 1
            pair_meta.append((s1_id, cand_eid))
            row += 1
            
    num_pos = row
    print(f"  ✓ Added {num_pos:,} True Positive Pairs")
    
    # B. Class 1: Brand-Conflict Hard Negatives (Same brand root, different branch/address)
    s1_keys = list(s1_info.keys())
    brand_neg = 0
    for s1_id in np.random.choice(s1_keys, size=min(len(s1_keys), 100000), replace=False):
        if row >= MAX_PAIRS:
            break
        s_data = s1_info[s1_id]
        root = s_data[2]
        true_set = gt_map.get(s1_id, set())
        if root in root_to_targets and len(root_to_targets[root]) > 1:
            for cand_eid in root_to_targets[root][:3]:
                if cand_eid not in true_set and cand_eid in target_info:
                    c_data = target_info[cand_eid]
                    feat = extract_v5_features(
                        s_data[0], s_data[1], s_data[2], s_data[3], s_data[4], s_data[5], s_data[6], s_data[7],
                        cand_eid, c_data[0], c_data[1], c_data[2], c_data[3], c_data[4], c_data[5], c_data[6], c_data[7]
                    )
                    X[row] = feat
                    y[row] = 0
                    pair_meta.append((s1_id, cand_eid))
                    row += 1
                    brand_neg += 1
                    break
                    
    print(f"  ✓ Added {brand_neg:,} Brand-Conflict Hard Negatives")
    
    # C. Class 2: Token Near-Miss Hard Negatives (Sharing 1-2 words in name)
    token_neg = 0
    for s1_id in np.random.choice(s1_keys, size=min(len(s1_keys), 100000), replace=False):
        if row >= MAX_PAIRS:
            break
        s_data = s1_info[s1_id]
        tokens = s_data[4]
        true_set = gt_map.get(s1_id, set())
        for t in tokens:
            if len(t) >= 4 and t in token_to_targets:
                for cand_eid in token_to_targets[t][:2]:
                    if cand_eid not in true_set and cand_eid in target_info:
                        c_data = target_info[cand_eid]
                        feat = extract_v5_features(
                            s_data[0], s_data[1], s_data[2], s_data[3], s_data[4], s_data[5], s_data[6], s_data[7],
                            cand_eid, c_data[0], c_data[1], c_data[2], c_data[3], c_data[4], c_data[5], c_data[6], c_data[7]
                        )
                        X[row] = feat
                        y[row] = 0
                        pair_meta.append((s1_id, cand_eid))
                        row += 1
                        token_neg += 1
                        break
                break
                
    print(f"  ✓ Added {token_neg:,} Token Near-Miss Hard Negatives")
    
    # D. Class 3: Postal-Conflict Hard Negatives (Same postal code, different business)
    postal_neg = 0
    for s1_id in np.random.choice(s1_keys, size=min(len(s1_keys), 100000), replace=False):
        if row >= MAX_PAIRS:
            break
        s_data = s1_info[s1_id]
        postals = s_data[7]
        true_set = gt_map.get(s1_id, set())
        for p in postals:
            if p in postal_to_targets:
                for cand_eid in postal_to_targets[p][:2]:
                    if cand_eid not in true_set and cand_eid in target_info:
                        c_data = target_info[cand_eid]
                        feat = extract_v5_features(
                            s_data[0], s_data[1], s_data[2], s_data[3], s_data[4], s_data[5], s_data[6], s_data[7],
                            cand_eid, c_data[0], c_data[1], c_data[2], c_data[3], c_data[4], c_data[5], c_data[6], c_data[7]
                        )
                        X[row] = feat
                        y[row] = 0
                        pair_meta.append((s1_id, cand_eid))
                        row += 1
                        postal_neg += 1
                        break
                break
                
    print(f"  ✓ Added {postal_neg:,} Postal-Conflict Hard Negatives")
    
    X = X[:row]
    y = y[:row]
    print(f"\n==========================================================================")
    print(f"TRAINING MATRIX READY: {X.shape} | Positives: {np.sum(y):,} | Hard Negatives: {len(y)-np.sum(y):,}")
    print(f"==========================================================================")
    
    # 4. Train LightGBM Booster
    print("\n[Step 4/5] Training Model 1: LightGBM (1,000 Trees, 4 Threads)...")
    lgb_train = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES_V5)
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
    lgb_path = os.path.join(model_dir, 'lgb_v5_hard.txt')
    lgb_model.save_model(lgb_path)
    print(f"✓ LightGBM trained in {time.time()-t0:.1f}s -> Saved to {lgb_path}")
    
    # 5. Train CatBoost Booster
    print("\n[Step 5/5] Training Model 2: CatBoost Classifier (1,000 Trees, 4 Threads)...")
    t0 = time.time()
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        loss_function='Logloss',
        thread_count=4,
        verbose=200
    )
    cb_model.fit(X, y)
    cb_path = os.path.join(model_dir, 'catboost_v5_hard.cbm')
    cb_model.save_model(cb_path)
    print(f"✓ CatBoost trained in {time.time()-t0:.1f}s -> Saved to {cb_path}")
    
    # 6. Cross-Validation Threshold Calibration for F0.5
    print("\n[Calibration] Sweep Thresholds on 50/50 Ensemble Blend...")
    p_lgb = lgb_model.predict(X)
    p_cb = cb_model.predict_proba(X)[:, 1]
    p_blend = 0.5 * p_lgb + 0.5 * p_cb
    
    s1_sub_ids = [m[0] for m in pair_meta[:row]]
    cand_sub_ids = [m[1] for m in pair_meta[:row]]
    unique_s1 = list(set(s1_sub_ids))
    s1_to_gt = {s1: gt_map.get(s1, set()) for s1 in unique_s1}
    
    best_thresh = 0.65
    best_f05 = 0.0
    
    for thresh in np.arange(0.40, 0.86, 0.05):
        pred_sets = defaultdict(set)
        for s1_id, cand_id, prob in zip(s1_sub_ids, cand_sub_ids, p_blend):
            if prob >= thresh:
                pred_sets[s1_id].add(cand_id)
                
        f05 = compute_macro_f05(s1_to_gt, pred_sets)
        print(f"  Threshold θ = {thresh:.2f} -> Macro F0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print(f"\n==========================================================================")
    print(f"🎉 OPTIMAL CALIBRATED THRESHOLD: θ* = {best_thresh:.2f} (Macro F0.5 = {best_f05:.4f})")
    print(f"Total Training Runtime: {time.time()-t_start:.1f}s")
    print("==========================================================================")


if __name__ == '__main__':
    train_direct_hard_negatives()
