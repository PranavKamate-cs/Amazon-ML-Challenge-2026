"""
Amazon ML Challenge 2026 - Stage 1: Validation Tuning & Metric Calibration
Splits training data into 80% Train / 20% Val by S1 query groups.
Mines true positives + blocker hard-negatives matching the inference distribution.
Optimizes LightGBM hyperparameters & calibrates exact θ* on Macro F0.5.
"""

import time
import os
import sys
import gc
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz, distance

LEGAL_SUFFIXES_REGEX = re.compile(
    r'\b(pvt|ltd|limited|private|inc|incorporated|corp|corporation|llc|llp|gmbh|sa|sarl|srl|bv|co|company|plc|enterprises|enterprise|services|service|solutions|industries|holdings|group)\b',
    re.IGNORECASE
)


def clean_name(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\(\)\[\]\{\}\"\']', ' ', text)
    text = re.sub(r'[\/\\,\.\-_:;!@#$%^&*+=\|<>\?~`]', ' ', text)
    return " ".join(text.split())


def clean_address(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\/\\,\.\-_:;!@#$%^&*+=\|<>\(\)\[\]\{\}\"\']', ' ', text)
    return " ".join(text.split())


def clean_compact_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    return "".join([c for c in name.lower() if c.isalnum()])


def extract_numbers(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    return re.findall(r'\b\d+\b', text)


def extract_postal_codes(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    return re.findall(r'\b\d{5,6}\b', text)


def get_char_ngrams(text: str, n: int = 3) -> List[str]:
    if not text or len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


class MultiIndexBlocker:
    def __init__(self, max_token_freq: int = 4000, max_addr_freq: int = 1500):
        self.max_token_freq = max_token_freq
        self.max_addr_freq = max_addr_freq
        self.exact_name_map = defaultdict(list)
        self.prefix2_map = defaultdict(list)
        self.compact_name_map = defaultdict(list)
        self.name_token_index = defaultdict(list)
        self.addr_combo_index = defaultdict(list)
        self.target_data = []

    def fit_targets(self, entity_ids: List[str], names: List[str], addresses: List[str]):
        token_doc_counts = defaultdict(int)
        for name in names:
            tokens = set(name.split())
            for t in tokens:
                if len(t) >= 3:
                    token_doc_counts[t] += 1

        for idx in range(len(entity_ids)):
            eid = entity_ids[idx]
            name = names[idx]
            addr = addresses[idx]
            comp = clean_compact_name(name)
            tokens = name.split()
            ngrams = frozenset(get_char_ngrams(name, n=3))
            nums = frozenset(extract_numbers(addr))
            postals = frozenset(extract_postal_codes(addr))
            is_s3 = 1.0 if eid.startswith('S3-') else 0.0
            
            self.target_data.append((
                eid, name, addr, comp, tokens, ngrams, nums, postals, is_s3,
                float(len(name)), float(len(addr))
            ))
            
            if name:
                self.exact_name_map[name].append(idx)
                if len(tokens) >= 2:
                    self.prefix2_map[f"{tokens[0]}_{tokens[1]}"].append(idx)
                    
            if comp and len(comp) >= 5:
                self.compact_name_map[comp].append(idx)

            for t in set(tokens):
                if len(t) >= 3 and token_doc_counts[t] <= self.max_token_freq:
                    self.name_token_index[t].append(idx)
                    
            addr_tokens = set([t for t in addr.split() if len(t) >= 4 and not t.isdigit()])
            for num in nums:
                for at in addr_tokens:
                    key = f"{num}_{at}"
                    self.addr_combo_index[key].append(idx)

    def query(self, s1_name: str, s1_addr: str, s1_comp: str, s1_tokens: List[str], top_k: int = 20) -> List[int]:
        scores = defaultdict(float)
        
        if s1_name in self.exact_name_map:
            for idx in self.exact_name_map[s1_name][:8]:
                scores[idx] += 5.0
                
        if len(s1_tokens) >= 2:
            p2 = f"{s1_tokens[0]}_{s1_tokens[1]}"
            if p2 in self.prefix2_map:
                for idx in self.prefix2_map[p2][:8]:
                    scores[idx] += 3.5
                    
        if s1_comp and len(s1_comp) >= 5 and s1_comp in self.compact_name_map:
            for idx in self.compact_name_map[s1_comp][:8]:
                scores[idx] += 4.0

        for t in set(s1_tokens):
            if len(t) >= 3 and t in self.name_token_index:
                postings = self.name_token_index[t]
                w = 1.0 / (1.0 + np.log1p(len(postings)))
                for idx in postings[:15]:
                    scores[idx] += (w * 1.5)
                    
        nums = extract_numbers(s1_addr)
        addr_tokens = set([t for t in s1_addr.split() if len(t) >= 4 and not t.isdigit()])
        for num in nums:
            for at in addr_tokens:
                key = f"{num}_{at}"
                if key in self.addr_combo_index:
                    postings = self.addr_combo_index[key]
                    if len(postings) <= self.max_addr_freq:
                        for idx in postings[:10]:
                            scores[idx] += 2.0
                        
        if not scores:
            return []
            
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [idx for idx, _ in top_items]


def fast_extract_features(s1_info, cand_info):
    s1_id, s1_name, s1_addr, s1_comp, s1_tokens, s1_ngrams, s1_nums, s1_postals, len1_n, len1_a = s1_info
    cand_eid, cand_name, cand_addr, cand_comp, cand_tokens, cand_ngrams, cand_nums, cand_postals, is_s3, len2_n, len2_a = cand_info
    
    nr = fuzz.ratio(s1_name, cand_name) / 100.0
    ntset = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    
    if nr < 0.35 and ntset < 0.40 and s1_name != cand_name:
        return None
        
    npr = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    nts = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    nwr = fuzz.WRatio(s1_name, cand_name) / 100.0
    nexact = 1.0 if s1_name == cand_name else 0.0
    nldiff = abs(len1_n - len2_n)
    nlratio = min(len1_n, len2_n) / max(len1_n, len2_n) if max(len1_n, len2_n) > 0 else 1.0
    
    nfirst = 1.0 if (s1_tokens and cand_tokens and s1_tokens[0] == cand_tokens[0]) else 0.0
    nlast = 1.0 if (s1_tokens and cand_tokens and s1_tokens[-1] == cand_tokens[-1]) else 0.0
    
    if s1_ngrams and cand_ngrams:
        common_ng = len(s1_ngrams.intersection(cand_ngrams))
        ncjacc = common_ng / (len(s1_ngrams.union(cand_ngrams)) + 1e-5)
    else:
        ncjacc = 0.0
        
    ncompact = fuzz.ratio(s1_comp, cand_comp) / 100.0 if (s1_comp and cand_comp) else 0.0
    
    addr_missing = 1.0 if (not s1_addr or not cand_addr) else 0.0
    if addr_missing:
        ar = atset = ats = apr = afirst = num_c = num_j = num_p = pin_match = pin_both = 0.0
    else:
        ar = fuzz.ratio(s1_addr, cand_addr) / 100.0
        atset = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        ats = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        apr = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
        afirst = 1.0 if (s1_addr[:6] == cand_addr[:6]) else 0.0
        
        if s1_nums and cand_nums:
            c_nums = s1_nums.intersection(cand_nums)
            num_c = float(len(c_nums))
            num_j = len(c_nums) / len(s1_nums.union(cand_nums))
            num_p = 1.0 if len(c_nums) > 0 else -1.0
        else:
            num_c = num_j = num_p = 0.0
            
        if s1_postals and cand_postals:
            c_pins = s1_postals.intersection(cand_postals)
            pin_match = 1.0 if len(c_pins) > 0 else -1.0
            pin_both = 1.0
        else:
            pin_match = pin_both = 0.0
            
    name_addr_avg = (nr + ar) / 2.0 if not addr_missing else nr
    
    return np.array([
        nr, npr, nts, ntset, nwr, nexact, nldiff, nlratio,
        nfirst, nlast, ncjacc, ncompact,
        ar, atset, ats, apr, addr_missing, afirst,
        num_c, num_j, num_p, pin_match, pin_both,
        name_addr_avg, is_s3,
        len1_n, len2_n, len1_a
    ], dtype=np.float32)


def compute_macro_f05(y_true_sets: Dict[str, Set[str]], y_pred_sets: Dict[str, Set[str]]) -> float:
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


def run_stage1_tuning(
    data_dir: str = 'student_resource/dataset',
    cache_dir: str = 'data_preprocessed',
    model_dir: str = 'models',
    val_sample_size: int = 50000
):
    os.makedirs(model_dir, exist_ok=True)
    print("==========================================================================")
    print(">> AMAZON ML CHALLENGE 2026: STAGE 1 HYPERPARAMETER TUNING & CALIBRATION")
    print("==========================================================================")
    
    # 1. Load Ground Truth
    print("\n[Step 1/5] Loading Training Ground Truth...")
    gt_df = pd.read_csv(os.path.join(data_dir, 'train/train_ground_truth.tsv'), sep='\t')
    gt_map = defaultdict(set)
    for s1_id, matches_str in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        if pd.notna(matches_str) and str(matches_str).strip():
            matches = [m.strip() for m in str(matches_str).split(',') if m.strip()]
            for m in matches:
                gt_map[s1_id].add(m)
        else:
            gt_map[s1_id] = set()
            
    print(f"Loaded ground truth for {len(gt_map):,} queries ({sum(1 for s in gt_map.values() if not s):,} singletons).")
    
    # 2. Split S1 queries into 80% Train / 20% Validation
    np.random.seed(42)
    all_s1_ids = list(gt_map.keys())
    np.random.shuffle(all_s1_ids)
    
    # For fast tuning, use val_sample_size queries for validation and 200k for training
    val_s1_set = set(all_s1_ids[:val_sample_size])
    train_s1_set = set(all_s1_ids[val_sample_size:val_sample_size + 200000])
    
    val_gt_map = {s1: gt_map[s1] for s1 in val_s1_set}
    
    # 3. Process partition by partition to extract Train Features & Candidates
    print(f"\n[Step 2/5] Mining Training Pairs (Positives + Blocker Hard Negatives)...")
    train_dir = os.path.join(cache_dir, 'train')
    
    X_train_list = []
    y_train_list = []
    
    val_s1_tuples_by_country = defaultdict(list)
    val_blockers_by_country = {}
    
    for country in ['US', 'France', 'India']:
        print(f"\n--- Processing {country} ---")
        t0 = time.time()
        c_s1 = pd.read_parquet(os.path.join(train_dir, 'source1.parquet'), filters=[('country', '==', country)])
        c_s2 = pd.read_parquet(os.path.join(train_dir, 'source2.parquet'), filters=[('country', '==', country)])
        c_s3 = pd.read_parquet(os.path.join(train_dir, 'source3.parquet'), filters=[('country', '==', country)])
        c_targets = pd.concat([c_s2, c_s3], ignore_index=True)
        del c_s2, c_s3
        gc.collect()
        
        blocker = MultiIndexBlocker()
        blocker.fit_targets(
            entity_ids=c_targets['entity_id'].tolist(),
            names=c_targets['c_name'].tolist(),
            addresses=c_targets['c_addr'].tolist()
        )
        del c_targets
        gc.collect()
        
        # Build target lookup map
        target_lookup = {item[0]: item for item in blocker.target_data}
        
        # Extract metadata
        country_train_s1 = []
        country_val_s1 = []
        for eid, name, addr in zip(c_s1['entity_id'], c_s1['c_name'], c_s1['c_addr']):
            eid_str = str(eid)
            name_str = str(name) if pd.notna(name) else ""
            addr_str = str(addr) if pd.notna(addr) else ""
            comp_str = clean_compact_name(name_str)
            tokens = name_str.split()
            ngrams = frozenset(get_char_ngrams(name_str, n=3))
            nums = frozenset(extract_numbers(addr_str))
            pins = frozenset(extract_postal_codes(addr_str))
            tup = (
                eid_str, name_str, addr_str, comp_str, tokens, ngrams, nums, pins,
                float(len(name_str)), float(len(addr_str))
            )
            if eid_str in train_s1_set:
                country_train_s1.append(tup)
            elif eid_str in val_s1_set:
                country_val_s1.append(tup)
                
        del c_s1
        gc.collect()
        
        print(f"  [{country}] {len(country_train_s1):,} Train S1, {len(country_val_s1):,} Val S1.")
        
        # Store validation entities and blocker
        val_s1_tuples_by_country[country] = country_val_s1
        val_blockers_by_country[country] = blocker
        
        # Mine training pairs
        print(f"  Mining training pairs for {len(country_train_s1):,} queries...")
        pos_cnt = 0
        neg_cnt = 0
        
        for s1_info in country_train_s1:
            s1_id = s1_info[0]
            true_matches = gt_map.get(s1_id, set())
            
            # 1. Add Ground Truth Positives
            for tm_id in true_matches:
                if tm_id in target_lookup:
                    cand_info = target_lookup[tm_id]
                    feat = fast_extract_features(s1_info, cand_info)
                    if feat is not None:
                        X_train_list.append(feat)
                        y_train_list.append(1)
                        pos_cnt += 1
                        
            # 2. Add Blocker Negatives
            cand_indices = blocker.query(s1_info[1], s1_info[2], s1_info[3], s1_info[4], top_k=10)
            added_neg = 0
            for c_idx in cand_indices:
                cand_info = blocker.target_data[c_idx]
                cand_eid = cand_info[0]
                if cand_eid not in true_matches:
                    feat = fast_extract_features(s1_info, cand_info)
                    if feat is not None:
                        X_train_list.append(feat)
                        y_train_list.append(0)
                        neg_cnt += 1
                        added_neg += 1
                        if added_neg >= 4:
                            break
                            
        print(f"  [{country}] Generated {pos_cnt:,} positives, {neg_cnt:,} hard negatives in {time.time()-t0:.2f}s.")
        
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    del X_train_list, y_train_list
    gc.collect()
    
    print(f"\n[Step 3/5] Assembled Training Matrix: {X_train.shape} (Positives: {np.sum(y_train):,}, Negatives: {len(y_train)-np.sum(y_train):,})")
    
    # 4. Hyperparameter Grid Evaluation
    print(f"\n[Step 4/5] Hyperparameter Grid Search on Macro F0.5 Validation Set...")
    
    param_candidates = [
        {'num_leaves': 63, 'max_depth': 8, 'learning_rate': 0.05, 'feature_fraction': 0.85, 'min_child_samples': 30, 'n_estimators': 600, 'name': 'Config-A (Balanced)'},
        {'num_leaves': 127, 'max_depth': 10, 'learning_rate': 0.04, 'feature_fraction': 0.80, 'min_child_samples': 20, 'n_estimators': 800, 'name': 'Config-B (Deep Pro)'},
        {'num_leaves': 45, 'max_depth': 7, 'learning_rate': 0.06, 'feature_fraction': 0.90, 'min_child_samples': 50, 'n_estimators': 500, 'name': 'Config-C (High-Precision)'},
    ]
    
    best_config = None
    best_val_f05 = -1.0
    best_threshold = 0.70
    
    for cfg in param_candidates:
        print(f"\n>> Training Model with {cfg['name']}: {cfg}")
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': cfg['num_leaves'],
            'max_depth': cfg['max_depth'],
            'learning_rate': cfg['learning_rate'],
            'feature_fraction': cfg['feature_fraction'],
            'min_child_samples': cfg['min_child_samples'],
            'n_estimators': cfg['n_estimators'],
            'lambda_l1': 0.05,
            'lambda_l2': 0.1,
            'verbose': -1,
            'n_jobs': 4,
            'random_state': 42
        }
        
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train)
        
        # Evaluate end-to-end on validation set
        print(f">> Evaluating {cfg['name']} on {len(val_s1_set):,} validation queries...")
        val_scores_by_threshold = defaultdict(dict)
        thresholds_to_test = [0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78]
        val_preds = {t: {} for t in thresholds_to_test}
        
        for country, val_s1_list in val_s1_tuples_by_country.items():
            blocker = val_blockers_by_country[country]
            
            # Predict in batches
            batch_size = 5000
            for start_idx in range(0, len(val_s1_list), batch_size):
                batch = val_s1_list[start_idx:start_idx+batch_size]
                batch_feats = []
                batch_meta = []
                
                for b_idx, s1_info in enumerate(batch):
                    cand_indices = blocker.query(s1_info[1], s1_info[2], s1_info[3], s1_info[4], top_k=20)
                    for c_idx in cand_indices:
                        cand_info = blocker.target_data[c_idx]
                        cand_eid = cand_info[0]
                        feat = fast_extract_features(s1_info, cand_info)
                        if feat is not None:
                            batch_feats.append(feat)
                            batch_meta.append((s1_info[0], cand_eid, s1_info, cand_info))
                            
                scored_map = defaultdict(list)
                if batch_feats:
                    probs = model.predict_proba(np.array(batch_feats, dtype=np.float32))[:, 1]
                    for (s1_id, cand_eid, s1_info, cand_info), prob in zip(batch_meta, probs):
                        # Constraints
                        s1_postals = s1_info[7]
                        cand_postals = cand_info[7]
                        s1_nums = s1_info[6]
                        cand_nums = cand_info[6]
                        if s1_postals and cand_postals and len(s1_postals.intersection(cand_postals)) == 0 and fuzz.ratio(s1_info[1], cand_info[1]) < 90:
                            continue
                        if s1_nums and cand_nums and len(s1_nums.intersection(cand_nums)) == 0 and fuzz.ratio(s1_info[1], cand_info[1]) < 85:
                            continue
                        scored_map[s1_id].append((cand_eid, prob))
                        
                for s1_info in batch:
                    s1_id = s1_info[0]
                    cands = scored_map.get(s1_id, [])
                    cands.sort(key=lambda x: x[1], reverse=True)
                    
                    for thresh in thresholds_to_test:
                        s2_m, s3_m = [], []
                        for ceid, prob in cands:
                            if prob >= thresh:
                                if ceid.startswith('S2-') and len(s2_m) < 3:
                                    s2_m.append(ceid)
                                elif ceid.startswith('S3-') and len(s3_m) < 3:
                                    s3_m.append(ceid)
                        val_preds[thresh][s1_id] = set(s2_m + s3_m)
                        
        print(f"--- Macro F0.5 Results for {cfg['name']} ---")
        for thresh in thresholds_to_test:
            f05 = compute_macro_f05(val_gt_map, val_preds[thresh])
            matches_cnt = sum(len(s) for s in val_preds[thresh].values())
            singletons_cnt = sum(1 for s in val_preds[thresh].values() if not s)
            print(f"  θ = {thresh:.2f} | Val F0.5: {f05:.4f} | Total Matches: {matches_cnt:,} | Singletons: {singletons_cnt:,} ({singletons_cnt/len(val_s1_set)*100:.2f}%)")
            
            if f05 > best_val_f05:
                best_val_f05 = f05
                best_config = cfg
                best_threshold = thresh
                
    print("\n==========================================================================")
    print(f"✓ BEST ARCHITECTURE DISCOVERED IN STAGE 1:")
    print(f"  Config: {best_config['name']}")
    print(f"  Optimal Decision Threshold θ*: {best_threshold:.2f}")
    print(f"  Peak Validation Macro F0.5: {best_val_f05:.4f}")
    print("==========================================================================")
    
    # Save best config summary
    with open(os.path.join(model_dir, 'stage1_best_config.txt'), 'w') as f:
        f.write(f"best_config={best_config}\n")
        f.write(f"best_threshold={best_threshold}\n")
        f.write(f"best_val_f05={best_val_f05}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-size', type=int, default=30000, help='Number of validation queries')
    args = parser.parse_args()
    
    run_stage1_tuning(val_sample_size=args.val_size)
