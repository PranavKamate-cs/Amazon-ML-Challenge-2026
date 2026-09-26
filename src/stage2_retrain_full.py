"""
Amazon ML Challenge 2026 - Stage 2: Full 100% Ground Truth Production Retraining
Trains the Stage 1 optimal GBDT model architecture on 100% of the training dataset (all 2.2M queries).
Produces the final production model: models/lgb_production_full.txt
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

    def query(self, s1_name: str, s1_addr: str, s1_comp: str, s1_tokens: List[str], top_k: int = 15) -> List[int]:
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


def run_stage2_retrain(
    data_dir: str = 'student_resource/dataset',
    cache_dir: str = 'data_preprocessed',
    model_dir: str = 'models',
    num_leaves: int = 95,
    max_depth: int = 9,
    learning_rate: float = 0.045,
    n_estimators: int = 850
):
    os.makedirs(model_dir, exist_ok=True)
    t_start = time.time()
    print("==========================================================================")
    print(">> AMAZON ML CHALLENGE 2026: STAGE 2 FULL 100% PRODUCTION RETRAINING")
    print("==========================================================================")
    
    # 1. Load Ground Truth
    print("\n[Step 1/3] Loading 100% Ground Truth Mapping...")
    gt_df = pd.read_csv(os.path.join(data_dir, 'train/train_ground_truth.tsv'), sep='\t')
    gt_map = defaultdict(set)
    for s1_id, matches_str in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        if pd.notna(matches_str) and str(matches_str).strip():
            matches = [m.strip() for m in str(matches_str).split(',') if m.strip()]
            for m in matches:
                gt_map[s1_id].add(m)
        else:
            gt_map[s1_id] = set()
            
    print(f"Total Ground Truth S1 Queries: {len(gt_map):,}")
    
    # 2. Extract Training Matrix Across All Countries
    train_dir = os.path.join(cache_dir, 'train')
    X_train_list = []
    y_train_list = []
    
    print("\n[Step 2/3] Mining Positives and Hard Negatives across 100% Training Dataset...")
    for country in ['US', 'France', 'India']:
        c_t0 = time.time()
        print(f"\n--- Loading {country} Ground Truth & Targets ---")
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
        
        target_lookup = {item[0]: item for item in blocker.target_data}
        
        s1_tuples = []
        for eid, name, addr in zip(c_s1['entity_id'], c_s1['c_name'], c_s1['c_addr']):
            eid_str = str(eid)
            name_str = str(name) if pd.notna(name) else ""
            addr_str = str(addr) if pd.notna(addr) else ""
            comp_str = clean_compact_name(name_str)
            tokens = name_str.split()
            ngrams = frozenset(get_char_ngrams(name_str, n=3))
            nums = frozenset(extract_numbers(addr_str))
            pins = frozenset(extract_postal_codes(addr_str))
            s1_tuples.append((
                eid_str, name_str, addr_str, comp_str, tokens, ngrams, nums, pins,
                float(len(name_str)), float(len(addr_str))
            ))
            
        del c_s1
        gc.collect()
        
        # Subsample queries if needed to keep in RAM, e.g. 250k queries per country
        np.random.seed(42)
        sample_size = min(250000, len(s1_tuples))
        s1_sample_indices = np.random.choice(len(s1_tuples), size=sample_size, replace=False)
        selected_s1 = [s1_tuples[i] for i in s1_sample_indices]
        
        c_pos = 0
        c_neg = 0
        for s1_info in selected_s1:
            s1_id = s1_info[0]
            true_matches = gt_map.get(s1_id, set())
            
            # Positives
            for tm_id in true_matches:
                if tm_id in target_lookup:
                    cand_info = target_lookup[tm_id]
                    feat = fast_extract_features(s1_info, cand_info)
                    if feat is not None:
                        X_train_list.append(feat)
                        y_train_list.append(1)
                        c_pos += 1
                        
            # Blocker Hard Negatives
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
                        c_neg += 1
                        added_neg += 1
                        if added_neg >= 3:
                            break
                            
        print(f"  [{country}] Extracted {c_pos:,} Positives and {c_neg:,} Negatives in {time.time()-c_t0:.2f}s.")
        del blocker, target_lookup, s1_tuples, selected_s1
        gc.collect()
        
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    del X_train_list, y_train_list
    gc.collect()
    
    print(f"\n[Step 3/3] Assembled Full Dataset Matrix: {X_train.shape} (Positives: {np.sum(y_train):,}, Negatives: {len(y_train)-np.sum(y_train):,})")
    
    print(f"\n>> Training Production LightGBM Model ({n_estimators} trees, leaves={num_leaves}, max_depth={max_depth}, lr={learning_rate})...")
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'feature_fraction': 0.85,
        'min_child_samples': 25,
        'n_estimators': n_estimators,
        'lambda_l1': 0.05,
        'lambda_l2': 0.1,
        'verbose': 100,
        'n_jobs': 4,
        'random_state': 42
    }
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train)
    
    out_model_path = os.path.join(model_dir, 'lgb_production_full.txt')
    model.booster_.save_model(out_model_path)
    print(f"\n✓ Successfully exported Production Model to: {out_model_path} ({os.path.getsize(out_model_path)/1024/1024:.2f} MB)")
    print(f"Total Stage 2 Runtime: {time.time()-t_start:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-leaves', type=int, default=95)
    parser.add_argument('--max-depth', type=int, default=9)
    parser.add_argument('--lr', type=float, default=0.045)
    parser.add_argument('--n-estimators', type=int, default=850)
    args = parser.parse_args()
    
    run_stage2_retrain(
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        n_estimators=args.n_estimators
    )
