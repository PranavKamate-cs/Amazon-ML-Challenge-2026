"""
Amazon ML Challenge 2026 - Ultra-Lean Hard Negative Blocker Ensemble Trainer (V5)
Pre-allocated 150 MB memory matrix with 65k queries & ~1M hard negative candidate pairs.
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


class MultiChannelBlockerV5:
    def __init__(self, max_token_freq: int = 3000, max_addr_freq: int = 1200):
        self.max_token_freq = max_token_freq
        self.max_addr_freq = max_addr_freq
        self.exact_name_map = defaultdict(list)
        self.prefix2_map = defaultdict(list)
        self.compact_name_map = defaultdict(list)
        self.brand_root_map = defaultdict(list)
        self.name_token_index = defaultdict(list)
        self.addr_combo_index = defaultdict(list)
        self.target_data = []

    def fit(self, entity_ids: List[str], names: List[str], addresses: List[str], roots: List[str], comps: List[str], numbers: List[str], postals: List[str]):
        token_doc_counts = defaultdict(int)
        for name in names:
            for t in set(str(name).split()):
                if len(t) >= 3:
                    token_doc_counts[t] += 1

        for idx in range(len(entity_ids)):
            eid = str(entity_ids[idx])
            name = str(names[idx]) if pd.notna(names[idx]) else ""
            addr = str(addresses[idx]) if pd.notna(addresses[idx]) else ""
            root = str(roots[idx]) if pd.notna(roots[idx]) else ""
            comp = str(comps[idx]) if pd.notna(comps[idx]) else ""
            num_val = numbers[idx]
            nums = frozenset(str(num_val).split()) if pd.notna(num_val) and num_val else frozenset()
            pin_val = postals[idx]
            pins = frozenset(str(pin_val).split()) if pd.notna(pin_val) and pin_val else frozenset()
            tokens = name.split()
            ngrams = frozenset(get_char_ngrams(name, n=3))
            is_s3 = 1.0 if eid.startswith('S3-') else 0.0
            
            self.target_data.append((
                eid, name, addr, root, comp, tokens, ngrams, nums, pins, is_s3,
                float(len(name)), float(len(addr))
            ))
            
            if name:
                self.exact_name_map[name].append(idx)
                if len(tokens) >= 2:
                    self.prefix2_map[f"{tokens[0]}_{tokens[1]}"].append(idx)
                    
            if comp and len(comp) >= 5:
                self.compact_name_map[comp].append(idx)
                
            if root and len(root) >= 4:
                self.brand_root_map[root].append(idx)
                
            for t in set(tokens):
                if len(t) >= 3 and token_doc_counts[t] <= self.max_token_freq:
                    self.name_token_index[t].append(idx)
                    
            addr_tokens = set([t for t in addr.split() if len(t) >= 4 and not t.isdigit()])
            for num in nums:
                for at in addr_tokens:
                    key = f"{num}_{at}"
                    self.addr_combo_index[key].append(idx)

    def retrieve_candidates(self, s1_name: str, s1_addr: str, s1_root: str, s1_comp: str, s1_tokens: List[str], top_k: int = 15) -> List[int]:
        scores = defaultdict(float)
        
        if s1_name in self.exact_name_map:
            for idx in self.exact_name_map[s1_name][:8]:
                scores[idx] += 6.0
                
        if len(s1_tokens) >= 2:
            p2 = f"{s1_tokens[0]}_{s1_tokens[1]}"
            if p2 in self.prefix2_map:
                for idx in self.prefix2_map[p2][:8]:
                    scores[idx] += 4.0
                    
        if s1_comp and len(s1_comp) >= 5 and s1_comp in self.compact_name_map:
            for idx in self.compact_name_map[s1_comp][:8]:
                scores[idx] += 5.0
                
        if s1_root and len(s1_root) >= 4 and s1_root in self.brand_root_map:
            for idx in self.brand_root_map[s1_root][:8]:
                scores[idx] += 4.5
                
        for t in set(s1_tokens):
            if len(t) >= 3 and t in self.name_token_index:
                postings = self.name_token_index[t]
                w = 1.0 / (1.0 + np.log1p(len(postings)))
                for idx in postings[:12]:
                    scores[idx] += (w * 1.5)
                    
        nums = re.findall(r'\b\d+\b', s1_addr)
        addr_tokens = set([t for t in s1_addr.split() if len(t) >= 4 and not t.isdigit()])
        for num in nums:
            for at in addr_tokens:
                key = f"{num}_{at}"
                if key in self.addr_combo_index:
                    postings = self.addr_combo_index[key]
                    if len(postings) <= self.max_addr_freq:
                        for idx in postings[:10]:
                            scores[idx] += 2.5
                        
        if not scores:
            return []
            
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [idx for idx, _ in top_items]


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


def train_hard_negative_ensemble(data_dir: str = 'student_resource/dataset', cache_dir: str = 'data_preprocessed', model_dir: str = 'models'):
    os.makedirs(model_dir, exist_ok=True)
    
    print("==========================================================================")
    print(">> AMAZON ML CHALLENGE 2026: MINING HARD NEGATIVES & TRAINING V5 ENSEMBLE")
    print("==========================================================================")
    
    # 1. Load Ground Truth
    gt_df = pd.read_csv(os.path.join(data_dir, 'train/train_ground_truth.tsv'), sep='\t')
    gt_map = {}
    for s1_id, matches_str in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        if pd.notna(matches_str) and str(matches_str).strip():
            gt_map[s1_id] = set([m.strip() for m in str(matches_str).split(',') if m.strip()])
        else:
            gt_map[s1_id] = set()
            
    print(f"Loaded {len(gt_map):,} ground truth mappings.")
    
    countries = ['US', 'France', 'India']
    samples_per_country = {'US': 25000, 'France': 15000, 'India': 25000}
    
    # Pre-allocate 1,000,000 rows
    MAX_ROWS = 1000000
    X = np.zeros((MAX_ROWS, 37), dtype=np.float32)
    y = np.zeros(MAX_ROWS, dtype=np.int32)
    row_idx = 0
    
    s1_id_list = []
    cand_id_list = []
    
    for country in countries:
        t0 = time.time()
        print(f"\n[Mining] Loading & Building Blocker for {country}...")
        
        s1_df = pd.read_parquet(os.path.join(cache_dir, 'train/source1.parquet'), filters=[('country', '==', country)])
        s2_df = pd.read_parquet(os.path.join(cache_dir, 'train/source2.parquet'), filters=[('country', '==', country)])
        s3_df = pd.read_parquet(os.path.join(cache_dir, 'train/source3.parquet'), filters=[('country', '==', country)])
        targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
        del s2_df, s3_df
        gc.collect()
        
        blocker = MultiChannelBlockerV5()
        blocker.fit(
            entity_ids=targets_df['entity_id'].tolist(),
            names=targets_df['c_name'].tolist(),
            addresses=targets_df['c_addr'].tolist(),
            roots=targets_df['brand_root'].tolist(),
            comps=targets_df['comp_name'].tolist(),
            numbers=targets_df['numbers'].tolist(),
            postals=targets_df['postals'].tolist()
        )
        del targets_df
        gc.collect()
        
        # Sample training queries
        np.random.seed(42)
        sample_size = min(samples_per_country[country], len(s1_df))
        s1_sample = s1_df.sample(n=sample_size, random_state=42)
        del s1_df
        gc.collect()
        
        print(f"Mining candidate pairs for {len(s1_sample):,} {country} queries...")
        country_pos = 0
        country_neg = 0
        
        for eid, name, addr, root, comp, nums_str, postals_str in zip(
            s1_sample['entity_id'], s1_sample['c_name'], s1_sample['c_addr'], s1_sample['brand_root'], s1_sample['comp_name'], s1_sample['numbers'], s1_sample['postals']
        ):
            if row_idx >= MAX_ROWS:
                break
                
            name_str = str(name) if pd.notna(name) else ""
            addr_str = str(addr) if pd.notna(addr) else ""
            root_str = str(root) if pd.notna(root) else ""
            comp_str = str(comp) if pd.notna(comp) else ""
            tokens = name_str.split()
            ngrams = frozenset(get_char_ngrams(name_str, n=3))
            nums = frozenset(str(nums_str).split()) if pd.notna(nums_str) and nums_str else frozenset()
            pins = frozenset(str(postals_str).split()) if pd.notna(postals_str) and postals_str else frozenset()
            
            true_matches = gt_map.get(eid, set())
            cand_indices = blocker.retrieve_candidates(name_str, addr_str, root_str, comp_str, tokens, top_k=15)
            
            for c_idx in cand_indices:
                if row_idx >= MAX_ROWS:
                    break
                cand_info = blocker.target_data[c_idx]
                cand_eid = cand_info[0]
                is_pos = 1 if cand_eid in true_matches else 0
                
                feat = extract_v5_features(
                    name_str, addr_str, root_str, comp_str, tokens, ngrams, nums, pins,
                    cand_eid, cand_info[1], cand_info[2], cand_info[3], cand_info[4], cand_info[5], cand_info[6], cand_info[7], cand_info[8]
                )
                X[row_idx] = feat
                y[row_idx] = is_pos
                row_idx += 1
                
                s1_id_list.append(eid)
                cand_id_list.append(cand_eid)
                
                if is_pos:
                    country_pos += 1
                else:
                    country_neg += 1
                    
        print(f"✓ {country} Mined: {country_pos:,} True Positives + {country_neg:,} Hard Negatives in {time.time()-t0:.1f}s")
        del blocker
        gc.collect()
        
    X = X[:row_idx]
    y = y[:row_idx]
    print(f"\n==========================================================================")
    print(f"TOTAL TRAINING MATRIX: {X.shape}, Positives: {np.sum(y):,}, Hard Negatives: {len(y)-np.sum(y):,}")
    print(f"==========================================================================")
    
    # 2. Train LightGBM Booster
    print("\n[Step 2/4] Training Model 1: LightGBM (1,000 Trees, 4 Threads)...")
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
    
    # 3. Train CatBoost Booster
    print("\n[Step 3/4] Training Model 2: CatBoost Classifier (1,000 Trees, 4 Threads)...")
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
    cb_path = os.path.join(model_dir, 'catboost_v5_hard.cbm')
    cb_model.save_model(cb_path)
    print(f"✓ CatBoost trained in {time.time()-t0:.1f}s -> Saved to {cb_path}")
    
    # 4. Cross-Validation Threshold Calibration for F0.5
    print("\n[Step 4/4] Cross-Validation F0.5 Threshold Calibration...")
    p_lgb = lgb_model.predict(X)
    p_cb = cb_model.predict_proba(X)[:, 1]
    p_blend = 0.5 * p_lgb + 0.5 * p_cb
    
    unique_s1 = list(set(s1_id_list))
    s1_to_gt = {s1: gt_map.get(s1, set()) for s1 in unique_s1}
    
    best_thresh = 0.65
    best_f05 = 0.0
    
    for thresh in np.arange(0.40, 0.85, 0.05):
        pred_sets = defaultdict(set)
        for s1_id, cand_id, prob in zip(s1_id_list, cand_id_list, p_blend):
            if prob >= thresh:
                pred_sets[s1_id].add(cand_id)
                
        f05 = compute_macro_f05(s1_to_gt, pred_sets)
        print(f"  Threshold θ = {thresh:.2f} -> Macro F0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print(f"\n==========================================================================")
    print(f"🎉 OPTIMAL CALIBRATED THRESHOLD: θ* = {best_thresh:.2f} (Macro F0.5 = {best_f05:.4f})")
    print("==========================================================================")


if __name__ == '__main__':
    train_hard_negative_ensemble()
