"""
Amazon ML Challenge 2026 - Pipeline V3 (High-Precision GBDT Matcher)
Features:
- Multi-Channel Blocking (Exact, Prefix-2, BM25 Name Tokens, Number+Locality, Char-3grams)
- 100% GBDT Model Scoring for ALL candidate pairs (Zero unverified fast-paths)
- 28-D Rich RapidFuzz + Address + Postal/Number Feature Engineering
- Calibrated High-Precision Threshold (θ = 0.68) for Macro F_0.5 Optimization
- Zero-copy, Memory-Safe Streaming Batch Inference (~20 min execution on EC2)
"""

import time
import os
import sys
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from tqdm import tqdm

from src.preprocessing import clean_name, clean_address, extract_numbers
from src.metrics import compute_macro_f05


FEATURE_NAMES_V3 = [
    'name_ratio', 'name_partial_ratio', 'name_token_sort', 'name_token_set',
    'name_wratio', 'name_exact_match', 'name_len_diff', 'name_len_ratio',
    'name_first_token_match', 'name_last_token_match', 'name_char_jaccard_3gram',
    'name_compact_ratio',
    'addr_ratio', 'addr_token_set', 'addr_token_sort', 'addr_partial_ratio',
    'addr_is_missing', 'addr_first_token_match',
    'num_common_count', 'num_jaccard', 'num_mismatch_penalty',
    'pin_exact_match', 'pin_both_present',
    'name_addr_avg', 'is_s3',
    'len_s1_name', 'len_cand_name', 'len_s1_addr'
]


def extract_postal_codes(address: str) -> Set[str]:
    """Extracts 5-6 digit PIN / Zip codes from address string."""
    if not address:
        return set()
    pins = re.findall(r'\b\d{5,6}\b', address)
    return set(pins)


def clean_compact_name(name: str) -> str:
    """Removes all non-alphanumeric chars for robust domain/name matching."""
    if not name:
        return ""
    return "".join([c for c in name.lower() if c.isalnum()])


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams."""
    if not text or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def extract_features_v3(
    s1_name: str, s1_addr: str,
    cand_name: str, cand_addr: str,
    cand_id: str
) -> np.ndarray:
    """Extracts 28 discriminative features for a candidate pair."""
    len1_n, len2_n = len(s1_name), len(cand_name)
    len1_a, len2_a = len(s1_addr), len(cand_addr)
    
    # 1. Name Features
    if not s1_name or not cand_name:
        nr = npr = nts = ntset = nwr = nexact = nlratio = nfirst = nlast = ncjacc = ncompact = 0.0
        nldiff = abs(len1_n - len2_n)
    else:
        nr = fuzz.ratio(s1_name, cand_name) / 100.0
        npr = fuzz.partial_ratio(s1_name, cand_name) / 100.0
        nts = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
        ntset = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
        nwr = fuzz.WRatio(s1_name, cand_name) / 100.0
        nexact = 1.0 if s1_name == cand_name else 0.0
        nldiff = abs(len1_n - len2_n)
        nlratio = min(len1_n, len2_n) / max(len1_n, len2_n)
        
        tokens1 = s1_name.split()
        tokens2 = cand_name.split()
        nfirst = 1.0 if (tokens1 and tokens2 and tokens1[0] == tokens2[0]) else 0.0
        nlast = 1.0 if (tokens1 and tokens2 and tokens1[-1] == tokens2[-1]) else 0.0
        
        ng1 = get_char_ngrams(s1_name, n=3)
        ng2 = get_char_ngrams(cand_name, n=3)
        ncjacc = len(ng1.intersection(ng2)) / (len(ng1.union(ng2)) + 1e-5)
        
        comp1 = "".join([c for c in s1_name if c.isalnum()])
        comp2 = "".join([c for c in cand_name if c.isalnum()])
        ncompact = fuzz.ratio(comp1, comp2) / 100.0

    # 2. Address Features
    addr_missing = 1.0 if (not s1_addr or not cand_addr) else 0.0
    if addr_missing:
        ar = atset = ats = apr = afirst = num_c = num_j = num_p = pin_match = pin_both = 0.0
    else:
        ar = fuzz.ratio(s1_addr, cand_addr) / 100.0
        atset = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        ats = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        apr = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
        
        atok1 = s1_addr.split()
        atok2 = cand_addr.split()
        afirst = 1.0 if (atok1 and atok2 and atok1[0] == atok2[0]) else 0.0
        
        nums1 = extract_numbers(s1_addr)
        nums2 = extract_numbers(cand_addr)
        if nums1 and nums2:
            common = nums1.intersection(nums2)
            num_c = float(len(common))
            num_j = len(common) / len(nums1.union(nums2))
            num_p = 1.0 if len(common) > 0 else -1.0
        else:
            num_c = num_j = num_p = 0.0
            
        pin1 = extract_postal_codes(s1_addr)
        pin2 = extract_postal_codes(cand_addr)
        if pin1 and pin2:
            pin_match = 1.0 if len(pin1.intersection(pin2)) > 0 else -1.0
            pin_both = 1.0
        else:
            pin_match = pin_both = 0.0

    name_addr_avg = (nr + ar) / 2.0 if not addr_missing else nr
    is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0

    return np.array([
        nr, npr, nts, ntset, nwr, nexact, nldiff, nlratio,
        nfirst, nlast, ncjacc, ncompact,
        ar, atset, ats, apr, addr_missing, afirst,
        num_c, num_j, num_p, pin_match, pin_both,
        name_addr_avg, is_s3,
        float(len1_n), float(len2_n), float(len1_a)
    ], dtype=np.float32)


# --- HIGH-RECALL MULTI-CHANNEL BLOCKER ---
class HighRecallBlocker:
    def __init__(self, max_token_freq: int = 15000):
        self.max_token_freq = max_token_freq
        self.exact_name_map = defaultdict(list)
        self.prefix2_map = defaultdict(list)
        self.compact_name_map = defaultdict(list)
        self.name_token_index = defaultdict(list)
        self.addr_combo_index = defaultdict(list)
        self.name_ngram_index = defaultdict(list)
        self.target_data = [] # stores (eid, name, addr, nums, postals)

    def fit(self, entity_ids: List[str], names: List[str], addresses: List[str]):
        """Builds multi-channel inverted indices."""
        token_doc_counts = defaultdict(int)
        ngram_doc_counts = defaultdict(int)
        
        for name in names:
            for t in set(name.split()):
                if len(t) >= 3:
                    token_doc_counts[t] += 1
            for ng in get_char_ngrams(name, n=3):
                ngram_doc_counts[ng] += 1

        for idx, (eid, name, addr) in enumerate(zip(entity_ids, names, addresses)):
            nums = extract_numbers(addr)
            postals = extract_postal_codes(addr)
            comp = clean_compact_name(name)
            self.target_data.append((eid, name, addr, nums, postals))
            
            if name:
                self.exact_name_map[name].append(idx)
                words = name.split()
                if len(words) >= 2:
                    self.prefix2_map[f"{words[0]}_{words[1]}"].append(idx)
                    
            if comp and len(comp) >= 5:
                self.compact_name_map[comp].append(idx)
                
            for t in set(name.split()):
                if len(t) >= 3 and token_doc_counts[t] <= self.max_token_freq:
                    self.name_token_index[t].append(idx)
                    
            # Number + locality token combo index
            addr_tokens = set([t for t in addr.split() if len(t) >= 4 and not t.isdigit()])
            for num in nums:
                for at in addr_tokens:
                    key = f"{num}_{at}"
                    self.addr_combo_index[key].append(idx)
                    
            for ng in get_char_ngrams(name, n=3):
                if ngram_doc_counts[ng] <= self.max_token_freq:
                    self.name_ngram_index[ng].append(idx)

    def retrieve_candidates(self, s1_name: str, s1_addr: str, top_k: int = 25) -> List[int]:
        """Retrieves candidate target indices using multi-channel accumulated scores."""
        scores = defaultdict(float)
        
        # 1. Exact Name & Prefix-2
        if s1_name in self.exact_name_map:
            for idx in self.exact_name_map[s1_name][:10]:
                scores[idx] += 5.0
                
        words = s1_name.split()
        if len(words) >= 2:
            p2 = f"{words[0]}_{words[1]}"
            if p2 in self.prefix2_map:
                for idx in self.prefix2_map[p2][:10]:
                    scores[idx] += 3.5
                    
        # 2. Compact Domain Name
        comp = clean_compact_name(s1_name)
        if comp and len(comp) >= 5 and comp in self.compact_name_map:
            for idx in self.compact_name_map[comp][:10]:
                scores[idx] += 4.0

        # 3. Informative Name Tokens with IDF
        for t in set(words):
            if len(t) >= 3 and t in self.name_token_index:
                postings = self.name_token_index[t]
                w = 1.0 / (1.0 + np.log1p(len(postings)))
                for idx in postings:
                    scores[idx] += (w * 1.5)
                    
        # 4. Number + Locality combo
        nums = extract_numbers(s1_addr)
        addr_tokens = set([t for t in s1_addr.split() if len(t) >= 4 and not t.isdigit()])
        for num in nums:
            for at in addr_tokens:
                key = f"{num}_{at}"
                if key in self.addr_combo_index:
                    for idx in self.addr_combo_index[key]:
                        scores[idx] += 2.0
                        
        # 5. Char-3grams for fuzzy typos
        for ng in get_char_ngrams(s1_name, n=3):
            if ng in self.name_ngram_index:
                postings = self.name_ngram_index[ng]
                if len(postings) <= self.max_token_freq:
                    w = 0.25 / (1.0 + np.log1p(len(postings)))
                    for idx in postings:
                        scores[idx] += w
                        
        if not scores:
            return []
            
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [idx for idx, _ in top_items]


# --- PIPELINE RUNNER ---
def run_pipeline_v3(data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("=========================================================")
    print(">> AMAZON ML CHALLENGE 2026: HIGH-PRECISION PIPELINE V3")
    print("=========================================================")
    
    model_path = 'models/lgb_matcher_v3.txt'
    if not os.path.exists(model_path):
        print("\n[Step 1/2] Training High-Precision 28-D Matcher on Ground Truth...")
        train_s1 = pd.read_csv(os.path.join(data_dir, 'train/train_source1.tsv'), sep='\t', nrows=60000)
        train_s2 = pd.read_csv(os.path.join(data_dir, 'train/train_source2.tsv'), sep='\t')
        train_s3 = pd.read_csv(os.path.join(data_dir, 'train/train_source3.tsv'), sep='\t')
        train_gt = pd.read_csv(os.path.join(data_dir, 'train/train_ground_truth.tsv'), sep='\t')
        
        train_gt['matched_entity_ids'] = train_gt['matched_entity_ids'].fillna('')
        gt_map = {row['source1_entity_id']: set([m.strip() for m in row['matched_entity_ids'].split(',') if m.strip()]) for _, row in train_gt.iterrows()}
        
        train_targets = pd.concat([train_s2, train_s3], ignore_index=True)
        train_targets['c_name'] = train_targets['business_name'].apply(clean_name)
        train_targets['c_addr'] = train_targets['business_address'].apply(clean_address)
        target_dict = {row['entity_id']: (row['c_name'], row['c_addr']) for _, row in train_targets.iterrows()}
        
        train_s1_us = train_s1[train_s1['country'] == 'US'].copy()
        train_s1_us['c_name'] = train_s1_us['business_name'].apply(clean_name)
        train_s1_us['c_addr'] = train_s1_us['business_address'].apply(clean_address)
        
        us_targets = train_targets[train_targets['country'] == 'US']
        blocker = HighRecallBlocker()
        blocker.fit(us_targets['entity_id'].tolist(), us_targets['c_name'].tolist(), us_targets['c_addr'].tolist())
        
        X_list, y_list = [], []
        for _, row in tqdm(train_s1_us.iterrows(), total=len(train_s1_us), desc="Building Training Dataset"):
            s1_id, s1_name, s1_addr = row['entity_id'], row['c_name'], row['c_addr']
            true_matches = gt_map.get(s1_id, set())
            cand_indices = blocker.retrieve_candidates(s1_name, s1_addr, top_k=25)
            cand_eids = {blocker.target_data[idx][0] for idx in cand_indices}.union(true_matches)
            
            for cid in cand_eids:
                if cid not in target_dict:
                    continue
                cname, caddr = target_dict[cid]
                X_list.append(extract_features_v3(s1_name, s1_addr, cname, caddr, cid))
                y_list.append(1.0 if cid in true_matches else 0.0)
                
        X_train = np.array(X_list, dtype=np.float32)
        y_train = np.array(y_list, dtype=np.float32)
        print(f"Generated {len(X_train):,} training pairs (Positives: {int(y_train.sum()):,}, Negatives: {int((1-y_train).sum()):,})")
        
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES_V3)
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.07, 'num_leaves': 45,
            'feature_fraction': 0.85, 'bagging_fraction': 0.85,
            'bagging_freq': 5, 'verbose': -1, 'random_state': 42
        }
        model = lgb.train(params, dtrain, num_boost_round=350)
        model.save_model(model_path)
        print(f"Model successfully saved to {model_path}!")
    else:
        print(f"\n[Step 1/2] Loading model from {model_path}...")
        model = lgb.Booster(model_file=model_path)

    # 2. Test Set Inference
    print("\n[Step 2/2] Loading Test Datasets for 100% GBDT Scoring...")
    test_s1 = pd.read_csv(os.path.join(data_dir, 'test/test_source1.tsv'), sep='\t')
    test_s2 = pd.read_csv(os.path.join(data_dir, 'test/test_source2.tsv'), sep='\t')
    test_s3 = pd.read_csv(os.path.join(data_dir, 'test/test_source3.tsv'), sep='\t')
    
    print("Normalizing test names & addresses...")
    test_s1['c_name'] = test_s1['business_name'].apply(clean_name)
    test_s1['c_addr'] = test_s1['business_address'].apply(clean_address)
    
    test_targets = pd.concat([test_s2, test_s3], ignore_index=True)
    test_targets['c_name'] = test_targets['business_name'].apply(clean_name)
    test_targets['c_addr'] = test_targets['business_address'].apply(clean_address)
    
    matching_out_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_out_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    
    with open(matching_out_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_out_path, 'w', encoding='utf-8') as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
    countries = ['US', 'France', 'India']
    decision_threshold = 0.68  # High-Precision Calibrated Threshold for F_0.5
    batch_size = 5000
    
    for country in countries:
        print(f"\n==========================================")
        print(f">> Processing Country: {country}")
        print(f"==========================================")
        
        c_s1 = test_s1[test_s1['country'] == country].copy()
        c_targets = test_targets[test_targets['country'] == country].copy()
        
        print(f"Indexing {country} targets ({len(c_targets):,} records)...")
        blocker = HighRecallBlocker()
        blocker.fit(
            entity_ids=c_targets['entity_id'].tolist(),
            names=c_targets['c_name'].tolist(),
            addresses=c_targets['c_addr'].tolist()
        )
        
        s1_rows = list(zip(c_s1['entity_id'], c_s1['c_name'], c_s1['c_addr']))
        start_c_t = time.time()
        
        with open(matching_out_path, 'a', encoding='utf-8') as f_match, \
             open(candidate_out_path, 'a', encoding='utf-8') as f_cand:
             
            for start_idx in tqdm(range(0, len(s1_rows), batch_size), desc=f"Inference ({country})"):
                batch = s1_rows[start_idx:start_idx+batch_size]
                
                batch_feats = []
                batch_meta = [] # (batch_row_idx, s1_id, cand_eid, cand_idx)
                batch_cand_ids = defaultdict(list)
                
                for b_idx, (s1_id, s1_name, s1_addr) in enumerate(batch):
                    cand_indices = blocker.retrieve_candidates(s1_name, s1_addr, top_k=25)
                    
                    for c_idx in cand_indices:
                        cand_eid, cname, caddr, _, _ = blocker.target_data[c_idx]
                        batch_cand_ids[s1_id].append(cand_eid)
                        batch_feats.append(extract_features_v3(s1_name, s1_addr, cname, caddr, cand_eid))
                        batch_meta.append((b_idx, s1_id, cand_eid, c_idx))
                        
                # 100% GBDT Model Scoring
                s1_scored_matches = defaultdict(list)
                if batch_feats:
                    probs = model.predict(np.array(batch_feats, dtype=np.float32))
                    for (b_idx, s1_id, cand_eid, c_idx), prob in zip(batch_meta, probs):
                        if prob >= decision_threshold:
                            s1_id, s1_name, s1_addr = batch[b_idx]
                            _, cname, caddr, cnums, cpostals = blocker.target_data[c_idx]
                            s1_nums = extract_numbers(s1_addr)
                            s1_postals = extract_postal_codes(s1_addr)
                            
                            # Negative constraints against false merges
                            if s1_postals and cpostals and len(s1_postals.intersection(cpostals)) == 0 and fuzz.ratio(s1_name, cname) < 90:
                                continue
                            if s1_nums and cnums and len(s1_nums.intersection(cnums)) == 0 and fuzz.ratio(s1_name, cname) < 85:
                                continue
                                
                            s1_scored_matches[s1_id].append((cand_eid, prob))
                            
                # Stream write batch outputs with top-K cap
                for b_idx, (s1_id, s1_name, s1_addr) in enumerate(batch):
                    cand_list = s1_scored_matches.get(s1_id, [])
                    cand_list.sort(key=lambda x: x[1], reverse=True)
                    
                    s2_matches = []
                    s3_matches = []
                    for ceid, _ in cand_list:
                        if ceid.startswith('S2-') and len(s2_matches) < 5:
                            s2_matches.append(ceid)
                        elif ceid.startswith('S3-') and len(s3_matches) < 6:
                            s3_matches.append(ceid)
                            
                    final_matches = s2_matches + s3_matches
                    cands_str = ",".join(batch_cand_ids.get(s1_id, []))
                    matches_str = ",".join(final_matches)
                    
                    f_cand.write(f"{s1_id}\t{cands_str}\n")
                    f_match.write(f"{s1_id}\t{matches_str}\n")
                    
        print(f"Completed {country} in {time.time() - start_c_t:.2f}s!")
        
    print("\n>> All inference complete! Final files generated:")
    print(f"1. {matching_out_path}")
    print(f"2. {candidate_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='student_resource/dataset')
    parser.add_argument('--output-dir', type=str, default='output_v3')
    args = parser.parse_args()
    
    run_pipeline_v3(args.data_dir, args.output_dir)
