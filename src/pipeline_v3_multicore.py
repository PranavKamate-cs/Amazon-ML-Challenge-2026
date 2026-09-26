"""
Amazon ML Challenge 2026 - Pipeline V3 Multi-Core (4x Parallel High-Precision GBDT Matcher)
Features:
- Multi-Core Sharding across all 4 vCPUs on EC2 (400% CPU utilization)
- Uses pre-trained 28-D LightGBM booster model (models/lgb_matcher_v3.txt)
- 100% GBDT Model Scoring for all candidate pairs (Zero unverified fast-paths)
- High-Precision Calibrated Threshold (θ = 0.68) for Macro F_0.5 Optimization
- Memory-safe per-shard streaming with automatic zero-overhead TSV concatenation
"""

import time
import os
import sys
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from multiprocessing import Pool, cpu_count

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
    if not address:
        return set()
    pins = re.findall(r'\b\d{5,6}\b', address)
    return set(pins)


def clean_compact_name(name: str) -> str:
    if not name:
        return ""
    return "".join([c for c in name.lower() if c.isalnum()])


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    if not text or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def extract_features_v3(
    s1_name: str, s1_addr: str,
    cand_name: str, cand_addr: str,
    cand_id: str
) -> np.ndarray:
    len1_n, len2_n = len(s1_name), len(cand_name)
    len1_a, len2_a = len(s1_addr), len(cand_addr)
    
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


class HighRecallBlocker:
    def __init__(self, max_token_freq: int = 15000):
        self.max_token_freq = max_token_freq
        self.exact_name_map = defaultdict(list)
        self.prefix2_map = defaultdict(list)
        self.compact_name_map = defaultdict(list)
        self.name_token_index = defaultdict(list)
        self.addr_combo_index = defaultdict(list)
        self.name_ngram_index = defaultdict(list)
        self.target_data = []

    def fit(self, entity_ids: List[str], names: List[str], addresses: List[str]):
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
                    
            addr_tokens = set([t for t in addr.split() if len(t) >= 4 and not t.isdigit()])
            for num in nums:
                for at in addr_tokens:
                    key = f"{num}_{at}"
                    self.addr_combo_index[key].append(idx)
                    
            for ng in get_char_ngrams(name, n=3):
                if ngram_doc_counts[ng] <= self.max_token_freq:
                    self.name_ngram_index[ng].append(idx)

    def retrieve_candidates(self, s1_name: str, s1_addr: str, top_k: int = 25) -> List[int]:
        scores = defaultdict(float)
        
        if s1_name in self.exact_name_map:
            for idx in self.exact_name_map[s1_name][:10]:
                scores[idx] += 5.0
                
        words = s1_name.split()
        if len(words) >= 2:
            p2 = f"{words[0]}_{words[1]}"
            if p2 in self.prefix2_map:
                for idx in self.prefix2_map[p2][:10]:
                    scores[idx] += 3.5
                    
        comp = clean_compact_name(s1_name)
        if comp and len(comp) >= 5 and comp in self.compact_name_map:
            for idx in self.compact_name_map[comp][:10]:
                scores[idx] += 4.0

        for t in set(words):
            if len(t) >= 3 and t in self.name_token_index:
                postings = self.name_token_index[t]
                w = 1.0 / (1.0 + np.log1p(len(postings)))
                for idx in postings:
                    scores[idx] += (w * 1.5)
                    
        nums = extract_numbers(s1_addr)
        addr_tokens = set([t for t in s1_addr.split() if len(t) >= 4 and not t.isdigit()])
        for num in nums:
            for at in addr_tokens:
                key = f"{num}_{at}"
                if key in self.addr_combo_index:
                    for idx in self.addr_combo_index[key]:
                        scores[idx] += 2.0
                        
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


# Global worker context
_GLOBAL_BLOCKER = None
_GLOBAL_MODEL = None
_GLOBAL_THRESHOLD = 0.68

def _init_worker(blocker, model_path, threshold):
    global _GLOBAL_BLOCKER, _GLOBAL_MODEL, _GLOBAL_THRESHOLD
    _GLOBAL_BLOCKER = blocker
    _GLOBAL_MODEL = lgb.Booster(model_file=model_path)
    _GLOBAL_THRESHOLD = threshold

def _process_shard(args):
    shard_id, s1_rows, match_out_file, cand_out_file = args
    global _GLOBAL_BLOCKER, _GLOBAL_MODEL, _GLOBAL_THRESHOLD
    
    batch_size = 5000
    with open(match_out_file, 'w', encoding='utf-8') as f_match, \
         open(cand_out_file, 'w', encoding='utf-8') as f_cand:
         
        for start_idx in range(0, len(s1_rows), batch_size):
            batch = s1_rows[start_idx:start_idx+batch_size]
            batch_feats = []
            batch_meta = []
            batch_cand_ids = defaultdict(list)
            
            for b_idx, (s1_id, s1_name, s1_addr) in enumerate(batch):
                cand_indices = _GLOBAL_BLOCKER.retrieve_candidates(s1_name, s1_addr, top_k=25)
                for c_idx in cand_indices:
                    cand_eid, cname, caddr, _, _ = _GLOBAL_BLOCKER.target_data[c_idx]
                    batch_cand_ids[s1_id].append(cand_eid)
                    batch_feats.append(extract_features_v3(s1_name, s1_addr, cname, caddr, cand_eid))
                    batch_meta.append((b_idx, s1_id, cand_eid, c_idx))
                    
            s1_scored_matches = defaultdict(list)
            if batch_feats:
                probs = _GLOBAL_MODEL.predict(np.array(batch_feats, dtype=np.float32))
                for (b_idx, s1_id, cand_eid, c_idx), prob in zip(batch_meta, probs):
                    if prob >= _GLOBAL_THRESHOLD:
                        s1_id, s1_name, s1_addr = batch[b_idx]
                        _, cname, caddr, cnums, cpostals = _GLOBAL_BLOCKER.target_data[c_idx]
                        s1_nums = extract_numbers(s1_addr)
                        s1_postals = extract_postal_codes(s1_addr)
                        
                        if s1_postals and cpostals and len(s1_postals.intersection(cpostals)) == 0 and fuzz.ratio(s1_name, cname) < 90:
                            continue
                        if s1_nums and cnums and len(s1_nums.intersection(cnums)) == 0 and fuzz.ratio(s1_name, cname) < 85:
                            continue
                            
                        s1_scored_matches[s1_id].append((cand_eid, prob))
                        
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
                
    return shard_id


def run_pipeline_v3_multicore(data_dir: str, output_dir: str, num_workers: int = 4):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    temp_dir = os.path.join(output_dir, 'tmp_shards')
    os.makedirs(temp_dir, exist_ok=True)
    
    print("=================================================================")
    print(">> AMAZON ML CHALLENGE 2026: 4x MULTI-CORE PIPELINE V3 (400% CPU)")
    print("=================================================================")
    
    model_path = 'models/lgb_matcher_v3.txt'
    if not os.path.exists(model_path):
        print("ERROR: Model models/lgb_matcher_v3.txt not found!")
        sys.exit(1)
        
    print(f">> Loaded pre-trained model from {model_path}!")
    
    # Load test data
    print("\nLoading Test Datasets...")
    test_s1 = pd.read_csv(os.path.join(data_dir, 'test/test_source1.tsv'), sep='\t')
    test_s2 = pd.read_csv(os.path.join(data_dir, 'test/test_source2.tsv'), sep='\t')
    test_s3 = pd.read_csv(os.path.join(data_dir, 'test/test_source3.tsv'), sep='\t')
    
    print("Normalizing strings...")
    test_s1['c_name'] = test_s1['business_name'].apply(clean_name)
    test_s1['c_addr'] = test_s1['business_address'].apply(clean_address)
    
    test_targets = pd.concat([test_s2, test_s3], ignore_index=True)
    test_targets['c_name'] = test_targets['business_name'].apply(clean_name)
    test_targets['c_addr'] = test_targets['business_address'].apply(clean_address)
    
    final_matching_path = os.path.join(output_dir, 'matching_results.tsv')
    final_cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    
    with open(final_matching_path, 'w', encoding='utf-8') as f_m, \
         open(final_cand_path, 'w', encoding='utf-8') as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        
    countries = ['US', 'France', 'India']
    threshold = 0.68
    
    for country in countries:
        c_start = time.time()
        c_s1 = test_s1[test_s1['country'] == country].copy()
        c_targets = test_targets[test_targets['country'] == country].copy()
        
        print(f"\n==========================================")
        print(f">> Indexing Country: {country} (S1: {len(c_s1):,}, Targets: {len(c_targets):,})")
        print(f"==========================================")
        
        blocker = HighRecallBlocker()
        blocker.fit(
            entity_ids=c_targets['entity_id'].tolist(),
            names=c_targets['c_name'].tolist(),
            addresses=c_targets['c_addr'].tolist()
        )
        
        s1_rows = list(zip(c_s1['entity_id'], c_s1['c_name'], c_s1['c_addr']))
        chunk_size = int(np.ceil(len(s1_rows) / num_workers))
        
        tasks = []
        shard_files = []
        for i in range(num_workers):
            sub_rows = s1_rows[i*chunk_size:(i+1)*chunk_size]
            if not sub_rows:
                continue
            m_f = os.path.join(temp_dir, f"m_{country}_{i}.tsv")
            c_f = os.path.join(temp_dir, f"c_{country}_{i}.tsv")
            tasks.append((i, sub_rows, m_f, c_f))
            shard_files.append((m_f, c_f))
            
        print(f"Spawning {len(tasks)} parallel workers for {country} (400% CPU)...")
        with Pool(processes=len(tasks), initializer=_init_worker, initargs=(blocker, model_path, threshold)) as pool:
            for _ in pool.imap_unordered(_process_shard, tasks):
                pass
                
        # Concatenate shards to main file
        print(f"Merging {country} worker outputs...")
        with open(final_matching_path, 'a', encoding='utf-8') as f_m, \
             open(final_cand_path, 'a', encoding='utf-8') as f_c:
            for m_f, c_f in shard_files:
                if os.path.exists(m_f):
                    with open(m_f, 'r', encoding='utf-8') as in_m:
                        f_m.write(in_m.read())
                    os.remove(m_f)
                if os.path.exists(c_f):
                    with open(c_f, 'r', encoding='utf-8') as in_c:
                        f_c.write(in_c.read())
                    os.remove(c_f)
                    
        print(f"✓ Completed {country} in {time.time() - c_start:.2f}s!")
        
    print("\n=======================================================")
    print(">> 4-CORE INFERENCE COMPLETE! Final Files Ready:")
    print(f"1. {final_matching_path}")
    print(f"2. {final_cand_path}")
    print("=======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='student_resource/dataset')
    parser.add_argument('--output-dir', type=str, default='output_v3')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    run_pipeline_v3_multicore(args.data_dir, args.output_dir, args.workers)
