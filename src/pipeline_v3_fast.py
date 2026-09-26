"""
Amazon ML Challenge 2026 - Memory-Safe Ultra-Fast Pipeline V3
Optimized Architecture:
- Memory footprint: ~4-6 GB (100% safe on 32 GB RAM, 0 swap)
- High-Recall Low-Latency Blocker with Frequency Caps:
  - Exact Name, Prefix-2, Compact Name (top 10 postings)
  - IDF Name Tokens (frequency <= 3,000)
  - Number + Locality combo (frequency <= 1,500, capped at top 15)
- Batch-vectorized LightGBM C++ OpenMP multi-threaded inference (num_threads=4)
- Full 28-D GBDT scoring with calibrated threshold (θ = 0.68)
- Country-specific or Full execution support
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
from rapidfuzz import fuzz

from src.preprocessing import clean_name, clean_address, extract_numbers


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


class FastOptimizedBlocker:
    def __init__(self, max_token_freq: int = 3000, max_addr_freq: int = 1500):
        self.max_token_freq = max_token_freq
        self.max_addr_freq = max_addr_freq
        self.exact_name_map = defaultdict(list)
        self.prefix2_map = defaultdict(list)
        self.compact_name_map = defaultdict(list)
        self.name_token_index = defaultdict(list)
        self.addr_combo_index = defaultdict(list)
        self.target_data = []

    def fit(self, entity_ids: List[str], names: List[str], addresses: List[str]):
        token_doc_counts = defaultdict(int)
        
        for name in names:
            for t in set(name.split()):
                if len(t) >= 3:
                    token_doc_counts[t] += 1

        for idx, (eid, name, addr) in enumerate(zip(entity_ids, names, addresses)):
            tokens = name.split()
            nums = frozenset(extract_numbers(addr))
            postals = frozenset(extract_postal_codes(addr))
            comp = clean_compact_name(name)
            ngrams = frozenset(get_char_ngrams(name, n=3))
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

    def retrieve_candidates(self, s1_name: str, s1_addr: str, s1_tokens: List[str], s1_comp: str, top_k: int = 25) -> List[int]:
        scores = defaultdict(float)
        
        # 1. Exact Name & Prefix-2
        if s1_name in self.exact_name_map:
            for idx in self.exact_name_map[s1_name][:10]:
                scores[idx] += 5.0
                
        if len(s1_tokens) >= 2:
            p2 = f"{s1_tokens[0]}_{s1_tokens[1]}"
            if p2 in self.prefix2_map:
                for idx in self.prefix2_map[p2][:10]:
                    scores[idx] += 3.5
                    
        # 2. Compact Domain Name
        if s1_comp and len(s1_comp) >= 5 and s1_comp in self.compact_name_map:
            for idx in self.compact_name_map[s1_comp][:10]:
                scores[idx] += 4.0

        # 3. Informative Name Tokens with IDF
        for t in set(s1_tokens):
            if len(t) >= 3 and t in self.name_token_index:
                postings = self.name_token_index[t]
                w = 1.0 / (1.0 + np.log1p(len(postings)))
                for idx in postings[:20]:
                    scores[idx] += (w * 1.5)
                    
        # 4. Number + Locality combo (with frequency cap for high-density countries like India)
        nums = extract_numbers(s1_addr)
        addr_tokens = set([t for t in s1_addr.split() if len(t) >= 4 and not t.isdigit()])
        for num in nums:
            for at in addr_tokens:
                key = f"{num}_{at}"
                if key in self.addr_combo_index:
                    postings = self.addr_combo_index[key]
                    if len(postings) <= self.max_addr_freq:
                        for idx in postings[:15]:
                            scores[idx] += 2.0
                        
        if not scores:
            return []
            
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [idx for idx, _ in top_items]


def fast_extract_features(s1_info, cand_info):
    """Ultra-fast feature extraction using pre-extracted metadata."""
    s1_id, s1_name, s1_addr, s1_comp, s1_tokens, s1_ngrams, s1_nums, s1_postals, len1_n, len1_a = s1_info
    cand_eid, cand_name, cand_addr, cand_comp, cand_tokens, cand_ngrams, cand_nums, cand_postals, is_s3, len2_n, len2_a = cand_info
    
    # 1. Fast Name Ratios
    nr = fuzz.ratio(s1_name, cand_name) / 100.0
    ntset = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    
    # Fast reject obvious non-matches
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
    
    # 3-gram Jaccard
    if s1_ngrams and cand_ngrams:
        common_ng = len(s1_ngrams.intersection(cand_ngrams))
        ncjacc = common_ng / (len(s1_ngrams.union(cand_ngrams)) + 1e-5)
    else:
        ncjacc = 0.0
        
    ncompact = fuzz.ratio(s1_comp, cand_comp) / 100.0 if (s1_comp and cand_comp) else 0.0
    
    # 2. Address Features
    addr_missing = 1.0 if (not s1_addr or not cand_addr) else 0.0
    if addr_missing:
        ar = atset = ats = apr = afirst = num_c = num_j = num_p = pin_match = pin_both = 0.0
    else:
        ar = fuzz.ratio(s1_addr, cand_addr) / 100.0
        atset = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        ats = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        apr = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
        afirst = 1.0 if (s1_addr[:6] == cand_addr[:6]) else 0.0
        
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


def run_pipeline_v3_fast(data_dir: str, output_dir: str, target_countries: List[str] = None):
    os.makedirs(output_dir, exist_ok=True)
    
    print("==========================================================================")
    print(">> AMAZON ML CHALLENGE 2026: HIGH-THROUGHPUT PIPELINE V3 (OPTIMIZED)")
    print("==========================================================================")
    
    model_path = 'models/lgb_matcher_v3.txt'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        sys.exit(1)
        
    model = lgb.Booster(model_file=model_path)
    threshold = 0.68
    print(f">> Loaded pre-trained model: {model_path} (GBDT threshold θ = {threshold})")
    
    final_matching_path = os.path.join(output_dir, 'matching_results.tsv')
    final_cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    
    if target_countries is None or len(target_countries) == 3:
        countries = ['US', 'France', 'India']
        with open(final_matching_path, 'w', encoding='utf-8') as f_m, \
             open(final_cand_path, 'w', encoding='utf-8') as f_c:
            f_m.write("source1_entity_id\tmatched_entity_ids\n")
            f_c.write("source1_entity_id\tcandidate_entity_ids\n")
    else:
        countries = target_countries
        print(f">> Targeted execution for: {countries} (appending to existing files)")
        
    print("\n[Step 1/2] Loading Test Set Sources...")
    test_s1 = pd.read_csv(os.path.join(data_dir, 'test/test_source1.tsv'), sep='\t')
    test_s2 = pd.read_csv(os.path.join(data_dir, 'test/test_source2.tsv'), sep='\t')
    test_s3 = pd.read_csv(os.path.join(data_dir, 'test/test_source3.tsv'), sep='\t')
    
    print("Normalizing strings...")
    test_s1['c_name'] = test_s1['business_name'].apply(clean_name)
    test_s1['c_addr'] = test_s1['business_address'].apply(clean_address)
    
    test_targets = pd.concat([test_s2, test_s3], ignore_index=True)
    test_targets['c_name'] = test_targets['business_name'].apply(clean_name)
    test_targets['c_addr'] = test_targets['business_address'].apply(clean_address)
    
    del test_s2, test_s3
    gc.collect()
    
    total_pipeline_start = time.time()
    batch_size = 10000
    
    for country in countries:
        c_start = time.time()
        c_s1 = test_s1[test_s1['country'] == country].copy()
        c_targets = test_targets[test_targets['country'] == country].copy()
        
        print(f"\n==========================================================================")
        print(f">> Pre-computing & Indexing {country} Targets ({len(c_targets):,} records)...")
        print(f"==========================================================================")
        
        blocker = FastOptimizedBlocker()
        blocker.fit(
            entity_ids=c_targets['entity_id'].tolist(),
            names=c_targets['c_name'].tolist(),
            addresses=c_targets['c_addr'].tolist()
        )
        
        del c_targets
        gc.collect()
        
        # Pre-compute S1 info tuples
        print(f"Pre-extracting metadata for {len(c_s1):,} S1 entities in {country}...")
        s1_tuples = []
        for eid, name, addr in zip(c_s1['entity_id'], c_s1['c_name'], c_s1['c_addr']):
            comp = clean_compact_name(name)
            tokens = name.split()
            ngrams = frozenset(get_char_ngrams(name, n=3))
            nums = frozenset(extract_numbers(addr))
            postals = frozenset(extract_postal_codes(addr))
            s1_tuples.append((eid, name, addr, comp, tokens, ngrams, nums, postals, float(len(name)), float(len(addr))))
            
        del c_s1
        gc.collect()
        
        total_s1 = len(s1_tuples)
        print(f"⚡ Streaming inference for {country} ({total_s1:,} queries, batch size = {batch_size:,})...")
        
        with open(final_matching_path, 'a', encoding='utf-8') as f_m, \
             open(final_cand_path, 'a', encoding='utf-8') as f_c:
             
            for start_idx in range(0, total_s1, batch_size):
                b_t0 = time.time()
                batch = s1_tuples[start_idx:start_idx+batch_size]
                batch_feats = []
                batch_meta = [] # (b_idx, s1_id, cand_eid, cand_idx)
                batch_cand_ids = defaultdict(list)
                
                for b_idx, s1_info in enumerate(batch):
                    s1_id = s1_info[0]
                    s1_name = s1_info[1]
                    s1_addr = s1_info[2]
                    s1_comp = s1_info[3]
                    s1_tokens = s1_info[4]
                    
                    cand_indices = blocker.retrieve_candidates(s1_name, s1_addr, s1_tokens, s1_comp, top_k=25)
                    
                    for c_idx in cand_indices:
                        cand_info = blocker.target_data[c_idx]
                        cand_eid = cand_info[0]
                        batch_cand_ids[s1_id].append(cand_eid)
                        
                        feat_vec = fast_extract_features(s1_info, cand_info)
                        if feat_vec is not None:
                            batch_feats.append(feat_vec)
                            batch_meta.append((b_idx, s1_id, cand_eid, c_idx))
                            
                # Native multi-threaded C++ OpenMP LightGBM prediction
                s1_scored_matches = defaultdict(list)
                if batch_feats:
                    X_mat = np.array(batch_feats, dtype=np.float32)
                    probs = model.predict(X_mat, num_threads=4)
                    
                    for (b_idx, s1_id, cand_eid, c_idx), prob in zip(batch_meta, probs):
                        if prob >= threshold:
                            s1_info = batch[b_idx]
                            cand_info = blocker.target_data[c_idx]
                            
                            s1_postals = s1_info[7]
                            cand_postals = cand_info[7]
                            s1_nums = s1_info[6]
                            cand_nums = cand_info[6]
                            
                            # Negative constraints
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
                        if ceid.startswith('S2-') and len(s2_matches) < 5:
                            s2_matches.append(ceid)
                        elif ceid.startswith('S3-') and len(s3_matches) < 6:
                            s3_matches.append(ceid)
                            
                    final_matches = s2_matches + s3_matches
                    cands_str = ",".join(batch_cand_ids.get(s1_id, []))
                    matches_str = ",".join(final_matches)
                    
                    f_c.write(f"{s1_id}\t{cands_str}\n")
                    f_m.write(f"{s1_id}\t{matches_str}\n")
                    
                processed = min(start_idx + batch_size, total_s1)
                elapsed = time.time() - c_start
                rate = processed / max(elapsed, 0.001)
                b_time = time.time() - b_t0
                if (start_idx // batch_size) % 2 == 0 or processed == total_s1:
                    print(f"  [{country}] Processed {processed:,}/{total_s1:,} ({processed/total_s1*100:.1f}%) | Speed: {rate:.1f} ent/s (batch {b_time:.2f}s)", flush=True)
                    
        del blocker, s1_tuples
        gc.collect()
        
        c_elapsed = time.time() - c_start
        c_rate = total_s1 / c_elapsed
        print(f"✓ Finished {country} ({total_s1:,} records) in {c_elapsed:.1f}s ({c_rate:.1f} ent/s)!")
        
    total_elapsed = time.time() - total_pipeline_start
    print("\n==========================================================================")
    print(f"🎉 INFERENCE COMPLETE in {total_elapsed/60:.2f} minutes!")
    print(f"1. {final_matching_path}")
    print(f"2. {final_cand_path}")
    print("==========================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='student_resource/dataset')
    parser.add_argument('--output-dir', type=str, default='output_v3')
    parser.add_argument('--country', type=str, default=None)
    args = parser.parse_args()
    
    countries = [args.country] if args.country else None
    run_pipeline_v3_fast(args.data_dir, args.output_dir, target_countries=countries)
