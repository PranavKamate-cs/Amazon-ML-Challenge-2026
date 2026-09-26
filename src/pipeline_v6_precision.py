"""
Amazon ML Challenge 2026 - Pipeline V6 High-Precision Engine
Calibrated to Ground Truth Mean (~3.2 matches/query, total ~5.2M matches)
Uses verified high-precision GBDT model with tight false-positive suppression.
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


class MultiIndexBlockerV6:
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


def run_pipeline_v6_precision(cache_dir: str = 'data_preprocessed', model_path: str = 'models/lgb_matcher_v3.txt', output_dir: str = 'output_v6', threshold: float = 0.70):
    os.makedirs(output_dir, exist_ok=True)
    
    print("==========================================================================")
    print(">> AMAZON ML CHALLENGE 2026: PIPELINE V6 HIGH-PRECISION ENGINE")
    print("==========================================================================")
    
    print(f">> Loading Verified High-Precision GBDT Model: {model_path}")
    model = lgb.Booster(model_file=model_path)
    
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
    total_matches_count = 0
    
    for country in countries:
        c_start = time.time()
        print(f"\n==========================================================================")
        print(f">> [Step 1/2] Loading Partitioned Cache & Building Index for {country}...")
        print(f"==========================================================================")
        
        t0 = time.time()
        c_s1 = pd.read_parquet(os.path.join(test_dir, 'source1.parquet'), filters=[('country', '==', country)])
        c_s2 = pd.read_parquet(os.path.join(test_dir, 'source2.parquet'), filters=[('country', '==', country)])
        c_s3 = pd.read_parquet(os.path.join(test_dir, 'source3.parquet'), filters=[('country', '==', country)])
        c_targets = pd.concat([c_s2, c_s3], ignore_index=True)
        del c_s2, c_s3
        gc.collect()
        print(f"✓ Loaded {len(c_s1):,} S1 queries and {len(c_targets):,} Target records for {country} in {time.time()-t0:.2f}s")
        
        blocker = MultiIndexBlockerV6()
        blocker.fit_targets(
            entity_ids=c_targets['entity_id'].tolist(),
            names=c_targets['c_name'].tolist(),
            addresses=c_targets['c_addr'].tolist()
        )
        del c_targets
        gc.collect()
        
        print(f"Pre-extracting metadata for {len(c_s1):,} S1 entities in {country}...")
        s1_tuples = []
        for eid, name, addr in zip(c_s1['entity_id'], c_s1['c_name'], c_s1['c_addr']):
            name_str = str(name) if pd.notna(name) else ""
            addr_str = str(addr) if pd.notna(addr) else ""
            comp_str = clean_compact_name(name_str)
            tokens = name_str.split()
            ngrams = frozenset(get_char_ngrams(name_str, n=3))
            nums = frozenset(extract_numbers(addr_str))
            pins = frozenset(extract_postal_codes(addr_str))
            s1_tuples.append((
                str(eid), name_str, addr_str, comp_str, tokens, ngrams, nums, pins,
                float(len(name_str)), float(len(addr_str))
            ))
            
        del c_s1
        gc.collect()
        
        total_s1 = len(s1_tuples)
        print(f"⚡ Streaming High-Precision Inference for {country} ({total_s1:,} queries | θ = {threshold})...")
        
        with open(final_matching_path, 'a', encoding='utf-8') as f_m, \
             open(final_cand_path, 'a', encoding='utf-8') as f_c:
             
            for start_idx in range(0, total_s1, batch_size):
                b_t0 = time.time()
                batch = s1_tuples[start_idx:start_idx+batch_size]
                batch_feats = []
                batch_meta = []
                batch_cand_ids = defaultdict(list)
                
                for b_idx, s1_info in enumerate(batch):
                    s1_id, s1_name, s1_addr, s1_comp, s1_tokens, s1_ngrams, s1_nums, s1_postals, len1_n, len1_a = s1_info
                    cand_indices = blocker.query(s1_name, s1_addr, s1_comp, s1_tokens, top_k=20)
                    
                    for c_idx in cand_indices:
                        cand_info = blocker.target_data[c_idx]
                        cand_eid = cand_info[0]
                        batch_cand_ids[s1_id].append(cand_eid)
                        
                        feat_vec = fast_extract_features(s1_info, cand_info)
                        if feat_vec is not None:
                            batch_feats.append(feat_vec)
                            batch_meta.append((b_idx, s1_id, cand_eid, c_idx))
                            
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
                            
                            # Strict negative constraints
                            if s1_postals and cand_postals and len(s1_postals.intersection(cand_postals)) == 0 and fuzz.ratio(s1_info[1], cand_info[1]) < 90:
                                continue
                            if s1_nums and cand_nums and len(s1_nums.intersection(cand_nums)) == 0 and fuzz.ratio(s1_info[1], cand_info[1]) < 85:
                                continue
                                
                            s1_scored_matches[s1_id].append((cand_eid, prob))
                            
                # Precision filter: max 3 S2 matches, max 3 S3 matches per query
                for b_idx, s1_info in enumerate(batch):
                    s1_id = s1_info[0]
                    cand_list = s1_scored_matches.get(s1_id, [])
                    cand_list.sort(key=lambda x: x[1], reverse=True)
                    
                    s2_matches = []
                    s3_matches = []
                    for ceid, _ in cand_list:
                        if ceid.startswith('S2-') and len(s2_matches) < 3:
                            s2_matches.append(ceid)
                        elif ceid.startswith('S3-') and len(s3_matches) < 3:
                            s3_matches.append(ceid)
                            
                    final_matches = s2_matches + s3_matches
                    cands_str = ",".join(batch_cand_ids.get(s1_id, []))
                    matches_str = ",".join(final_matches)
                    
                    if not matches_str:
                        total_singletons += 1
                    else:
                        total_matches_count += len(final_matches)
                        
                    f_c.write(f"{s1_id}\t{cands_str}\n")
                    f_m.write(f"{s1_id}\t{matches_str}\n")
                    
                processed = min(start_idx + batch_size, total_s1)
                elapsed = time.time() - c_start
                rate = processed / max(elapsed, 0.001)
                b_time = time.time() - b_t0
                if (start_idx // batch_size) % 2 == 0 or processed == total_s1:
                    print(f"  [{country}] Processed {processed:,}/{total_s1:,} ({processed/total_s1*100:.1f}%) | Singletons: {total_singletons:,} | Matches: {total_matches_count:,} | Speed: {rate:.1f} ent/s (batch {b_time:.2f}s)", flush=True)
                    
        del blocker, s1_tuples
        gc.collect()
        
        c_elapsed = time.time() - c_start
        c_rate = total_s1 / c_elapsed
        print(f"✓ Finished {country} ({total_s1:,} records) in {c_elapsed:.1f}s ({c_rate:.1f} ent/s)!")
        
    total_elapsed = time.time() - total_start
    print("\n==========================================================================")
    print(f"🎉 PIPELINE V6 PRECISION INFERENCE COMPLETE in {total_elapsed/60:.2f} minutes!")
    print(f"Total Singletons: {total_singletons:,} ({(total_singletons/1732544)*100:.2f}%)")
    print(f"Total Matches Predicted: {total_matches_count:,} (Mean: {total_matches_count/1732544:.2f} matches/query)")
    print(f"1. {final_matching_path}")
    print(f"2. {final_cand_path}")
    print("==========================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, default='data_preprocessed')
    parser.add_argument('--model-path', type=str, default='models/lgb_matcher_v3.txt')
    parser.add_argument('--output-dir', type=str, default='output_v6')
    parser.add_argument('--threshold', type=float, default=0.70)
    args = parser.parse_args()
    
    run_pipeline_v6_precision(args.cache_dir, args.model_path, args.output_dir, args.threshold)
