"""
Enhanced Multi-Index Blocker with Character N-Gram & Substring Keys.
Aims for >=95% Recall Ceiling on Entity Resolution while maintaining <30 candidates/entity.
"""

import time
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from src.preprocessing import clean_name, clean_address, extract_numbers


class HighRecallBlocker:
    def __init__(self, max_token_freq: int = 25000, min_token_len: int = 3):
        self.max_token_freq = max_token_freq
        self.min_token_len = min_token_len
        self.name_token_index = defaultdict(list)
        self.name_ngram_index = defaultdict(list)
        self.addr_token_index = defaultdict(list)
        self.exact_name_index = defaultdict(list)
        self.target_data = []  # (entity_id, c_name, c_addr, numbers, compact_name)

    def _get_char_ngrams(self, text: str, n=3) -> Set[str]:
        """Extracts character n-grams from compact alphanumeric string."""
        s = "".join([c for c in text if c.isalnum()])
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i+n] for i in range(len(s) - n + 1)}

    def fit_targets(self, entity_ids: List[str], names: List[str], addresses: List[str]):
        """Builds multi-channel inverted indices over target records (S2 and S3)."""
        print(f"Indexing {len(entity_ids):,} target records...")
        start_t = time.time()
        
        token_doc_counts = defaultdict(int)
        ngram_doc_counts = defaultdict(int)
        
        # Pass 1: compute term frequencies for IDF pruning
        for name in names:
            name_tokens = set([t for t in name.split() if len(t) >= self.min_token_len])
            for t in name_tokens:
                token_doc_counts[t] += 1
                
            ngrams = self._get_char_ngrams(name, n=3)
            for ng in ngrams:
                ngram_doc_counts[ng] += 1

        # Pass 2: build posting lists
        for idx, (eid, name, addr) in enumerate(zip(entity_ids, names, addresses)):
            nums = extract_numbers(addr)
            compact_name = "".join([c for c in name if c.isalnum()])
            self.target_data.append((eid, name, addr, nums, compact_name))
            
            # Exact clean name index
            if name:
                self.exact_name_index[name].append(idx)
                words = name.split()
                if len(words) >= 2:
                    self.exact_name_index[f"{words[0]}_{words[1]}"].append(idx)
                
            # Informative name tokens
            for t in set(name.split()):
                if len(t) >= self.min_token_len and token_doc_counts[t] <= self.max_token_freq:
                    self.name_token_index[t].append(idx)
                    
            # Rare character 3-grams for typo resilience
            ngrams = self._get_char_ngrams(name, n=3)
            for ng in ngrams:
                if ngram_doc_counts[ng] <= self.max_token_freq:
                    self.name_ngram_index[ng].append(idx)
                    
            # Address number + locality tokens
            addr_tokens = set([t for t in addr.split() if len(t) >= 4 and not t.isdigit()])
            for num in nums:
                for at in addr_tokens:
                    key = f"{num}_{at}"
                    self.addr_token_index[key].append(idx)
                    
        print(f"Indexed in {time.time() - start_t:.2f}s | Name terms: {len(self.name_token_index):,} | N-grams: {len(self.name_ngram_index):,} | Addr terms: {len(self.addr_token_index):,}")

    def query(self, s1_name: str, s1_addr: str, top_k: int = 30) -> Set[str]:
        """Queries candidates for a single S1 record with multi-channel fusion."""
        candidate_scores = defaultdict(float)
        
        # 1. Exact Name match (high weight)
        if s1_name in self.exact_name_index:
            for idx in self.exact_name_index[s1_name]:
                candidate_scores[idx] += 6.0
                
        words = s1_name.split()
        if len(words) >= 2:
            prefix2 = f"{words[0]}_{words[1]}"
            if prefix2 in self.exact_name_index:
                for idx in self.exact_name_index[prefix2]:
                    candidate_scores[idx] += 3.5
                    
        # 2. Informative Name Tokens (IDF-weighted)
        name_tokens = set([t for t in s1_name.split() if len(t) >= self.min_token_len])
        for t in name_tokens:
            if t in self.name_token_index:
                postings = self.name_token_index[t]
                w = 1.0 / (1.0 + np.log1p(len(postings)))
                for idx in postings:
                    candidate_scores[idx] += (w * 1.5)

        # 3. Rare Character 3-Grams (Typo / Domain matching)
        s1_ngrams = self._get_char_ngrams(s1_name, n=3)
        for ng in s1_ngrams:
            if ng in self.name_ngram_index:
                postings = self.name_ngram_index[ng]
                if len(postings) <= self.max_token_freq:
                    w = 0.2 / (1.0 + np.log1p(len(postings)))
                    for idx in postings:
                        candidate_scores[idx] += w

        # 4. Address Number + Locality
        nums = extract_numbers(s1_addr)
        addr_tokens = set([t for t in s1_addr.split() if len(t) >= 4 and not t.isdigit()])
        for num in nums:
            for at in addr_tokens:
                key = f"{num}_{at}"
                if key in self.addr_token_index:
                    for idx in self.addr_token_index[key]:
                        candidate_scores[idx] += 2.5

        if not candidate_scores:
            return set()
            
        top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return {self.target_data[idx][0] for idx, _ in top_candidates}


if __name__ == "__main__":
    from src.blocking import benchmark_inverted_index
    # Test HighRecallBlocker
    print("Testing HighRecallBlocker on US validation slice...")
    
    s1_all = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t')
    s2_all = pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t')
    s3_all = pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t')
    gt_all = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t')
    
    country = 'US'
    sample_size = 5000
    s1_df = s1_all[s1_all['country'] == country].head(sample_size).copy()
    s1_ids = set(s1_df['entity_id'])
    
    gt_df = gt_all[gt_all['source1_entity_id'].isin(s1_ids)].copy()
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna('')
    gt_map = {
        row['source1_entity_id']: [m.strip() for m in row['matched_entity_ids'].split(',') if m.strip()]
        for _, row in gt_df.iterrows()
    }
    
    s2_df = s2_all[s2_all['country'] == country].copy()
    s3_df = s3_all[s3_all['country'] == country].copy()
    
    s1_df['c_name'] = s1_df['business_name'].apply(clean_name)
    s1_df['c_addr'] = s1_df['business_address'].apply(clean_address)
    
    s2_df['c_name'] = s2_df['business_name'].apply(clean_name)
    s2_df['c_addr'] = s2_df['business_address'].apply(clean_address)
    
    s3_df['c_name'] = s3_df['business_name'].apply(clean_name)
    s3_df['c_addr'] = s3_df['business_address'].apply(clean_address)
    
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    blocker = HighRecallBlocker(max_token_freq=25000, min_token_len=3)
    blocker.fit_targets(
        entity_ids=targets_df['entity_id'].tolist(),
        names=targets_df['c_name'].tolist(),
        addresses=targets_df['c_addr'].tolist()
    )
    
    start_q = time.time()
    candidates_dict = {}
    for _, row in s1_df.iterrows():
        s1_id = row['entity_id']
        cand_set = blocker.query(row['c_name'], row['c_addr'], top_k=30)
        candidates_dict[s1_id] = cand_set
        
    query_time = time.time() - start_q
    print(f"Queried {len(s1_df):,} S1 in {query_time:.2f}s ({len(s1_df)/query_time:.1f} queries/sec)")
    
    total_true_matches = sum(len(v) for v in gt_map.values())
    captured_matches = 0
    total_candidates = sum(len(c) for c in candidates_dict.values())
    avg_cands = total_candidates / len(s1_df)
    
    for s1_id, true_list in gt_map.items():
        cand_set = candidates_dict.get(s1_id, set())
        for m in true_list:
            if m in cand_set:
                captured_matches += 1
                
    recall_ceiling = (captured_matches / total_true_matches) * 100 if total_true_matches > 0 else 0
    print("\n------------------------------------------")
    print(f">> HIGH RECALL BLOCKING RESULTS ({country})")
    print(f"Total True Matches: {total_true_matches:,}")
    print(f"Captured Matches: {captured_matches:,}")
    print(f"Recall Ceiling: {recall_ceiling:.2f}%")
    print(f"Avg Candidates / S1: {avg_cands:.2f}")
    print(f"Reduction Ratio: {(1.0 - (total_candidates / (len(s1_df) * len(targets_df)))) * 100:.6f}%")
    print("------------------------------------------\n")
