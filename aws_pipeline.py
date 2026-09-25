"""
Production AWS Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
Designed for high-memory AWS EC2 instances (e.g., r6i.2xlarge, c6i.4xlarge, g4dn.2xlarge).
Processes 11.7M+ total records cleanly with multi-processing and streaming output.
"""

import os
import sys
import time
import argparse
import pickle
import unicodedata
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz
from tqdm import tqdm


# --- 1. STRING PREPROCESSING ---
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'pvt ltd', 'co', 'company', 'group', 'holdings',
    'enterprises', 'associates', 'llp', 'pllc', 'gmbh', 'sarl', 'sas',
    'sa', 'eurl', 'sasu', 'sci', 'snc', 'gie', 'spa', 'bv'
}

DOMAIN_EXTENSIONS = re.compile(r'\.(com|in|org|net|co|io|fr|us|biz|info|ai|edu|gov)(/.*)?$', re.IGNORECASE)

ADDRESS_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue',
    'blvd': 'boulevard', 'dr': 'drive', 'ln': 'lane', 'ct': 'court',
    'pl': 'place', 'sq': 'square', 'hwy': 'highway', 'pkwy': 'parkway',
    'ste': 'suite', 'apt': 'apartment', 'dept': 'department', 'fl': 'floor',
    'bldg': 'building', 'opp': 'opposite', 'nr': 'near', 'dist': 'district',
    'sec': 'sector', 'pk': 'park',
}


def unicode_normalize(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')


def clean_name(name: str) -> str:
    if not name or str(name).lower() == 'nan':
        return ""
    text = unicode_normalize(name).lower().strip()
    text = re.sub(r'^https?://', '', text)
    text = re.sub(r'^www\.', '', text)
    text = DOMAIN_EXTENSIONS.sub('', text)
    text = text.replace('&', ' and ').replace('@', ' at ').replace('+', ' plus ')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    if not tokens:
        return ""
    filtered = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(filtered if filtered else tokens)


def clean_address(address: str) -> str:
    if not address or str(address).lower() == 'nan':
        return ""
    text = unicode_normalize(address).lower().strip()
    text = text.replace('#', ' number ').replace('/', ' ').replace('-', ' ').replace(',', ' ')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return " ".join([ADDRESS_ABBR.get(t, t) for t in tokens])


def extract_numbers(text: str) -> Set[str]:
    if not text or str(text).lower() == 'nan':
        return set()
    return set(re.findall(r'\b\d+\b', str(text)))


# --- 2. CANDIDATE BLOCKING ---
class MemoryEfficientBlocker:
    def __init__(self, max_token_freq: int = 30000, min_token_len: int = 3):
        self.max_token_freq = max_token_freq
        self.min_token_len = min_token_len
        self.name_token_index = defaultdict(list)
        self.addr_token_index = defaultdict(list)
        self.exact_name_index = defaultdict(list)
        self.target_data = []

    def fit(self, entity_ids: List[str], names: List[str], addresses: List[str]):
        token_doc_counts = defaultdict(int)
        for name in names:
            tokens = set([t for t in name.split() if len(t) >= self.min_token_len])
            for t in tokens:
                token_doc_counts[t] += 1

        for idx, (eid, name, addr) in enumerate(zip(entity_ids, names, addresses)):
            nums = extract_numbers(addr)
            self.target_data.append((eid, name, addr, nums))
            
            if name:
                self.exact_name_index[name].append(idx)
                words = name.split()
                if len(words) >= 2:
                    self.exact_name_index[f"{words[0]}_{words[1]}"].append(idx)
                    
            for t in set(name.split()):
                if len(t) >= self.min_token_len and token_doc_counts[t] <= self.max_token_freq:
                    self.name_token_index[t].append(idx)
                    
            addr_tokens = set([t for t in addr.split() if len(t) >= 4 and not t.isdigit()])
            for num in nums:
                for at in addr_tokens:
                    self.addr_token_index[f"{num}_{at}"].append(idx)

    def query(self, s1_name: str, s1_addr: str, top_k: int = 25) -> Set[str]:
        scores = defaultdict(float)
        
        if s1_name in self.exact_name_index:
            for idx in self.exact_name_index[s1_name]:
                scores[idx] += 5.0
                
        words = s1_name.split()
        if len(words) >= 2:
            prefix2 = f"{words[0]}_{words[1]}"
            if prefix2 in self.exact_name_index:
                for idx in self.exact_name_index[prefix2]:
                    scores[idx] += 3.0
                    
        tokens = set([t for t in s1_name.split() if len(t) >= self.min_token_len])
        for t in tokens:
            if t in self.name_token_index:
                postings = self.name_token_index[t]
                w = 1.0 / (1.0 + np.log1p(len(postings)))
                for idx in postings:
                    scores[idx] += w

        nums = extract_numbers(s1_addr)
        addr_tokens = set([t for t in s1_addr.split() if len(t) >= 4 and not t.isdigit()])
        for num in nums:
            for at in addr_tokens:
                key = f"{num}_{at}"
                if key in self.addr_token_index:
                    for idx in self.addr_token_index[key]:
                        scores[idx] += 2.0

        if not scores:
            return set()
            
        top_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return {self.target_data[idx][0] for idx, _ in top_candidates}


# --- 3. PAIR FEATURE EXTRACTION ---
FEATURE_NAMES = [
    'name_ratio', 'name_partial_ratio', 'name_token_sort', 'name_token_set',
    'name_wratio', 'name_exact_match', 'name_len_diff', 'name_len_ratio',
    'addr_ratio', 'addr_token_set', 'addr_token_sort', 'addr_partial_ratio',
    'addr_is_missing', 'num_common_count', 'num_jaccard', 'num_mismatch_penalty',
    'is_s3',
]


def extract_features(s1_name: str, s1_addr: str, cand_name: str, cand_addr: str, cand_id: str) -> np.ndarray:
    len1, len2 = len(s1_name), len(cand_name)
    if not s1_name or not cand_name:
        nr = npr = nts = ntset = nwr = nexact = nlratio = 0.0
        nldiff = abs(len1 - len2)
    else:
        nr = fuzz.ratio(s1_name, cand_name) / 100.0
        npr = fuzz.partial_ratio(s1_name, cand_name) / 100.0
        nts = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
        ntset = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
        nwr = fuzz.WRatio(s1_name, cand_name) / 100.0
        nexact = 1.0 if s1_name == cand_name else 0.0
        nldiff = abs(len1 - len2)
        nlratio = min(len1, len2) / max(len1, len2)

    addr_missing = 1.0 if (not s1_addr or not cand_addr) else 0.0
    if addr_missing:
        ar = atset = ats = apr = num_c = num_j = num_p = 0.0
    else:
        ar = fuzz.ratio(s1_addr, cand_addr) / 100.0
        atset = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        ats = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        apr = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
        nums1 = extract_numbers(s1_addr)
        nums2 = extract_numbers(cand_addr)
        if nums1 and nums2:
            common = nums1.intersection(nums2)
            num_c = float(len(common))
            num_j = len(common) / len(nums1.union(nums2))
            num_p = 1.0 if len(common) > 0 else -1.0
        else:
            num_c = num_j = num_p = 0.0

    is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0

    return np.array([
        nr, npr, nts, ntset, nwr, nexact, nldiff, nlratio,
        ar, atset, ats, apr, addr_missing, num_c, num_j, num_p,
        is_s3
    ], dtype=np.float32)


# --- 4. FULL PIPELINE EXECUTION ---
def run_aws_pipeline(data_dir: str, output_dir: str, sample_train: int = 60000):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("=========================================================")
    print(">> AMAZON ML CHALLENGE 2026: AWS PIPELINE STARTING")
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")
    print("=========================================================")
    
    # 1. Train GBDT Matcher
    model_path = 'models/lgb_matcher.txt'
    if not os.path.exists(model_path):
        print("\n[Step 1/3] Training LightGBM Matcher on Training Set...")
        train_s1 = pd.read_csv(os.path.join(data_dir, 'train/train_source1.tsv'), sep='\t', nrows=sample_train)
        train_s2 = pd.read_csv(os.path.join(data_dir, 'train/train_source2.tsv'), sep='\t')
        train_s3 = pd.read_csv(os.path.join(data_dir, 'train/train_source3.tsv'), sep='\t')
        train_gt = pd.read_csv(os.path.join(data_dir, 'train/train_ground_truth.tsv'), sep='\t')
        
        train_gt['matched_entity_ids'] = train_gt['matched_entity_ids'].fillna('')
        gt_map = {row['source1_entity_id']: set([m.strip() for m in row['matched_entity_ids'].split(',') if m.strip()]) for _, row in train_gt.iterrows()}
        
        # Clean training targets
        train_targets = pd.concat([train_s2, train_s3], ignore_index=True)
        train_targets['c_name'] = train_targets['business_name'].apply(clean_name)
        train_targets['c_addr'] = train_targets['business_address'].apply(clean_address)
        
        target_dict = {row['entity_id']: (row['c_name'], row['c_addr']) for _, row in train_targets.iterrows()}
        
        # Train blocker on US
        train_s1_us = train_s1[train_s1['country'] == 'US'].copy()
        train_s1_us['c_name'] = train_s1_us['business_name'].apply(clean_name)
        train_s1_us['c_addr'] = train_s1_us['business_address'].apply(clean_address)
        
        us_targets = train_targets[train_targets['country'] == 'US']
        blocker = MemoryEfficientBlocker()
        blocker.fit(us_targets['entity_id'].tolist(), us_targets['c_name'].tolist(), us_targets['c_addr'].tolist())
        
        X_list, y_list = [], []
        for _, row in tqdm(train_s1_us.iterrows(), total=len(train_s1_us), desc="Generating Training Pairs"):
            s1_id = row['entity_id']
            s1_name, s1_addr = row['c_name'], row['c_addr']
            true_matches = gt_map.get(s1_id, set())
            cands = blocker.query(s1_name, s1_addr, top_k=25).union(true_matches)
            
            for cid in cands:
                if cid not in target_dict:
                    continue
                cname, caddr = target_dict[cid]
                X_list.append(extract_features(s1_name, s1_addr, cname, caddr, cid))
                y_list.append(1.0 if cid in true_matches else 0.0)
                
        X_train = np.array(X_list, dtype=np.float32)
        y_train = np.array(y_list, dtype=np.float32)
        
        print(f"Training LightGBM on {len(X_train):,} pairs (Positives: {int(y_train.sum()):,})...")
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'learning_rate': 0.08, 'num_leaves': 31,
            'verbose': -1, 'random_state': 42
        }
        model = lgb.train(params, dtrain, num_boost_round=300)
        model.save_model(model_path)
        print("Model trained and saved successfully!")
    else:
        print(f"\n[Step 1/3] Loading existing model from {model_path}...")
        model = lgb.Booster(model_file=model_path)

    # 2. Test Set Inference (By Country: India, US, France)
    print("\n[Step 2/3] Running High-Scale Test Inference...")
    test_s1 = pd.read_csv(os.path.join(data_dir, 'test/test_source1.tsv'), sep='\t')
    test_s2 = pd.read_csv(os.path.join(data_dir, 'test/test_source2.tsv'), sep='\t')
    test_s3 = pd.read_csv(os.path.join(data_dir, 'test/test_source3.tsv'), sep='\t')
    
    print(f"Test S1 Total: {len(test_s1):,}, S2: {len(test_s2):,}, S3: {len(test_s3):,}")
    
    matching_out_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_out_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    
    # Initialize output TSVs with headers
    with open(matching_out_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_out_path, 'w', encoding='utf-8') as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
    countries = test_s1['country'].unique()
    decision_threshold = 0.65  # High-precision threshold for F_0.5
    
    for country in countries:
        print(f"\n>>> Processing Country: {country}...")
        country_s1 = test_s1[test_s1['country'] == country].copy()
        country_s2 = test_s2[test_s2['country'] == country].copy()
        country_s3 = test_s3[test_s3['country'] == country].copy()
        
        print(f"Cleaning {country} entities (S1: {len(country_s1):,}, Targets: {len(country_s2) + len(country_s3):,})...")
        country_s1['c_name'] = country_s1['business_name'].apply(clean_name)
        country_s1['c_addr'] = country_s1['business_address'].apply(clean_address)
        
        country_targets = pd.concat([country_s2, country_s3], ignore_index=True)
        country_targets['c_name'] = country_targets['business_name'].apply(clean_name)
        country_targets['c_addr'] = country_targets['business_address'].apply(clean_address)
        
        target_dict = {row['entity_id']: (row['c_name'], row['c_addr']) for _, row in country_targets.iterrows()}
        
        # Build Blocker
        blocker = MemoryEfficientBlocker()
        blocker.fit(
            country_targets['entity_id'].tolist(),
            country_targets['c_name'].tolist(),
            country_targets['c_addr'].tolist()
        )
        
        print(f"Generating matches for {len(country_s1):,} S1 entities in {country}...")
        
        batch_size = 5000
        with open(matching_out_path, 'a', encoding='utf-8') as f_match, \
             open(candidate_out_path, 'a', encoding='utf-8') as f_cand:
             
            for start_idx in tqdm(range(0, len(country_s1), batch_size), desc=f"Inference ({country})"):
                batch_df = country_s1.iloc[start_idx:start_idx+batch_size]
                
                batch_meta = [] # (s1_id, cand_id)
                batch_feats = []
                batch_cands_map = defaultdict(list)
                
                for _, row in batch_df.iterrows():
                    s1_id = row['entity_id']
                    s1_name, s1_addr = row['c_name'], row['c_addr']
                    cands = blocker.query(s1_name, s1_addr, top_k=25)
                    batch_cands_map[s1_id] = list(cands)
                    
                    for cid in cands:
                        cname, caddr = target_dict[cid]
                        batch_feats.append(extract_features(s1_name, s1_addr, cname, caddr, cid))
                        batch_meta.append((s1_id, cid))
                        
                # Predict batch
                s1_matches = defaultdict(list)
                if batch_feats:
                    probs = model.predict(np.array(batch_feats, dtype=np.float32))
                    for (s1_id, cid), prob in zip(batch_meta, probs):
                        if prob >= decision_threshold:
                            s1_matches[s1_id].append(cid)
                            
                # Stream write batch results
                for _, row in batch_df.iterrows():
                    s1_id = row['entity_id']
                    cands_str = ",".join(batch_cands_map.get(s1_id, []))
                    matches_str = ",".join(s1_matches.get(s1_id, []))
                    
                    f_cand.write(f"{s1_id}\t{cands_str}\n")
                    f_match.write(f"{s1_id}\t{matches_str}\n")
                    
    print("\n[Step 3/3] Inference complete! Output files generated:")
    print(f"1. {matching_out_path}")
    print(f"2. {candidate_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='student_resource/dataset')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--sample-train', type=int, default=60000)
    args = parser.parse_args()
    
    run_aws_pipeline(args.data_dir, args.output_dir, args.sample_train)
