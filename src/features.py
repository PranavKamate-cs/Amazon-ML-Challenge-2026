"""
Pairwise Feature Extraction for Business Entity Resolution.
Accelerated using RapidFuzz (C++ bindings).
"""

from typing import Dict, List, Tuple, Set
import numpy as np
from rapidfuzz import fuzz
from src.preprocessing import extract_numbers


FEATURE_NAMES = [
    'name_ratio',
    'name_partial_ratio',
    'name_token_sort',
    'name_token_set',
    'name_wratio',
    'name_exact_match',
    'name_len_diff',
    'name_len_ratio',
    'addr_ratio',
    'addr_token_set',
    'addr_token_sort',
    'addr_partial_ratio',
    'addr_is_missing',
    'num_common_count',
    'num_jaccard',
    'num_mismatch_penalty',
    'is_s3',
]


def extract_pair_features(
    s1_name: str, s1_addr: str,
    s2_name: str, s2_addr: str,
    s2_id: str
) -> np.ndarray:
    """Extracts a vector of discriminative features for a candidate pair."""
    # Lengths
    len1_name = len(s1_name)
    len2_name = len(s2_name)
    
    # 1. Name Features
    if not s1_name or not s2_name:
        name_r = name_pr = name_ts = name_tset = name_wr = name_exact = 0.0
        name_ldiff = abs(len1_name - len2_name)
        name_lratio = 0.0
    else:
        name_r = fuzz.ratio(s1_name, s2_name) / 100.0
        name_pr = fuzz.partial_ratio(s1_name, s2_name) / 100.0
        name_ts = fuzz.token_sort_ratio(s1_name, s2_name) / 100.0
        name_tset = fuzz.token_set_ratio(s1_name, s2_name) / 100.0
        name_wr = fuzz.WRatio(s1_name, s2_name) / 100.0
        name_exact = 1.0 if s1_name == s2_name else 0.0
        name_ldiff = abs(len1_name - len2_name)
        name_lratio = min(len1_name, len2_name) / max(len1_name, len2_name)

    # 2. Address Features
    addr_missing = 1.0 if (not s1_addr or not s2_addr) else 0.0
    if addr_missing:
        addr_r = addr_tset = addr_ts = addr_pr = 0.0
        num_common = 0.0
        num_jacc = 0.0
        num_penalty = 0.0
    else:
        addr_r = fuzz.ratio(s1_addr, s2_addr) / 100.0
        addr_tset = fuzz.token_set_ratio(s1_addr, s2_addr) / 100.0
        addr_ts = fuzz.token_sort_ratio(s1_addr, s2_addr) / 100.0
        addr_pr = fuzz.partial_ratio(s1_addr, s2_addr) / 100.0
        
        nums1 = extract_numbers(s1_addr)
        nums2 = extract_numbers(s2_addr)
        
        if nums1 and nums2:
            common = nums1.intersection(nums2)
            num_common = float(len(common))
            num_jacc = len(common) / len(nums1.union(nums2))
            num_penalty = 1.0 if len(common) > 0 else -1.0
        else:
            num_common = 0.0
            num_jacc = 0.0
            num_penalty = 0.0

    # 3. Source Metadata
    is_s3 = 1.0 if s2_id.startswith('S3-') else 0.0

    return np.array([
        name_r, name_pr, name_ts, name_tset, name_wr, name_exact,
        name_ldiff, name_lratio,
        addr_r, addr_tset, addr_ts, addr_pr, addr_missing,
        num_common, num_jacc, num_penalty,
        is_s3
    ], dtype=np.float32)
