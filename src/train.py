"""
Training pipeline for Business Entity Resolution pair classifier.
Trains LightGBM on candidate pairs and optimizes F_0.5 threshold.
"""

import time
import sys
import os
import pickle
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import lightgbm as lgb
from collections import defaultdict
from sklearn.model_selection import KFold
from src.preprocessing import clean_name, clean_address
from src.blocking import MultiIndexBlocker
from src.features import extract_pair_features, FEATURE_NAMES
from src.metrics import compute_macro_f05


def prepare_training_data(sample_s1_size=25000, country='US'):
    print(f"\n[1/4] Preparing training data for {country} ({sample_s1_size:,} S1 entities)...")
    
    s1_all = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t')
    s2_all = pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t')
    s3_all = pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t')
    gt_all = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t')
    
    # Filter country
    s1_df = s1_all[s1_all['country'] == country].head(sample_s1_size).copy()
    s1_ids = set(s1_df['entity_id'])
    
    gt_df = gt_all[gt_all['source1_entity_id'].isin(s1_ids)].copy()
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna('')
    gt_map = {
        row['source1_entity_id']: set([m.strip() for m in row['matched_entity_ids'].split(',') if m.strip()])
        for _, row in gt_df.iterrows()
    }
    
    s2_df = s2_all[s2_all['country'] == country].copy()
    s3_df = s3_all[s3_all['country'] == country].copy()
    
    print("Cleaning names and addresses...")
    s1_df['c_name'] = s1_df['business_name'].apply(clean_name)
    s1_df['c_addr'] = s1_df['business_address'].apply(clean_address)
    
    s2_df['c_name'] = s2_df['business_name'].apply(clean_name)
    s2_df['c_addr'] = s2_df['business_address'].apply(clean_address)
    
    s3_df['c_name'] = s3_df['business_name'].apply(clean_name)
    s3_df['c_addr'] = s3_df['business_address'].apply(clean_address)
    
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    target_dict = {
        row['entity_id']: (row['c_name'], row['c_addr'])
        for _, row in targets_df.iterrows()
    }
    
    # Blocking
    blocker = MultiIndexBlocker()
    blocker.fit_targets(
        entity_ids=targets_df['entity_id'].tolist(),
        names=targets_df['c_name'].tolist(),
        addresses=targets_df['c_addr'].tolist()
    )
    
    print("Generating candidate pairs & extracting features...")
    X_list = []
    y_list = []
    pair_metadata = [] # (s1_id, cand_id)
    
    for _, row in s1_df.iterrows():
        s1_id = row['entity_id']
        s1_name = row['c_name']
        s1_addr = row['c_addr']
        true_matches = gt_map.get(s1_id, set())
        
        cands = blocker.query(s1_name, s1_addr, top_k=25)
        
        # Include all queried candidates + true positives
        all_eval_cands = cands.union(true_matches)
        
        for cand_id in all_eval_cands:
            if cand_id not in target_dict:
                continue
            cand_name, cand_addr = target_dict[cand_id]
            feats = extract_pair_features(s1_name, s1_addr, cand_name, cand_addr, cand_id)
            is_match = 1.0 if cand_id in true_matches else 0.0
            
            X_list.append(feats)
            y_list.append(is_match)
            pair_metadata.append((s1_id, cand_id))
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"Dataset generated: {len(X):,} candidate pairs (Positives: {int(y.sum()):,}, Negatives: {int((1-y).sum()):,})")
    
    return X, y, pair_metadata, s1_df, gt_map


def train_and_evaluate():
    X, y, pair_meta, s1_df, gt_map = prepare_training_data(sample_s1_size=20000, country='US')
    
    # Train / Validation Split on S1 entities to prevent leakage
    unique_s1 = s1_df['entity_id'].unique()
    np.random.seed(42)
    val_s1 = set(np.random.choice(unique_s1, size=int(len(unique_s1) * 0.25), replace=False))
    
    train_indices = [i for i, (s1_id, _) in enumerate(pair_meta) if s1_id not in val_s1]
    val_indices = [i for i, (s1_id, _) in enumerate(pair_meta) if s1_id in val_s1]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    print(f"\n[2/4] Training LightGBM Classifier...")
    print(f"Train Pairs: {len(X_train):,}, Val Pairs: {len(X_val):,}")
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=FEATURE_NAMES)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 31,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=400,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    os.makedirs('models', exist_ok=True)
    model.save_model('models/lgb_matcher.txt')
    print("Model saved to models/lgb_matcher.txt")
    
    print(f"\n[3/4] Threshold Tuning for F_0.5 Score...")
    val_preds_prob = model.predict(X_val)
    val_meta = [pair_meta[i] for i in val_indices]
    
    val_gt_map = {s1_id: gt_map[s1_id] for s1_id in val_s1}
    
    best_thresh = 0.5
    best_f05 = 0.0
    
    for thresh in np.arange(0.3, 0.95, 0.05):
        pred_map = defaultdict(list)
        for (s1_id, cand_id), prob in zip(val_meta, val_preds_prob):
            if prob >= thresh:
                pred_map[s1_id].append(cand_id)
                
        metrics = compute_macro_f05(val_gt_map, pred_map)
        macro_f05 = metrics['macro_f05']
        print(f"Threshold: {thresh:.2f} | Macro F_0.5: {macro_f05:.4f} | Singletons: {metrics['singleton_f05']:.4f} | Non-Singletons: {metrics['non_singleton_f05']:.4f}")
        
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = thresh
            
    print(f"\n==========================================")
    print(f"🎯 OPTIMAL F_0.5 VALIDATION RESULTS")
    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"Best Validation Macro F_0.5: {best_f05:.4f}")
    print(f"==========================================\n")
    
    # Save optimal threshold config
    with open('models/model_config.pkl', 'wb') as f:
        pickle.dump({'best_threshold': best_thresh, 'best_f05': best_f05}, f)


if __name__ == "__main__":
    train_and_evaluate()
