# train.py
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, auc, classification_report
import lightgbm as lgb
import joblib

def generate_synthetic_data(data_path):
    print(f"[*] {data_path} not found. Generating a robust synthetic dataset...")
    np.random.seed(42)
    n_samples = 5000
    
    df = pd.DataFrame({
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_samples)],
        'amount': np.random.exponential(scale=50, size=n_samples) + 5,
        'transaction_hour': np.random.randint(0, 24, n_samples),
        'device_trust_score': np.random.uniform(0, 1, n_samples),
        'velocity_last_24h': np.random.poisson(lam=2, size=n_samples) + 1,
        'cardholder_age': np.random.randint(18, 85, n_samples),
        'merchant_category': np.random.choice(['Retail', 'Travel', 'Online', 'Food', 'Entertainment'], n_samples),
        'foreign_transaction': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'location_mismatch': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    })
    
    # Induce logical fraud patterns for the model to learn
    fraud_prob = np.zeros(n_samples)
    fraud_prob += np.where(df['amount'] > 300, 0.1, 0)
    fraud_prob += np.where(df['velocity_last_24h'] > 6, 0.2, 0)
    fraud_prob += np.where(df['location_mismatch'] == 1, 0.3, 0)
    fraud_prob += np.where(df['device_trust_score'] < 0.2, 0.15, 0)
    
    df['is_fraud'] = (np.random.rand(n_samples) < fraud_prob).astype(int)
    df.to_csv(data_path, index=False)
    print(f"[+] Synthetic data saved to {data_path}")
    return df

def run_pipeline(data_path, model_dir):
    if not os.path.exists(data_path):
        df = generate_synthetic_data(data_path)
    else:
        print(f"[*] Loading data from: {data_path}")
        df = pd.read_csv(data_path)
    
    # 1. Enforce Temporal Ordering
    print("[*] Sorting data chronologically...")
    df = df.sort_values(by=['transaction_hour']).reset_index(drop=True)
    
    # Drop identifiers and target for X
    cols_to_drop = ['transaction_id', 'is_fraud']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['is_fraud']
    
    # Removed card_id from features
    num_features = ['amount', 'transaction_hour', 'device_trust_score', 'velocity_last_24h', 'cardholder_age']
    cat_features = ['merchant_category']
    passthrough_features = ['foreign_transaction', 'location_mismatch']
    
    # 2. Sequential Split
    train_idx = int(len(df) * 0.70)
    val_idx = int(len(df) * 0.85)
    
    X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
    X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
    X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]
    
    # 3. Processing Pipeline
    print("[*] Building transformation network...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ],
        remainder='passthrough'
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    all_feature_names = num_features + encoded_cat_names + passthrough_features
    
    # 4. Handle Imbalance
    pos_count = np.sum(y_train == 1)
    scale_weight = np.sum(y_train == 0) / pos_count if pos_count > 0 else 1.0
    
    print(f"[*] Training LightGBM (Weight Multiplier: {scale_weight:.2f})...")
    base_model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=5, num_leaves=31,
        scale_pos_weight=scale_weight, random_state=42, n_jobs=-1, verbose=-1,
        deterministic=True, force_row_wise=True
    )
    base_model.fit(X_train_processed, y_train)
    
    # 5. Calibration Layer
    print("[*] Calibrating risk output scales...")
    val_raw_probs = base_model.predict_proba(X_val_processed)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_raw_probs, y_val)
    
    # 6. Evaluate
    test_cal_probs = calibrator.transform(base_model.predict_proba(X_test_processed)[:, 1])
    precision, recall, _ = precision_recall_curve(y_test, test_cal_probs)
    print(f"\n[*] PR-AUC Baseline Score: {auc(recall, precision):.4f}\n")
    
    # 7. Package Unified Core State
    os.makedirs(model_dir, exist_ok=True)
    pipeline_artifact = {
        'preprocessor': preprocessor,
        'model': base_model,
        'calibrator': calibrator,
        'feature_names': all_feature_names
    }
    
    out_path = os.path.join(model_dir, 'fraud_pipeline.joblib')
    joblib.dump(pipeline_artifact, out_path)
    print(f"[+] Pipeline artifact exported safely to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='credit_card_fraud_10k.csv')
    parser.add_argument('--out_dir', type=str, default='models')
    args = parser.parse_args()
    run_pipeline(args.data, args.out_dir)