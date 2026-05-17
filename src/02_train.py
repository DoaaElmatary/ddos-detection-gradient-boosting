"""
02_train.py
Model training and evaluation for DDoS detection
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Configuration
# ============================================
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

# اسم ملف البيانات (في مجلد data)
DATA_FILE = 'data/ddos_balanced_7000_clean.csv'

# Feature columns (12 features)
FEATURE_COLS = [
    'total_packets', 'size_mean', 'size_std', 'size_min', 'size_max',
    'tcp_ratio', 'udp_ratio', 'icmp_ratio',
    'duration_sec', 'throughput_mbps', 'jitter_ms', 'packet_rate_pps'
]

# Models to train
MODELS = {
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, random_state=RANDOM_STATE
    ),
    'SVM (RBF)': SVC(
        kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=RANDOM_STATE
    )
}

# ============================================
# Load and prepare data
# ============================================
def load_and_prepare_data():
    """Load dataset and prepare features and labels"""
    print("Loading dataset...")
    
    # Check if file exists
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: File not found: {DATA_FILE}")
        print("   Please make sure the dataset is in the 'data/' folder")
        print("   Or update DATA_FILE variable with the correct path")
        return None, None, None
    
    df = pd.read_csv(DATA_FILE)
    
    # Features (use only available columns)
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    X = df[available_features].values
    
    # Labels (encode to integers)
    le = LabelEncoder()
    y = le.fit_transform(df['attack_type'])
    
    # Create results folder if not exists
    os.makedirs('results', exist_ok=True)
    
    # Save label encoder for later use
    joblib.dump(le, 'results/label_encoder.pkl')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {le.classes_}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, le

# ============================================
# Train-test split with stratification
# ============================================
def split_data(X, y):
    """Stratified train-test split (70/30)"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]} samples")
    print(f"Test size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

# ============================================
# Feature standardization
# ============================================
def standardize_features(X_train, X_test):
    """Standardize features to zero mean and unit variance"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for inference
    joblib.dump(scaler, 'results/scaler.pkl')
    
    return X_train_scaled, X_test_scaled

# ============================================
# Train and evaluate a single model
# ============================================
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Train model and return performance metrics"""
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")
    
    # Measure training time
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    # Measure inference time
    inference_times = []
    for i in range(min(500, len(X_test))):
        start_infer = time.time()
        _ = model.predict(X_test[i:i+1])
        inference_times.append(time.time() - start_infer)
    inference_time_ms = np.mean(inference_times) * 1000
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Save model
    joblib.dump(model, f'results/model_{model_name.replace(" ", "_").lower()}.pkl')
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"CV Mean (±Std): {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Inference Time: {inference_time_ms:.3f} ms/sample")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_time': train_time,
        'inference_time_ms': inference_time_ms
    }

# ============================================
# Main execution
# ============================================
def main():
    print("="*60)
    print("DDoS Detection - Model Training and Evaluation")
    print("="*60)
    
    # Load data
    X, y, label_encoder = load_and_prepare_data()
    if X is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Standardize features
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    
    # Train and evaluate all models
    results = {}
    for model_name, model in MODELS.items():
        result = train_and_evaluate(
            model, model_name, 
            X_train_scaled, X_test_scaled, 
            y_train, y_test
        )
        results[model_name] = result
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE (Table 8 in paper)")
    print("="*60)
    print(f"{'Model':<22} {'Accuracy':<10} {'F1-Score':<10} {'CV Mean (±Std)':<18} {'Train Time (s)':<15} {'Inference (ms)':<12}")
    print("-"*85)
    for model_name, res in results.items():
        print(f"{model_name:<22} {res['accuracy']:<10.4f} {res['f1_score']:<10.4f} {res['cv_mean']:.4f} (±{res['cv_std']:.4f})   {res['train_time']:<15.2f} {res['inference_time_ms']:<12.3f}")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'Model': res['model_name'],
            'Accuracy': res['accuracy'],
            'F1-Score': res['f1_score'],
            'CV_Mean': res['cv_mean'],
            'CV_Std': res['cv_std'],
            'Training_Time_s': res['train_time'],
            'Inference_Time_ms': res['inference_time_ms']
        }
        for res in results.values()
    ])
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/table8_results.csv', index=False)
    print("\n✓ Results saved to 'results/table8_results.csv'")
    
    print("\n✓ Training and evaluation complete!")

if __name__ == "__main__":
    main()