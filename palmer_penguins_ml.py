
"""
Palmer Penguins Classification - Complete ML Pipeline
Algorithms: LightGBM, XGBoost, AdaBoost, CatBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean Palmer Penguins dataset"""
    print("🐧 Loading Palmer Penguins Dataset...")

    # Create sample data (in practice, you'd load from CSV)
    data = {
        'species': ['Adelie'] * 50 + ['Chinstrap'] * 50 + ['Gentoo'] * 50,
        'island': ['Torgersen'] * 25 + ['Biscoe'] * 25 + ['Dream'] * 50 + ['Biscoe'] * 50,
        'bill_length_mm': [39.1, 39.5, 40.3, 36.7, 39.3, 38.9, 39.2, 34.1, 42.0, 37.8] * 15,
        'bill_depth_mm': [18.7, 17.4, 18.0, 19.3, 20.6, 17.8, 19.6, 18.1, 20.2, 17.1] * 15,
        'flipper_length_mm': [181, 186, 195, 193, 190, 181, 195, 193, 190, 186] * 15,
        'body_mass_g': [3750, 3800, 3250, 3450, 3650, 3625, 4675, 3475, 4250, 3300] * 15,
        'sex': ['Male', 'Female'] * 75
    }
    df = pd.DataFrame(data)

    print(f"📊 Original shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")

    # Data Cleaning
    print("\n🧹 Starting Data Cleaning...")

    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values}")

    # Handle missing values (if any)
    df = df.dropna()

    # Encode categorical variables
    le_species = LabelEncoder()
    le_island = LabelEncoder()
    le_sex = LabelEncoder()

    df['species_encoded'] = le_species.fit_transform(df['species'])
    df['island_encoded'] = le_island.fit_transform(df['island'])
    df['sex_encoded'] = le_sex.fit_transform(df['sex'])

    # Features and target
    feature_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                   'body_mass_g', 'island_encoded', 'sex_encoded']
    X = df[feature_cols]
    y = df['species_encoded']

    print(f"✅ Cleaned shape: {X.shape}")
    print(f"✅ Classes: {np.unique(y)}")

    return X, y, le_species

def train_models(X, y):
    """Train all boosting models"""
    print("\n🚀 Training Models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # 1. LightGBM
    print("\n📊 Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_accuracy = accuracy_score(y_test, lgb_pred)
    results['LightGBM'] = {'model': lgb_model, 'accuracy': lgb_accuracy, 'predictions': lgb_pred}

    # 2. XGBoost
    print("📊 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    results['XGBoost'] = {'model': xgb_model, 'accuracy': xgb_accuracy, 'predictions': xgb_pred}

    # 3. AdaBoost
    print("📊 Training AdaBoost...")
    ada_model = AdaBoostClassifier(random_state=42, algorithm='SAMME')
    ada_model.fit(X_train, y_train)
    ada_pred = ada_model.predict(X_test)
    ada_accuracy = accuracy_score(y_test, ada_pred)
    results['AdaBoost'] = {'model': ada_model, 'accuracy': ada_accuracy, 'predictions': ada_pred}

    # 4. CatBoost
    print("📊 Training CatBoost...")
    cat_model = cb.CatBoostClassifier(
        iterations=100,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_test)
    cat_accuracy = accuracy_score(y_test, cat_pred)
    results['CatBoost'] = {'model': cat_model, 'accuracy': cat_accuracy, 'predictions': cat_pred}

    return results, X_test, y_test

def evaluate_models(results, X_test, y_test, label_encoder):
    """Evaluate and compare all models"""
    print("\n📈 Model Evaluation Results:")
    print("=" * 50)

    for name, result in results.items():
        accuracy = result['accuracy']
        print(f"{name}: {accuracy:.4f}")

    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']

    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

    # Detailed evaluation of best model
    best_pred = results[best_model_name]['predictions']
    print(f"\n📊 Detailed Results for {best_model_name}:")
    print("Classification Report:")
    print(classification_report(y_test, best_pred, 
                              target_names=label_encoder.classes_))

def main():
    """Main execution function"""
    print("🐧 PALMER PENGUINS CLASSIFICATION PIPELINE")
    print("=" * 60)

    # Load and clean data
    X, y, label_encoder = load_and_clean_data()

    # Train models
    results, X_test, y_test = train_models(X, y)

    # Evaluate models
    evaluate_models(results, X_test, y_test, label_encoder)

    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
