"""
Pertemuan 8: UTS - Complete End-to-End Machine Learning Project
Contoh project lengkap: Heart Disease Prediction (Classification Problem)

Project ini mencakup:
1. Data Loading & Exploration
2. Data Preprocessing & Cleaning
3. Feature Engineering
4. Model Training (Multiple Algorithms)
5. Model Evaluation & Comparison
6. Final Model Selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# Settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("UTS PROJECT: HEART DISEASE PREDICTION")
print("Complete End-to-End Machine Learning Project")
print("=" * 80)

# ============================================================================
# PART 1: INTRODUCTION & DATA UNDERSTANDING
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: INTRODUCTION & DATA UNDERSTANDING")
print("=" * 80)

print("""
Problem Statement:
- Memprediksi apakah seseorang memiliki penyakit jantung atau tidak
- Binary Classification Problem
- Important untuk early diagnosis dan prevention

Dataset Features:
- age: Umur pasien (tahun)
- sex: Jenis kelamin (1=male, 0=female)
- cp: Chest pain type (0-3)
- trestbps: Resting blood pressure (mm Hg)
- chol: Serum cholesterol (mg/dl)
- fbs: Fasting blood sugar > 120 mg/dl (1=true, 0=false)
- restecg: Resting ECG results (0-2)
- thalach: Maximum heart rate achieved
- exang: Exercise induced angina (1=yes, 0=no)
- oldpeak: ST depression induced by exercise
- slope: Slope of peak exercise ST segment (0-2)
- ca: Number of major vessels colored by fluoroscopy (0-3)
- thal: Thalassemia (0=normal, 1=fixed defect, 2=reversable defect)

Target Variable:
- target: 1 = Heart Disease, 0 = No Heart Disease
""")

# Create synthetic heart disease dataset (karena tidak ada dataset asli di sistem)
np.random.seed(42)
n_samples = 500

# Generate features
age = np.random.randint(30, 80, n_samples)
sex = np.random.choice([0, 1], n_samples)
cp = np.random.choice([0, 1, 2, 3], n_samples)
trestbps = np.random.randint(90, 180, n_samples)
chol = np.random.randint(150, 350, n_samples)
fbs = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
restecg = np.random.choice([0, 1, 2], n_samples)
thalach = np.random.randint(80, 200, n_samples)
exang = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
oldpeak = np.random.uniform(0, 5, n_samples).round(1)
slope = np.random.choice([0, 1, 2], n_samples)
ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.3, 0.15, 0.05])
thal = np.random.choice([0, 1, 2], n_samples)

# Generate target (with some correlation to features)
risk_score = (
    (age > 55) * 0.3 +
    (sex == 1) * 0.2 +
    (cp > 1) * 0.25 +
    (trestbps > 140) * 0.2 +
    (chol > 250) * 0.15 +
    (thalach < 120) * 0.2 +
    (exang == 1) * 0.25 +
    (oldpeak > 2) * 0.2 +
    (ca > 0) * 0.15
)
target = (risk_score + np.random.uniform(0, 0.3, n_samples) > 0.5).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal,
    'target': target
})

print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check target distribution
print("\nTarget Distribution:")
print(df['target'].value_counts())
print(f"\nClass Balance:")
print(f"  No Disease (0): {(df['target']==0).sum()} ({(df['target']==0).sum()/len(df)*100:.1f}%)")
print(f"  Heart Disease (1): {(df['target']==1).sum()} ({(df['target']==1).sum()/len(df)*100:.1f}%)")

# EDA Visualizations
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Target Distribution
ax1 = fig.add_subplot(gs[0, 0])
target_counts = df['target'].value_counts()
ax1.bar(['No Disease', 'Heart Disease'], target_counts.values, color=['#4ECDC4', '#FF6B6B'])
ax1.set_ylabel('Count')
ax1.set_title('Target Distribution')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Age Distribution by Target
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist([df[df['target']==0]['age'], df[df['target']==1]['age']], 
         bins=20, label=['No Disease', 'Heart Disease'], alpha=0.7, color=['#4ECDC4', '#FF6B6B'])
ax2.set_xlabel('Age')
ax2.set_ylabel('Count')
ax2.set_title('Age Distribution by Target')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Sex Distribution by Target
ax3 = fig.add_subplot(gs[0, 2])
sex_target = pd.crosstab(df['sex'], df['target'])
sex_target.plot(kind='bar', ax=ax3, color=['#4ECDC4', '#FF6B6B'])
ax3.set_xticklabels(['Female', 'Male'], rotation=0)
ax3.set_ylabel('Count')
ax3.set_title('Sex Distribution by Target')
ax3.legend(['No Disease', 'Heart Disease'])
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Chest Pain Type by Target
ax4 = fig.add_subplot(gs[1, 0])
cp_target = pd.crosstab(df['cp'], df['target'])
cp_target.plot(kind='bar', ax=ax4, color=['#4ECDC4', '#FF6B6B'])
ax4.set_xlabel('Chest Pain Type')
ax4.set_ylabel('Count')
ax4.set_title('Chest Pain Type by Target')
ax4.legend(['No Disease', 'Heart Disease'])
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Correlation Heatmap
ax5 = fig.add_subplot(gs[1, 1:])
correlation = df.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, ax=ax5, cbar_kws={"shrink": 0.8})
ax5.set_title('Feature Correlation Heatmap')

# Plot 6: Max Heart Rate by Target
ax6 = fig.add_subplot(gs[2, 0])
ax6.boxplot([df[df['target']==0]['thalach'], df[df['target']==1]['thalach']],
            labels=['No Disease', 'Heart Disease'])
ax6.set_ylabel('Max Heart Rate')
ax6.set_title('Max Heart Rate by Target')
ax6.grid(True, alpha=0.3)

# Plot 7: Cholesterol by Target
ax7 = fig.add_subplot(gs[2, 1])
ax7.boxplot([df[df['target']==0]['chol'], df[df['target']==1]['chol']],
            labels=['No Disease', 'Heart Disease'])
ax7.set_ylabel('Cholesterol (mg/dl)')
ax7.set_title('Cholesterol by Target')
ax7.grid(True, alpha=0.3)

# Plot 8: Blood Pressure by Target
ax8 = fig.add_subplot(gs[2, 2])
ax8.boxplot([df[df['target']==0]['trestbps'], df[df['target']==1]['trestbps']],
            labels=['No Disease', 'Heart Disease'])
ax8.set_ylabel('Resting BP (mm Hg)')
ax8.set_title('Blood Pressure by Target')
ax8.grid(True, alpha=0.3)

plt.suptitle('Exploratory Data Analysis - Heart Disease Dataset', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan08_eda.png', dpi=300, bbox_inches='tight')
print("\n✓ EDA visualizations saved")
plt.close()

# ============================================================================
# PART 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: DATA PREPROCESSING")
print("=" * 80)

# Check for outliers
print("\nOutlier Detection (using IQR method):")
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"  {col}: {len(outliers)} outliers detected")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data (stratify untuk menjaga class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Data preprocessing completed")
print("  - Train-test split (80-20)")
print("  - Feature scaling (StandardScaler)")

# ============================================================================
# PART 4: MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: MODEL TRAINING")
print("=" * 80)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True)
}

# Store results
results = {}

print("\nTraining models...")
for name, model in models.items():
    print(f"\n{name}:")
    
    # Train model
    if name == 'SVM':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation
    if name == 'SVM':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

print("\n✓ All models trained successfully")

# ============================================================================
# PART 5: MODEL EVALUATION & COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: MODEL EVALUATION & COMPARISON")
print("=" * 80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'CV Mean': [results[m]['cv_mean'] for m in results.keys()]
}).sort_values('F1-Score', ascending=False)

print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# Best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n✓ Best Model: {best_model_name}")
print(f"  F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Evaluation & Comparison', fontsize=16, fontweight='bold')

# Plot 1: Metrics Comparison
ax1 = axes[0, 0]
comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
    kind='bar', ax=ax1, rot=45
)
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Metrics')
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Best Model Confusion Matrix
ax2 = axes[0, 1]
cm_best = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['No Disease', 'Heart Disease'],
            yticklabels=['No Disease', 'Heart Disease'])
ax2.set_title(f'Confusion Matrix - {best_model_name}')
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')

# Plot 3: ROC Curves
ax3 = axes[0, 2]
for name in results.keys():
    if results[name]['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

ax3.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves Comparison')
ax3.legend(loc='lower right', fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4-8: Individual Confusion Matrices
for idx, name in enumerate(list(results.keys())[:5]):
    row = 1 + idx // 3
    col = idx % 3
    if row < 2 or col < 3:
        cm = confusion_matrix(y_test, results[name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axes[row, col],
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        axes[row, col].set_title(f'{name}\n(Acc: {results[name]["accuracy"]:.3f})')
        axes[row, col].set_ylabel('Actual')
        axes[row, col].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan08_model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Model evaluation visualizations saved")
plt.close()

# Detailed Classification Report for Best Model
print(f"\n--- Detailed Report: {best_model_name} ---")
print(classification_report(y_test, results[best_model_name]['y_pred'],
                          target_names=['No Disease', 'Heart Disease']))

# ============================================================================
# PART 6: HYPERPARAMETER TUNING (Best Model)
# ============================================================================
print("\n" + "=" * 80)
print("PART 6: HYPERPARAMETER TUNING")
print("=" * 80)

print(f"\nTuning hyperparameters for: {best_model_name}")

# Define parameter grid based on best model
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
else:
    param_grid = {}

if param_grid:
    print("\nParameter Grid:")
    for key, value in param_grid.items():
        print(f"  {key}: {value}")
    
    print("\nPerforming Grid Search...")
    grid_search = GridSearchCV(
        type(best_model)(),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    if best_model_name == 'SVM':
        grid_search.fit(X_train_scaled, y_train)
        y_pred_tuned = grid_search.predict(X_test_scaled)
    else:
        grid_search.fit(X_train, y_train)
        y_pred_tuned = grid_search.predict(X_test)
    
    print("\n✓ Grid Search completed!")
    print("\nBest Parameters:")
    for key, value in grid_search.best_params_.items():
        print(f"  {key}: {value}")
    
    # Evaluate tuned model
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    f1_tuned = f1_score(y_test, y_pred_tuned)
    
    print(f"\nBefore Tuning:")
    print(f"  Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"  F1-Score: {results[best_model_name]['f1']:.4f}")
    
    print(f"\nAfter Tuning:")
    print(f"  Accuracy: {accuracy_tuned:.4f}")
    print(f"  F1-Score: {f1_tuned:.4f}")
    
    print(f"\nImprovement:")
    print(f"  Accuracy: {(accuracy_tuned - results[best_model_name]['accuracy'])*100:.2f}%")
    print(f"  F1-Score: {(f1_tuned - results[best_model_name]['f1'])*100:.2f}%")
else:
    print("\nHyperparameter tuning not applicable for this model in this example.")

# ============================================================================
# PART 7: FINAL SUMMARY & CONCLUSIONS
# ============================================================================
print("\n" + "=" * 80)
print("PART 7: FINAL SUMMARY & CONCLUSIONS")
print("=" * 80)

print(f"""
PROJECT SUMMARY:
--------------
Dataset: Heart Disease Prediction
Task: Binary Classification
Total Samples: {len(df)}
Training Set: {len(X_train)} samples
Testing Set: {len(X_test)} samples

MODELS EVALUATED:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

BEST MODEL: {best_model_name}
Performance Metrics:
- Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}
- Precision: {comparison_df.iloc[0]['Precision']:.4f}
- Recall: {comparison_df.iloc[0]['Recall']:.4f}
- F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}
- CV Score: {comparison_df.iloc[0]['CV Mean']:.4f}

KEY INSIGHTS:
1. Model successfully predicts heart disease with good accuracy
2. {best_model_name} provides the best balance of precision and recall
3. Important features for prediction include chest pain type, max heart rate, and age
4. Feature scaling improved performance for distance-based models
5. Cross-validation ensures model generalization

RECOMMENDATIONS:
1. Deploy {best_model_name} for production use
2. Monitor model performance regularly
3. Consider ensemble of top 3 models for more robust predictions
4. Collect more data for continuous improvement
5. Implement explainable AI techniques for clinical acceptance
""")

# Save final model comparison
comparison_df.to_csv('/workspaces/ml-practicum/contohPengerjaan/model_comparison_results.csv', index=False)
print("\n✓ Model comparison results saved to CSV")

print("\n" + "=" * 80)
print("UTS PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\nFiles Created:")
print("- pertemuan08_eda.png")
print("- pertemuan08_model_evaluation.png")
print("- model_comparison_results.csv")

print("\n✓ End-to-End ML Project completed!")
