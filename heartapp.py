import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.impute             import SimpleImputer
from sklearn.feature_selection  import VarianceThreshold
from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics            import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def load_and_clean(path):
    df = pd.read_csv(path)

    # 1. Rename and drop extras
    df.rename(columns={'num': 'target'}, inplace=True)
    df.drop(columns=['id', 'dataset'], inplace=True)

    # 2. Map binary text → numeric
    df['sex']   = df['sex'].map({'Female': 0, 'Male': 1})
    df['fbs']   = df['fbs'].astype(float)
    df['exang'] = df['exang'].astype(float)

    # 3. Impute missing values
    num_feats = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    df[num_feats] = SimpleImputer(strategy='median').fit_transform(df[num_feats])
    df[['fbs', 'exang']] = SimpleImputer(strategy='most_frequent')\
                                 .fit_transform(df[['fbs', 'exang']])

    # 4. One-hot encode all remaining categoricals
    df = pd.get_dummies(df, drop_first=True)

    # 5. Convert to pure binary: 0 = no disease, 1 = any disease
    df['target'] = (df['target'] > 0).astype(int)

    return df

def main():
    # Load & clean
    df = load_and_clean('heart_disease_uci (1).csv')

    # --- Exploratory Visualizations ---

    # Cholesterol vs. Age (by decade)
    df['age_decade'] = (df['age'] // 10) * 10
    chol_by_age = df.groupby('age_decade')['chol'].mean()

    plt.figure()
    chol_by_age.plot(kind='bar')
    plt.title('Average Cholesterol by Age Decade')
    plt.xlabel('Age Decade')
    plt.ylabel('Average Cholesterol')
    plt.tight_layout()
    plt.show()

    # Chest Pain Type vs Disease Rate
    raw_df = pd.read_csv('heart_disease_uci (1).csv')
    cp_rate = raw_df.groupby('cp')['num'].apply(lambda x: (x > 0).mean())

    plt.figure()
    cp_rate.plot(kind='bar')
    plt.title('Heart Disease Rate by Chest Pain Type')
    plt.xlabel('Chest Pain Type')
    plt.ylabel('Disease Rate')
    plt.tight_layout()
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap of Features")
    plt.tight_layout()
    plt.show()

    # Split features & target
    X = df.drop('target', axis=1)
    y = df['target']

    # Drop zero-variance features
    selector = VarianceThreshold(threshold=0.0)
    X = selector.fit_transform(X)
    feature_names = selector.get_feature_names_out(df.drop(columns=['target']).columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Define models (replacing Decision Tree with Gradient Boosting)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Gradient Boosting':   GradientBoostingClassifier(),
        'Random Forest':       RandomForestClassifier(n_estimators=100)
    }

    # Train & evaluate
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds    = m.predict(X_test)
        prob_pos = m.predict_proba(X_test)[:, 1]

        results[name] = {
            'Accuracy':       accuracy_score(y_test, preds),
            'Precision':      precision_score(y_test, preds),
            'Recall':         recall_score(y_test, preds),
            'ROC AUC':        roc_auc_score(y_test, prob_pos),
            'Confusion Mat.': confusion_matrix(y_test, preds),
            'ROC Curve':      roc_curve(y_test, prob_pos)
        }

    # Print metrics
    for name, m in results.items():
        print(f"\n=== {name} ===")
        print(f"Accuracy : {m['Accuracy']:.3f}")
        print(f"Precision: {m['Precision']:.3f}")
        print(f"Recall   : {m['Recall']:.3f}")
        print(f"ROC AUC  : {m['ROC AUC']:.3f}")
        print("Confusion Matrix:")
        print(m['Confusion Mat.'])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for name, m in results.items():
        fpr, tpr, _ = m['ROC Curve']
        plt.plot(fpr, tpr, label=f"{name} (AUC={m['ROC AUC']:.3f})")

    plt.plot([0, 1], [0, 1], '--', label='Chance', color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Feature Importance Plot for Random Forest ---
    rf_model = models['Random Forest']
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        top_indices = importances.argsort()[::-1][:10]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_features, palette='viridis')
        plt.title('Top 10 Feature Importances (Random Forest)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    # --- Model Performance Comparison (Metrics on X-axis, Models as Bars) ---
    metric_order = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']

    for model in results:
        prec = results[model]['Precision']
        rec  = results[model]['Recall']
        results[model]['F1'] = 2 * (prec * rec) / (prec + rec)

    model_names = list(results.keys())
    data = {metric.lower(): [results[model][metric] for model in model_names]
            for metric in metric_order}

    df_metrics = pd.DataFrame(data, index=model_names).T

    plt.figure(figsize=(8, 6))
    df_metrics.plot(kind='bar', width=0.8)
    plt.title('Model Performance Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=0)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()

    # --- Confusion Matrix for Random Forest ---
    rf_cm = results['Random Forest']['Confusion Mat.']

    plt.figure(figsize=(6, 5))
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    # --- Cross-Validation Scores (ROC AUC ± Std) ---
    labels = []
    means = []
    stds = []

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        labels.append(name)
        means.append(scores.mean())
        stds.append(scores.std())

    plt.figure(figsize=(8, 6))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel('ROC AUC')
    plt.ylim(0.5, 1.0)
    plt.title('Cross-Validation Scores (Mean ± Std)')
    plt.tight_layout()
    plt.show()

    # --- Probability Distribution Plot - Random Forest ---
    rf_probs = models['Random Forest'].predict_proba(X_test)[:, 1]

    plt.figure(figsize=(10, 6))
    sns.histplot(rf_probs[y_test == 0], color='blue', label='No Disease', bins=20, stat='density')
    sns.histplot(rf_probs[y_test == 1], color='orange', label='Disease', bins=20, stat='density')
    plt.title('Prediction Probability Distribution - Random Forest')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
