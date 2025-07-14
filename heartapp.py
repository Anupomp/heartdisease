import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute             import SimpleImputer
from sklearn.feature_selection  import VarianceThreshold
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier
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

    # Split features & target
    X = df.drop('target', axis=1)
    y = df['target']

    # Drop zero-variance features
    X = VarianceThreshold(threshold=0.0).fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree':      DecisionTreeClassifier(),
        'Random Forest':      RandomForestClassifier(n_estimators=100)
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
        plt.plot(fpr, tpr, label=f"{name} (AUC={m['ROC AUC']:.2f})")

    plt.plot([0, 1], [0, 1], '--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
