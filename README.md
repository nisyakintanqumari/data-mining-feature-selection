# Data Mining - Feature Selection using Correlation Matrix

This project explores feature selection methods using a correlation matrix to improve machine learning model performance in both classification and prediction tasks. It was developed as part of a Master's research thesis at NTUST (National Taiwan University of Science and Technology).

---

## 🔍 Objectives
- Identify multicollinearity among features using correlation matrix
- Select optimal subset of features to reduce model complexity
- Enhance classification and forecasting accuracy
- Provide visualization of feature relationships

---

## 🧰 Tools & Libraries
- Python 3.x
- pandas, numpy
- seaborn, matplotlib
- scikit-learn

---

## 📁 Folder Structure
```
FeatureSelection_CorrMatrix/
├── data/
│   └── dataset.csv
├── models/
│   └── trained_model.pkl
├── results/
│   ├── correlation_heatmap.png
│   └── performance_report.txt
├── main.py
├── feature_selector.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/FeatureSelection_CorrMatrix.git
cd FeatureSelection_CorrMatrix
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the script:
```bash
python main.py
```

---

## 📈 Sample Output
- `correlation_heatmap.png` - visualization of correlation matrix
- `performance_report.txt` - classification report before & after feature selection

---

## 🧠 main.py
```python
from feature_selector import select_features
from utils import load_data, evaluate_model

if __name__ == "__main__":
    df = load_data("data/dataset.csv")
    reduced_df = select_features(df, threshold=0.85)
    evaluate_model(df, reduced_df)
```

---

## 🧪 feature_selector.py
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def select_features(df, threshold=0.85):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(~pd.np.tril(pd.np.ones(corr_matrix.shape)).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Save heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig("results/correlation_heatmap.png")

    print(f"Dropped features: {to_drop}")
    return df.drop(columns=to_drop)
```

---

## 🧰 utils.py
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def evaluate_model(original_df, reduced_df):
    target = original_df.columns[-1]  # Assuming last column is target

    X_orig = original_df.drop(columns=target)
    X_red = reduced_df.drop(columns=target)
    y = original_df[target]

    for name, X in zip(["Original", "Reduced"], [X_orig, X_red]):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred)
        with open(f"results/performance_report_{name.lower()}.txt", "w") as f:
            f.write(report)
        print(f"Performance Report ({name}):\n{report}\n")

        # Save model
        joblib.dump(clf, f"models/{name.lower()}_model.pkl")
```

---

## 📎 requirements.txt
```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```
