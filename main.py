
---

### 🐍 `main.py`

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Correlation matrix
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('results/correlation_matrix.png')

# Remove highly correlated features (example threshold = 0.9)
threshold = 0.9
to_drop = [column for column in corr.columns if any(corr[column] > threshold) and column != 'target']
df_reduced = df.drop(columns=to_drop)

# Train/test split
X = df_reduced.drop('target', axis=1)
y = df_reduced['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
report = classification_report(y_test, y_pred)
print(report)

with open('results/performance_metrics.txt', 'w') as f:
    f.write(report)
