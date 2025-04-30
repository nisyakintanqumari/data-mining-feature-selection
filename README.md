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

## 🖼️ Correlation Heatmap

![Correlation Heatmap](results/correlation_heatmap.png)


- `correlation_heatmap.png` - visualization of correlation matrix
- `performance_report.txt` - classification report before & after feature selection
