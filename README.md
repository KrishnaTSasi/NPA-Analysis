# NPA-Analysis


## ** NPA-Analysis-Using-Python**

### **1. Project Overview**

This project analyzes **Non-Performing Asset (NPA)** loan data from ESAF Bank (Dharwad Cluster) to identify NPA-prone customer profiles, visualize patterns and build a basic prediction model using Python.
Explored relationships between loan attributes such as **overdue amount, delinquency days, income, disbursed amount and current balance** and used regression and classification techniques to detect risk.


### **2. Folder Structure**

```
NPA-Analysis-Using-Python/
│
├── data/
│   ├── npa_dataset.csv            # Raw dataset
│   ├── cleaned_npa_dataset.csv    # Preprocessed data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Data loading, cleaning, and EDA
│   ├── 02_visualization.ipynb     # Histograms, boxplots, scatter plots
│   ├── 03_regression_analysis.ipynb # Regression between loan variables
│   ├── 04_npa_prediction_model.ipynb # Logistic Regression / Random Forest model
│
├── scripts/
│   ├── data_cleaning.py           # Functions for handling missing values & formatting
│   ├── visualization.py           # Plotly/Matplotlib visualizations
│   ├── regression_model.py        # Regression model training and evaluation
│   ├── classification_model.py    # NPA prediction model
│
├── results/
│   ├── eda_plots/                  # Exported plots (histograms, boxplots, scatter plots)
│   ├── regression_results.csv      # Regression outputs
│   ├── classification_report.txt   # Accuracy, confusion matrix
│
├── requirements.txt               # List of Python dependencies
├── README.md                      # Project description, usage instructions
└── LICENSE                        # License information
```


### **3. Installation**

```bash
# Clone the repository
git clone https://github.com/yourusername/NPA-Analysis-Using-Python.git
cd NPA-Analysis-Using-Python

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
```


### **4. Usage**

Run **Jupyter Notebook** for step-by-step analysis:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Or run the Python scripts:

```bash
python scripts/data_cleaning.py
python scripts/visualization.py
python scripts/regression_model.py
python scripts/classification_model.py
```


### **5. Example Code Snippets**

**Data Cleaning (`data_cleaning.py`)**

```python
import pandas as pd

def clean_npa_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['OVERDUE_AMT', 'ESAF_NEW_DELIQUENCY_DAYS', 'INCOME'])
    df['OVERDUE_AMT'] = df['OVERDUE_AMT'].astype(float)
    df['INCOME'] = df['INCOME'].astype(float)
    return df

if __name__ == "__main__":
    cleaned_df = clean_npa_data("data/npa_dataset.csv")
    cleaned_df.to_csv("data/cleaned_npa_dataset.csv", index=False)
    print("Data cleaned and saved!")
```

**Visualization (`visualization.py`)**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned_npa_dataset.csv")

# Boxplot for Overdue Amount
plt.figure(figsize=(8,5))
sns.boxplot(x=df['OVERDUE_AMT'])
plt.title("Boxplot of Overdue Amount")
plt.savefig("results/eda_plots/boxplot_overdue_amt.png")
plt.show()
```

**Regression Model (`regression_model.py`)**

```python
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("data/cleaned_npa_dataset.csv")
X = df[['OVERDUE_AMT', 'INCOME']]
y = df['DISBURSED_AMT']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
```

**Classification Model (`classification_model.py`)**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/cleaned_npa_dataset.csv")
df['NPA_Flag'] = (df['OVERDUE_AMT'] > 0).astype(int)

X = df[['OVERDUE_AMT', 'ESAF_NEW_DELIQUENCY_DAYS', 'INCOME']]
y = df['NPA_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```


### **6. Outputs**

* **EDA Plots**: Histograms, boxplots, scatter plots
* **Regression Output**: Relationship between overdue amount, income and loan disbursement
* **Classification Output**: NPA prediction accuracy and confusion matrix



