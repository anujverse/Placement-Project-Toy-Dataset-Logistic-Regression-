# Student Placement Prediction using Logistic Regression

## 📌 Project Overview

This project predicts whether a student is likely to be placed based on two features:

- **CGPA**
- **IQ**

The project demonstrates a complete basic Machine Learning workflow, including data preprocessing, train-test splitting, feature scaling, model training, prediction, and visualization of decision regions.

## 🎯 Objective

The objective of this project is to build a **binary classification model** that predicts student placement status.

- `0` → Not Placed
- `1` → Placed

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Mlxtend
- Jupyter Notebook / Google Colab

## 📊 Dataset

The dataset contains student information with the following features:

| Feature | Description |
|---|---|
| `cgpa` | Student's CGPA |
| `iq` | Student's IQ score |
| `placement` | Placement status (0 or 1) |

An unnecessary `Unnamed: 0` column was removed during preprocessing.

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Decision Region Visualization
```

## ⚙️ Data Preprocessing

The unnecessary index column was removed:

```python
df = df.drop("Unnamed: 0", axis=1)
```

The independent and dependent variables were separated:

```python
X = df[['cgpa', 'iq']]
y = df['placement']
```

The dataset was then divided into training and testing sets.

## 📏 Feature Scaling

Since CGPA and IQ have different numerical ranges, `StandardScaler` was used to standardize the input features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Standardization transforms the features based on their mean and standard deviation.

## 🤖 Model

The project uses **Logistic Regression** for binary classification.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
```

The trained model learns the relationship between CGPA, IQ, and placement status.

## 📈 Prediction

After training, the model can predict placement status for unseen test data:

```python
y_pred = model.predict(X_test)
```

## 📊 Decision Region Visualization

The decision boundary can be visualized using `mlxtend`:

```python
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(
    X=X_train,
    y=y_train,
    clf=model,
    legend=2
)
```

This visualization shows how the Logistic Regression model separates the two classes.

## 📁 Project Structure

```text
placement-prediction/
│
├── placement_prediction.ipynb
├── README.md
├── requirements.txt
└── data/
    └── placement.csv
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Install the required libraries

```bash
pip install pandas numpy scikit-learn matplotlib mlxtend jupyter
```

### 3. Open the notebook

```bash
jupyter notebook placement_prediction.ipynb
```

Alternatively, the notebook can be opened directly in **Google Colab**.

## 📌 Key Concepts Demonstrated

- Data Cleaning
- Feature Selection
- Train-Test Split
- Feature Scaling
- StandardScaler
- Logistic Regression
- Binary Classification
- Model Prediction
- Decision Boundary / Decision Regions
- Basic Model Evaluation

## 🔮 Future Improvements

Possible improvements for this project include:

- Adding more student-related features
- Comparing Logistic Regression with other classification algorithms
- Using cross-validation
- Adding a confusion matrix
- Calculating precision, recall, and F1-score
- Improving model performance through hyperparameter tuning
- Deploying the model as a web application

## 👨‍💻 Author

**Anuj Yadav**

BCA Student | Aspiring Data Scientist / Data Analyst
