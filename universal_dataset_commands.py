# ============================================================
#   UNIVERSAL DATASET COMMANDS
#   Works on ANY dataset — just change filename & target col
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer


# ============================================================
# ✅ CHANGE ONLY THESE 2 LINES — REST WORKS AUTOMATICALLY
# ============================================================

df = pd.read_csv("your_file.csv")   # 👈 change filename
TARGET = "target"                   # 👈 change to your label column name


# ============================================================
# STEP 1: EXPLORE — works on any dataset
# ============================================================

print("Shape       :", df.shape)
print("Columns     :", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates  :", df.duplicated().sum())
print("\nStatistics:\n", df.describe())
print("\nTarget Distribution:\n", df[TARGET].value_counts())


# ============================================================
# STEP 2: CLEAN — auto handles any dataset
# ============================================================

# Remove duplicates
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Auto-detect numeric and categorical columns
num_cols = df.select_dtypes(include=['number']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove target from feature lists if present
if TARGET in num_cols: num_cols.remove(TARGET)
if TARGET in cat_cols: cat_cols.remove(TARGET)

print("Numeric columns    :", num_cols)
print("Categorical columns:", cat_cols)

# Fill missing — numeric with median, categorical with mode
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing after clean:", df.isnull().sum().sum())  # should be 0


# ============================================================
# STEP 3: ENCODE — auto label-encodes all categorical columns
# ============================================================

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Encode target if it's text (e.g., 'Yes'/'No', 'cat'/'dog')
if df[TARGET].dtype == 'object':
    df[TARGET] = le.fit_transform(df[TARGET].astype(str))


# ============================================================
# STEP 4: SPLIT — auto separates X and y
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size :", X_test.shape)


# ============================================================
# STEP 5: TRAIN DECISION TREE
# ============================================================

model = DecisionTreeClassifier(
    criterion='gini',    # try 'entropy' too
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)


# ============================================================
# STEP 6: EVALUATE
# ============================================================

y_pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("CV Scores:", cv_scores)
print("CV Mean  :", round(cv_scores.mean(), 4))


# ============================================================
# STEP 7: VISUALIZE TREE & FEATURE IMPORTANCE
# ============================================================

# Tree Plot
plt.figure(figsize=(14, 6))
plot_tree(model,
          feature_names=X.columns.tolist(),
          class_names=[str(c) for c in sorted(y.unique())],
          filled=True, rounded=True)
plt.title("Decision Tree")
plt.tight_layout()
plt.show()

# Text Tree
print(export_text(model, feature_names=X.columns.tolist()))

# Feature Importance Bar Chart
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar', color='steelblue')
plt.title("Feature Importance")
plt.ylabel("Score")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 8: PREDICT NEW SAMPLE (auto-sized)
# ============================================================

# Creates a sample row of all zeros — replace values as needed
sample = pd.DataFrame([np.zeros(X.shape[1])], columns=X.columns)
print("\nPredicted Class      :", model.predict(sample)[0])
print("Prediction Probability:", model.predict_proba(sample)[0])


# ============================================================
# QUICK REFERENCE — COMMON COMMANDS
# ============================================================

# df.shape                          → rows, cols
# df.head() / df.tail()             → first/last 5 rows
# df.info()                         → dtypes + non-null count
# df.describe()                     → stats (mean, std, min, max)
# df.isnull().sum()                 → null count per column
# df.duplicated().sum()             → total duplicate rows
# df['col'].value_counts()          → frequency of each value
# df.select_dtypes(include='number')→ only numeric columns
# df.select_dtypes(include='object')→ only text columns
# df.drop(columns=['col'])          → remove a column
# df.rename(columns={'a':'b'})      → rename column
# df.dropna()                       → drop rows with any null
# df.fillna(value)                  → fill nulls with value
# df.drop_duplicates()              → remove duplicate rows
# df.reset_index(drop=True)         → reset row index

# ============================================================
