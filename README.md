# MLC1
# ============================================
# Machine Learning Lab - Data Cleaning & Charts
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("data.csv")

# ============================================
# View Dataset
# ============================================

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== LAST 5 ROWS =====")
print(df.tail())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns)

print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIPTION =====")
print(df.describe())

# ============================================
# Check Missing Values
# ============================================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ============================================
# Fill Missing Values
# ============================================

# Fill numerical column with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill categorical column with mode
df['City'] = df['City'].fillna(df['City'].mode()[0])

# ============================================
# Remove Duplicate Rows
# ============================================

print("\n===== DUPLICATES =====")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

# ============================================
# Rename Column
# ============================================

df.rename(columns={'Marks': 'Student_Marks'}, inplace=True)

# ============================================
# Change Data Type
# ============================================

df['Age'] = df['Age'].astype(int)

# ============================================
# Remove Column
# ============================================

# Uncomment if needed
# df.drop('Name', axis=1, inplace=True)

# ============================================
# Filter Data
# ============================================

print("\n===== FILTER AGE > 20 =====")
print(df[df['Age'] > 20])

# ============================================
# Sort Values
# ============================================

print("\n===== SORT BY AGE =====")
print(df.sort_values(by='Age'))

# ============================================
# Unique Values
# ============================================

print("\n===== UNIQUE CITIES =====")
print(df['City'].unique())

print("\n===== NUMBER OF UNIQUE CITIES =====")
print(df['City'].nunique())

# ============================================
# Group By
# ============================================

print("\n===== GROUP BY CITY =====")
print(df.groupby('City')['Student_Marks'].mean())

# ============================================
# Correlation
# ============================================

print("\n===== CORRELATION =====")
print(df.corr(numeric_only=True))

# ============================================
# LINE CHART
# ============================================

plt.figure(figsize=(6,4))
plt.plot(df['Age'])

plt.title("Line Chart - Age")
plt.xlabel("Index")
plt.ylabel("Age")

plt.show()

# ============================================
# BAR CHART
# ============================================

plt.figure(figsize=(6,4))
plt.bar(df['Name'], df['Student_Marks'])

plt.title("Bar Chart - Student Marks")
plt.xlabel("Student Name")
plt.ylabel("Marks")

plt.show()

# ============================================
# HISTOGRAM
# ============================================

plt.figure(figsize=(6,4))
plt.hist(df['Age'])

plt.title("Histogram - Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

# ============================================
# PIE CHART
# ============================================

values = [30, 40, 30]
labels = ['Python', 'Java', 'C++']

plt.figure(figsize=(6,4))
plt.pie(values, labels=labels, autopct='%1.1f%%')

plt.title("Pie Chart - Programming Languages")

plt.show()

# ============================================
# SCATTER PLOT
# ============================================

plt.figure(figsize=(6,4))
plt.scatter(df['Age'], df['Student_Marks'])

plt.title("Scatter Plot")
plt.xlabel("Age")
plt.ylabel("Marks")

plt.show()

# ============================================
# BOX PLOT
# ============================================

plt.figure(figsize=(6,4))
plt.boxplot(df['Student_Marks'])

plt.title("Box Plot - Marks")

plt.show()

# ============================================
# Save Cleaned Dataset
# ============================================

df.to_csv("cleaned_data.csv", index=False)

print("\n===== CLEANED DATA SAVED SUCCESSFULLY =====")
