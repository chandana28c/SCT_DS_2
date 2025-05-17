import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load dataset
titanic = pd.read_csv('titanic.csv')

# --- Data Cleaning ---
print("Initial Info:")
print(titanic.info())
print("\nMissing values:")
print(titanic.isnull().sum())

# Safe filling (avoid chained assignment warning)
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])

# Drop cabin (too many missing values)
titanic = titanic.drop(columns='Cabin')

# Convert to category
titanic['Sex'] = titanic['Sex'].astype('category')
titanic['Embarked'] = titanic['Embarked'].astype('category')

# Create output folder
os.makedirs('charts', exist_ok=True)

# --- EDA Charts ---

# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=titanic)
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.savefig('charts/1_survival_count.png')
plt.close()

# 2. Survival by Gender
plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=titanic)
plt.title('Survival Rate by Gender')
plt.savefig('charts/2_survival_by_gender.png')
plt.close()

# 3. Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=titanic)
plt.title('Survival Rate by Passenger Class')
plt.savefig('charts/3_survival_by_pclass.png')
plt.close()

# 4. Age Distribution by Survival
plt.figure(figsize=(8,5))
sns.kdeplot(titanic.loc[titanic['Survived'] == 1, 'Age'], label='Survived', fill=True)
sns.kdeplot(titanic.loc[titanic['Survived'] == 0, 'Age'], label='Did Not Survive', fill=True)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.legend()
plt.savefig('charts/4_age_distribution_by_survival.png')
plt.close()

# 5. Fare Distribution by Survival
plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Fare', data=titanic)
plt.title('Fare Paid by Survival')
plt.savefig('charts/5_fare_by_survival.png')
plt.close()

# 6. Embarkation vs Survival
plt.figure(figsize=(6,4))
sns.barplot(x='Embarked', y='Survived', data=titanic)
plt.title('Survival Rate by Embarkation Port')
plt.savefig('charts/6_survival_by_embarked.png')
plt.close()

# 7. Correlation Heatmap (only numeric columns)
numeric_titanic = titanic.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8,6))
sns.heatmap(numeric_titanic.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('charts/7_correlation_matrix.png')
plt.close()

# --- Summary ---
print(f"\n✔️ All charts saved to 'charts' folder.")
print(f"✔️ Overall survival rate: {titanic['Survived'].mean():.2%}")
