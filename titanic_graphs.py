import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/train.csv")

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# ✅ Graph 1: Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.savefig("graphs/survival_by_gender.png")
plt.show()

# ✅ Graph 2: Survival by Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Class")
plt.savefig("graphs/survival_by_class.png")  # <-- Important
plt.show()
