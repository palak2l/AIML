import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    'Age': [20, 22, 25, None, 30, 35, 40],
    'Salary': [20000, 25000, 30000, 35000, None, 50000, 60000],
    'City': ['Pune', 'Mumbai', 'Delhi', 'Pune',
             'Delhi', 'Mumbai', 'Pune']
})

print("Missing Values:\n", df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

df = pd.get_dummies(df, columns=['City'], dtype=int)

print("Statistics:\n", df.describe())
print("Correlation:\n", df.corr())

fig, ax = plt.subplots(1, 3, figsize=(13, 4))

ax[0].hist(df['Age'], bins=5)
ax[0].set_title("Age Histogram")

ax[1].boxplot(df['Salary'])
ax[1].set_title("Salary Boxplot")

corr = df.corr()
im = ax[2].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax[2].set_xticks(range(len(corr)), corr.columns, rotation=90)
ax[2].set_yticks(range(len(corr)), corr.columns)
ax[2].set_title("Correlation Heatmap")
fig.colorbar(im, ax=ax[2])

plt.tight_layout()
plt.show()

df[['Age', 'Salary']] = StandardScaler().fit_transform(
    df[['Age', 'Salary']]
)
print("Processed Data:\n", df)