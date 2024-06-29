import pandas as pd

df = pd.read_csv("DrugVirus.csv", encoding="latin1")

print("Primeiras cinco linhas do Data Frame: ")
print(df.head())
