import pandas as pd
from scipy.stats import f_oneway, ttest_ind, kruskal

df = pd.read_csv("mental_health_dataset_corregido.csv")
label = "gender"  # o employment_status
variable = "stress_level"

print(df[label].value_counts())
group1 = df[df[label] == df[label].unique()[0]][variable]
group2 = df[df[label] == df[label].unique()[1]][variable]

t_stat, p_val = ttest_ind(group1, group2)
print("T-test:")
print("Estadístico:", t_stat)
print("Valor p:", p_val)


# Agrupar todos los valores por grupo
groups = [group[variable].dropna() for name, group in df.groupby(label)]

#ANOVA
f_stat, p_anova = f_oneway(*groups)
print("\nANOVA:")
print("Estadístico F:", f_stat)
print("Valor p:", p_anova)

#Kruskal-Wallis
h_stat, p_kw = kruskal(*groups)
print("\nKruskal-Wallis:")
print("Estadístico H:", h_stat)
print("Valor p:", p_kw)
