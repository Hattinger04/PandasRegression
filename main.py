import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('bev_meld.csv', sep=";")
print(df)
# Tirolweit:
df_sum = df.sum()[3:]
print(df_sum)
df_sum = df_sum.astype("int")

df_reg = pd.DataFrame({"years": np.arange(1992,2019), "bev": df_sum})
df_reg = df_reg.astype({'years': 'int'})

model = sm.OLS.from_formula('bev ~ years', df_reg).fit()
df_pred = pd.DataFrame({"years" : [2030]})
predictions = model.predict(df_pred)

plt.plot([2030], predictions, markersize="10", marker="+")

model = sm.OLS.from_formula('bev ~ years', df_reg).fit()
df_pred = pd.DataFrame({"years" : np.arange(2000,2100)})
predictions = model.predict(df_pred)

plt.plot(df_pred.years, predictions)
plt.xlim([2000, 2100])
plt.show()
print(model.params)


# Hatting
df_gemeinde = df.loc[df["Gemeinde"] == "Hatting"]
df_sum = df_gemeinde.sum()[3:]
print(df_sum)
df_sum = df_sum.astype("int")

df_reg = pd.DataFrame({"years": np.arange(1992,2019), "bev": df_sum})
df_reg = df_reg.astype({'years': 'int'})

model = sm.OLS.from_formula('bev ~ years', df_reg).fit()
df_pred = pd.DataFrame({"years" : [2030]})
predictions = model.predict(df_pred)

plt.plot([2030], predictions, markersize="10", marker="+")

model = sm.OLS.from_formula('bev ~ years', df_reg).fit()
df_pred = pd.DataFrame({"years" : np.arange(2000,2100)})
predictions = model.predict(df_pred)

plt.plot(df_pred.years, predictions)
plt.xlim([2000, 2100])
plt.show()
print(model.params)

fig, axes = plt.subplots(2,1)
t1 = df.values[1:24, 3:].sum(axis=0)
d1 = pd.DataFrame({"Bev": t1})
t2 = df.values[140:169, 3:].sum(axis=0)
d2 = pd.DataFrame({"Bev": t2})

d1.plot(ax=axes[0], title="Innsbruck Land")
d2.plot(ax=axes[1], title="Landeck")

fig.tight_layout()
plt.show()

plt.plot(df.columns[3:], d1, label="IL")
plt.plot(df.columns[3:], d2, label="L")
plt.legend()
plt.show()
