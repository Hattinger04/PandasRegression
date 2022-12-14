import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('bev_meld.csv', sep=";")
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