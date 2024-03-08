import numpy as np
import polars as ps
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib as mpl
import cycler
import statsmodels.api as sm
sb.color_palette("rocket")
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color',
                                            plt.cm.Dark2(np.linspace(0, 1, 8)))


#data = ps.read_csv("fuel_cpi-44.csv",has_header=False)
#data = data.rename({"column_1": "fuel"})
#data = data.rename({"column_2": "cpi"})
#fig, ax = plt.subplots(1,3);
#sb.histplot(data, x='fuel',label='fuel price',ax=ax[0])
#sb.histplot(data, x='cpi',label='consumer price index',ax=ax[1])
#sb.scatterplot(data, x='fuel',y='cpi',ax=ax[2])
#ax[2].set_xlabel("fuel_price")
#ax[2].set_ylabel("consumer_price_index")
#fig.tight_layout()
#plt.show()

data = ps.read_csv("walmart.csv")#,has_header=False)
fig, ax = plt.subplots(1,4);
sb.histplot(data, x='Fuel_Price',label='fuel price',ax=ax[0])
sb.histplot(data, x='CPI',label='consumer price index',ax=ax[1])

model1 = sm.OLS(data["Fuel_Price"].to_numpy(), data["CPI"].to_numpy()).fit()
model2 = sm.OLS(data["CPI"].to_numpy(), data["Fuel_Price"].to_numpy()).fit()
print(model1.summary())
print(model2.summary())

sb.scatterplot(data, x='Fuel_Price',y='CPI',ax=ax[2])
sb.scatterplot(data, x='CPI',y='Fuel_Price',ax=ax[3])




ax[2].set_xlabel("fuel_price")
ax[2].set_ylabel("consumer_price_index")
ax[3].set_ylabel("fuel_price")
ax[3].set_xlabel("consumer_price_index")
fig.tight_layout()
plt.show()

