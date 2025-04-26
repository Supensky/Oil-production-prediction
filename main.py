import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("oil-production-by-country.csv")
data = df.groupby('Year')['Oil production (TWh)'].sum()
data = data.apply(lambda x: round(x))
x = data.index.values
y = data.values

# 1 эквивалент тонны нефти = 11630 киловатт-час
# запасы нефти на земле = 244,6 млрд т
# Тераватт-час (TWh) = 10**9 киловатт-час

fig = plt.figure(figsize=(6,6))
fig.suptitle('Oil consumption')
ax = fig.add_subplot(111)
ax.grid()
ax.plot(x, y)
