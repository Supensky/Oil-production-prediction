import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 1 эквивалент тонны нефти = 11630 киловатт-час
# Запасы нефти на земле = 244,6 млрд т
# Тераватт-час (TWh) = 10**9 киловатт-час

df = pd.read_csv('oil-production-by-country.csv')
data = df.groupby('Year')['Oil production (TWh)'].sum()
data = data.apply(lambda x: round(x))
x = data.index.values.reshape(-1, 1)
y = data.values.reshape(-1, 1)

fig = plt.figure(figsize=(6,6))
fig.suptitle('Oil consumption')
ax = fig.add_subplot(111)
ax.set_xlabel('Year')
ax.set_ylabel('TWh')
ax.grid()
ax.plot(x, y);
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=42)

# Pred 1
model1 = LinearRegression()
model1.fit(x_train, y_train)
x_new = np.arange(2024, 2101).reshape(-1, 1)
yp = model1.predict(x_new)
model1.score(x_test, y_test)

fig = plt.figure(figsize=(6,6))
fig.suptitle('Oil consumption')
ax = fig.add_subplot(111)
ax.set_xlabel('Year')
ax.set_ylabel('TWh')
ax.plot(x, y, 'b', label='True')
ax.plot(x_new, yp, 'r', label='Predicted')
ax.grid()
ax.legend();

# Pred 2
# Пробуем обучить модель на данных после 1995, так как во время распада
# различных государств добыча исчезнувшего гос-ва и нового, появившегося вместо
# него, считаются дважды, что делает данные недостоверными в некоторые года.
data2 = data[data.index.values >= 1995]
x2 = data2.index.values.reshape(-1, 1)
y2 = data2.values.reshape(-1, 1)

fig = plt.figure(figsize=(6,6))
fig.suptitle('Oil consumption')
ax = fig.add_subplot(111)
ax.set_xlabel('Year')
ax.set_ylabel('TWh')
ax.grid()
ax.plot(x2, y2);

x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2,
                                                    random_state=42)
model2 = LinearRegression()
model2.fit(x_train, y_train)
yp2 = model2.predict(x_new)
model2.score(x_test, y_test)

fig = plt.figure(figsize=(6,6))
fig.suptitle('Oil consumption')
ax = fig.add_subplot(111)
ax.set_xlabel('Year')
ax.set_ylabel('TWh')
ax.plot(x, y, 'b', label='True')
ax.plot(x_new, yp2, 'r', label='Predicted')
ax.grid()
ax.legend();

# Pred 3
# Попробуем предсказать добычу нефти при помощи аппроксимации.
x = x.reshape(1, -1)[0]
y = y.reshape(1, -1)[0]

# Выбираем оптимальный коэффициент k
fig = plt.figure(figsize=(12, 12))
t1 = fig.add_subplot(221)
t1.set_title('Approximation')
t1.set_xlabel('Year')
t1.set_ylabel('TWh')
t1.plot(x, y, label='True')
t1.grid()

t2 = fig.add_subplot(222)
t2.set_title('MSE')
t2.set_xlabel('Polynomial degree')
t2.set_ylabel('MSE')
t2.grid()
for k in range(2, 7):
    p = np.polyfit(x, y, k)
    py = np.polyval(p, x)
    mse = mean_squared_error(py, y)
    t1.plot(x, py, label=f'k={k}')
    t2.scatter(k, mse)
t1.legend()
# 5 степень полинома наиболее оптимальная

p = np.polyfit(x, y, 5)
py = np.polyval(p, x_new)

fig = plt.figure(figsize=(6,6))
fig.suptitle('Oil consumption')
ax = fig.add_subplot(111)
ax.set_xlabel('Year')
ax.set_ylabel('TWh')
ax.plot(x, y, 'b', label='True')
ax.plot(x_new, py, 'r', label='Predicted')
ax.grid()
ax.legend();

# Обучим модель на данных после 1995
x_new1 = np.arange(2023, 2040)
x2 = x2.reshape(1, -1)[0]
y2 = y2.reshape(1, -1)[0]
p2 = np.polyfit(x2, y2, 5)
py2 = np.polyval(p2, x_new1)

fig = plt.figure(figsize=(6,6))
fig.suptitle('Oil consumption')
ax = fig.add_subplot(111)
ax.set_xlabel('Year')
ax.set_ylabel('TWh')
ax.plot(x, y, 'b', label='True')
ax.plot(x_new1, py2, 'r', label='Predicted')
ax.grid()
ax.legend();
# Этот прогноз будем считать недействительным из-за слишком резкой прямой.
# Тогда оставшиеся три прогноза будем считать за благоприятный, нормальный и 
# неблагоприятный.
fig = plt.figure(figsize=(6,6))
fig.suptitle('Oil consumption')
ax = fig.add_subplot(111)
ax.set_xlabel('Year')
ax.set_ylabel('TWh')
ax.plot(x, y, 'b', label='True')
ax.plot(x_new, yp, label='Predicted 1')
ax.plot(x_new, yp2, label='Predicted 2')
ax.plot(x_new, py, label='Predicted 3')
ax.grid()
ax.legend();

# Расчитаем количество оставшихся ресурсов на каждый год после 2023.
# 1 эквивалент тонны нефти = 11630 киловатт-час
# Запасы нефти на земле = 244,6 млрд т
# Тераватт-час (TWh) = 10**9 киловатт-час

# Pred 1
res = 244.6e9
res_1 = yp * 1e9 / 116305
r_1 = np.array([res])
for i in res_1:
    r_1 = np.append(r_1, r_1[-1] - i)
r_1 = r_1[r_1 > 0]
r_1 = np.append(r_1, 0)


fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(221)
ax1.set_title('Oil consumption')
ax1.set_xlabel('Year')
ax1.set_ylabel('TWh')
ax1.grid()
ax1.plot(np.arange(2023, 2023 + len(r_1)), yp[:len(r_1)])

ax = fig.add_subplot(222)
ax.set_title('Oil reserves')
ax.set_xlabel('Year')
ax.set_ylabel('Tons')
ax.plot(np.arange(2023, 2023 + len(r_1)), r_1)
ax.scatter(2023 + len(r_1) - 1, 0, label=2023 + len(r_1) - 1, marker='*')
ax.grid()
ax.legend();

# Pred 2
res_2 = yp2 * 1e9 / 116305
r_2 = np.array([res])
for i in res_2:
    r_2 = np.append(r_2, r_2[-1] - i)
r_2 = r_2[r_2 > 0]
r_2 = np.append(r_2, 0)


fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(221)
ax1.set_title('Oil consumption')
ax1.set_xlabel('Year')
ax1.set_ylabel('TWh')
ax1.grid()
ax1.plot(np.arange(2023, 2023 + len(r_2)), yp2[:len(r_2)])

ax = fig.add_subplot(222)
ax.set_title('Oil reserves')
ax.set_xlabel('Year')
ax.set_ylabel('Tons')
ax.plot(np.arange(2023, 2023 + len(r_2)), r_2)
ax.scatter(2023 + len(r_2) - 1, 0, label=2023 + len(r_2) - 1, marker='*')
ax.grid()
ax.legend();

# Pred 3
res_3 = py * 1e9 / 116305
r_3 = np.array([res])
for i in res_3:
    r_3 = np.append(r_3, r_3[-1] - i)
r_3 = r_3[r_3 > 0]
r_3 = np.append(r_3, 0)


fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(221)
ax1.set_title('Oil consumption')
ax1.set_xlabel('Year')
ax1.set_ylabel('TWh')
ax1.grid()
ax1.plot(np.arange(2023, 2023 + len(r_3)), py[:len(r_3)])

ax = fig.add_subplot(222)
ax.set_title('Oil reserves')
ax.set_xlabel('Year')
ax.set_ylabel('Tons')
ax.plot(np.arange(2023, 2023 + len(r_3)), r_3)
ax.scatter(2023 + len(r_3) - 1, 0, label=2023 + len(r_3) - 1, marker='*')
ax.grid()
ax.legend();

# Итог
fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(221)
ax1.set_title('Oil consumption')
ax1.set_xlabel('Year')
ax1.set_ylabel('TWh')
ax1.grid()

ax1.plot(x, y)
ax1.plot(np.arange(2023, 2023 + len(r_1)), yp[:len(r_1)], 'r', label='Negative')
ax1.plot(np.arange(2023, 2023 + len(r_2)), yp2[:len(r_2)], 'g', \
         label='Positive')
ax1.plot(np.arange(2023, 2023 + len(r_3)), py[:len(r_3)], 'y', label='Normal')

ax = fig.add_subplot(222)
ax.set_title('Oil reserves')
ax.set_xlabel('Year')
ax.set_ylabel('Tons')

ax.plot(np.arange(2023, 2023 + len(r_1)), r_1, color='r', label='Negative')
ax.scatter(2023 + len(r_1) - 1, 0, label=2023 + len(r_1) - 1, marker='*', \
           color='r')

ax.plot(np.arange(2023, 2023 + len(r_2)), r_2, color='g', label='Positive')
ax.scatter(2023 + len(r_2) - 1, 0, label=2023 + len(r_2) - 1, marker='*', \
           color='g')

ax.plot(np.arange(2023, 2023 + len(r_3)), r_3, color='y', label='Normal')
ax.scatter(2023 + len(r_3) - 1, 0, label=2023 + len(r_3) - 1, marker='*', \
           color='y')

ax.grid()
ax1.legend()
ax.legend();
