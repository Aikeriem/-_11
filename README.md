import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Создаем случайные данные
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Визуализация данных
plt.scatter(X, y, alpha=0.7)
plt.title('Сгенерированные данные')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Используем линейную регрессию
model = LinearRegression()
model.fit(X, y)

# Получаем коэффициенты модели
slope = model.coef_[0]
intercept = model.intercept_

# Визуализация регрессионной линии
plt.scatter(X, y, alpha=0.7)
plt.plot(X, model.predict(X), color='red', linewidth=2, label=f'Линейная регрессия: y = {slope[0]:.2f}x + {intercept[0]:.2f}')
plt.title('Линейная регрессия')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Создаем данные для графика
x_values = np.linspace(-2*np.pi, 2*np.pi, 100)  # 100 точек от -2*pi до 2*pi
y_values = np.sin(x_values)

# Создаем новый график
plt.figure(figsize=(8, 6))  # Размер графика

# Строим график
plt.plot(x_values, y_values, label='sin(x)', color='blue', linewidth=2)

# Добавляем заголовок и метки осей
plt.title('График синусоиды')
plt.xlabel('x')
plt.ylabel('sin(x)')

# Добавляем сетку
plt.grid(True)

# Добавляем легенду
plt.legend()

# Показываем график
plt.show()
