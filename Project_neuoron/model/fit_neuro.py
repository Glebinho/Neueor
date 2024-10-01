import numpy as np
from neuron import SingleNeuron


# Пример данных (X - входные данные, y - целевые значения)
X = np.array([[41, 90,9],
              [39, 80,8],
              [36, 60,2],
              [39, 95,8],
              [36, 65,3],
              [35, 60,2],
              [41, 90,9]])
y = np.array([1, 1, 0, 1, 0, 0, 1])  # Ожидаемый выход
# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=3)
neuron.train(X, y, epochs=5000, learning_rate=0.001)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')