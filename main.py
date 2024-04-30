import numpy as np
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt

NORM_LEARNING = 0.3
my_file = open("Intelligent_information_security_technologies.txt", "w")
def activation_function(net):
    if net >= 0:
        return 1
    return 0

def Сalculation_phi(x_values, neuron_arr):
    phi=0
    for i in range(len(neuron_arr)):
        phi+=(x_values[i]-neuron_arr[i])*(x_values[i]-neuron_arr[i])*(-1)
    return np.exp(phi)

def calculation_net(weights,x_values,center_neurons_arrays):
    net = 0
    for i in range(len(center_neurons_arrays)):
        phi = Сalculation_phi(x_values, center_neurons_arrays[i])
        net += weights[i+1] * phi
    net += weights[0]
    return net

def find_neuron_centers(matrix,func_vector):
    fal=0
    tru=0
    vic=0
    neuron_centers=[]
    for i in range(len(func_vector)):
        if func_vector[i]==0:
            fal=fal+1
        else:
            tru=tru+1
    if fal<=tru:
        vic=0

    else:
        vic=1
    for i in range(len(func_vector)):
        if func_vector[i]==vic:
            neuron_centers.append(matrix[i])
    return neuron_centers

def reset_model_states(model):
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()

def learning_process(matrix, func_vector, vector_learning, sample_learning, n=NORM_LEARNING, eralim=False):

    neuron_centers = find_neuron_centers(matrix, func_vector)
    weights = np.ones(len(neuron_centers) + 1)
    data = {'Номер эпохи': [], 'Вектор весов w': [], 'Выходной вектор y': [], 'Суммарная ошибка Е': []}
    generation = 0
    prev_weights = weights.copy()

    while True:
        generation_s_weights = weights.copy()
        error = 0
        generation_s_y = []

        for i in range(len(sample_learning)):
            net = calculation_net(weights, vector_learning[i], neuron_centers)
            y = activation_function(net)
            error += sample_learning[i] - y
            phi_array = [1] + [Сalculation_phi(vector_learning[i], neuro_i) for neuro_i in neuron_centers]
            delta = n * error * np.array(phi_array)
            weights += delta
        # error = 0 # 1
        # generation_s_y = list()
        for i in range(len(func_vector)):
            net = calculation_net(weights, matrix[i], neuron_centers)
            y = activation_function(net)
            generation_s_y.append(y)
            if func_vector[i] != generation_s_y[i]:
                error += 1

        data['Номер эпохи'].append(generation)
        data['Вектор весов w'].append(np.round(generation_s_weights, 3))
        data['Выходной вектор y'].append(generation_s_y)
        data['Суммарная ошибка Е'].append(error)

        if error == 0 or (eralim and eralim - 1 == 0) or all(x == y for x, y in zip(generation_s_y, func_vector)):
            break
        # Проверка изменения весов
        if np.allclose(prev_weights, weights, atol=1e-5):
            break

        prev_weights = weights.copy()

        generation += 1
        if eralim:
            eralim -= 1

    return data, error == 0

# Вместо того, чтобы возвращать результат после первой успешной итерации,
# продолжим поиск лучшего набора данных до конца всех комбинаций и вернуть лучший результат

def find_less_process(matrix, func_vector, n=NORM_LEARNING, lim=20):
    sample = list()
    sample_data = None
    vector_y = []
    flag = False

    for index in range(2, len(matrix)+1):
        all_combinations = list(combinations(matrix, index))
        print('\nПроверка набора длины ' + str(index))
        my_file.write(f"\nПроверка набора длины: {str(index)}\n");
        for subset in all_combinations:
            vector_y = [func_vector[matrix.index(sub)] for sub in subset]
            data, flag = learning_process(matrix, func_vector, subset, vector_y, n, lim)
            if flag and (sample_data is None or data['Суммарная ошибка Е'][-1] < sample_data['Суммарная ошибка Е'][-1]):
                sample = subset
                sample_data = data
                my_file.write(f"Найден лучший минимальный набор: {str(sample)}\n");
                print(f"Найден лучший минимальный набор: {str(sample)}")

                # return sample, sample_data # первый лучший вариант

    return sample, sample_data

matrix =[[ 0, 0, 0, 0 ],
    [ 0, 0, 0, 1 ],
    [ 0, 0, 1, 0 ],
    [ 0, 0, 1, 1 ],
    [ 0, 1, 0, 0 ],
    [ 0, 1, 0, 1 ],
    [ 0, 1, 1, 0,],
    [ 0, 1, 1, 1,],
    [ 1, 0, 0, 0,],
    [ 1, 0, 0, 1,],
    [ 1, 0, 1, 0,],
    [ 1, 0, 1, 1,],
    [ 1, 1, 0, 0,],
    [ 1, 1, 0, 1,],
    [ 1, 1, 1, 0,],
    [ 1, 1, 1, 1,]]

func_vector = [ 0,0,0,1, 1,1,1,1, 0,0,0,0, 0,1,1,1 ]

from itertools import product

def evaluate_expression(X1, X2, X3, X4):
    return (not X1 or X3) and X2 or X2 and X4

func = []
def truth_table():
    variables = ['X1', 'X2', 'X3', 'X4']
    table = []

    for assignment in product([False, True], repeat=len(variables)):
        values = dict(zip(variables, assignment))
        result = evaluate_expression(**values)
        func.append(int(result))
        row = [int(values[var]) for var in variables] + [int(result)]
        table.append(row)
    return table


result_table = truth_table()
my_file.write("[x1,x2,x3,x4,f]\n");
print("[x1,x2,x3,x4,f]")

for row in result_table:
    my_file.write(str(row)+'\n');
    print(str(row))

func_vector = func

print('[f(x1,x2,x3)]\n', func_vector)
my_file.write("\n[Полный набор]\n");
print('\n[Полный набор] ')
data = learning_process(matrix,
                        func_vector,
                        matrix,
                        func_vector)[0]
print(pd.DataFrame(data).to_string())
my_file.write(pd.DataFrame(data).to_string());
 # data = {'Номер эпохи': [], 'Вектор весов w': [], 'Выходной вектор y': [], 'Суммарная ошибка Е': []}
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.plot(data['Номер эпохи'], data['Суммарная ошибка Е'], marker='.', color='red')
plt.plot(data['Номер эпохи'], data['Суммарная ошибка Е'], marker='.', color='darkred', markerfacecolor='white')
plt.title('Cуммарная ошибка НС \nна всей выборке')
plt.xlabel('Номер эпохи')
plt.ylabel('Суммарная ошибка Е')
plt.grid()

# Поиск минимального набора
sample, sample_data = find_less_process(matrix,func_vector)

print('\n[Минимальный набор]\n' + str(sample))
print(pd.DataFrame(sample_data).to_string())
my_file.write('\n[Минимальный набор]\n' + str(sample));
my_file.write(pd.DataFrame(sample_data).to_string());
my_file.close()

plt.subplot(2, 2, 2)
plt.plot(sample_data['Номер эпохи'], sample_data['Суммарная ошибка Е'], marker='.', color='darkred', markerfacecolor='white')
plt.title('Cуммарная ошибка НС \nна мин выборке ')
plt.xlabel('Номер эпохи\n')
plt.ylabel('Суммарная ошибка Е')
plt.grid()
plt.savefig('Intelligent_information_security_technologies.png')
plt.show()
plt.close()
