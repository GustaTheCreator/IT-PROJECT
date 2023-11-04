import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from huffmancodec import *
from utils import *

def ex1(excel, prints=False):
    data = pd.read_excel(excel)
    matrix = data.values
    if prints:
        print(matrix)
    names=data.columns.values.tolist()
    if prints:
        print(names)
    return matrix, names

def ex2(matrix, names):
    # Representar graficamente os valores das variáveis MPG, Acceleration, Cylinders, Displacement, Horsepower e Weight,

    fig, axs = plt.subplots(3, 2)
    for i in range(0,6):
        ex7(matrix[:,i])
        ex8(matrix[:,i].size,matrix[:,i])
        axs[i//2,i%2].scatter(matrix[:,i],matrix[:,6],c='purple')
        axs[i//2,i%2].set_title(names[i]+" vs "+names[6])
        axs[i//2,i%2].set_xlabel(names[i])
        axs[i//2,i%2].set_ylabel(names[6])
    fig.tight_layout()
    plt.show()

def ex3(matrix, prints=False):
    # Converte matriz e alfabeto para uint16

    matrix = matrix.astype(np.uint16)
    if prints:
        print(matrix)
    alphabet = np.zeros(65536, dtype=np.uint16)
    return alphabet, matrix


def ex4(matrix, prints=False):
    # Para cada valor da matriz, incrementar o valor no alfabeto

    freq = np.zeros(7, dtype=np.ndarray)
    for i in range(0,7):
        freq[i] = frequency(matrix[:,i],65536)
    if prints:
        print(freq)
    return freq

def ex5(names, values):
    # Implementar uma função que permita representar um gráfico de barras mediante o resultado obtido no ponto anterior
    # O resultado para cada variável, deve ser apresentado em figuras individuais
    # Representar no gráfico somente elementos do alfabeto com número de ocorrências não nulo

    freq = ex4(values)
    for i in range(0,7):
        plt.xlabel(names[i])
        plt.ylabel("Count")
        # Remover valores sem ocorrências
        freq[i] = freq[i][freq[i]>0]
        plt.bar(np.arange(freq[i].size), freq[i], color='red')
        plt.show()

def ex6(values):
    # Fazer agrupamento de símbolos (binning) para as variáveis Weight, Displacement e Horsepower.
    # Na fonte, símbolos dentro de um intervalo predefinido, deverão assumir todos o mesmo valor.
    # A escolha do símbolo mais representativo para cada intervalo, e que irá substituir todos os
    # elementos do intervalo, deverá ser aquele com maior número de ocorrências.
    # Para a variável Weight, deve considerar o agrupamento de 40 símbolos consecutivos, começando pelo primeiro
    # elemento do alfabeto. Este parâmetro deverá ser inserido como variável de entrada da função.
    # Para as variáveis Displacement e Horsepower deverá considerar um agrupamento de 5 símbolos
    # consecutivos, começando pelo primeiro elemento do alfabeto
    # Uma vez feita a substituição, deverá repetir os pontos 4 e 5, para estas três variáveis.
    # Acceleration	Cylinders	Displacement	Horsepower	ModelYear	Weight	MPG

    #Weight
    weight = binning(values[:,5],40)

    #Displacement
    displacement = binning(values[:,2],5)

    #Horsepower
    horsepower = binning(values[:,3],5)
    ex7(weight)
    ex7(displacement)
    ex7(horsepower)
    ex8(weight.size,weight)
    ex8(displacement.size,displacement)
    ex8(horsepower.size,horsepower)
    plot_values("Weight",weight)
    plot_values("Displacement",displacement)
    plot_values("Horsepower",horsepower)

def ex7(val, prints=True):
    size = int(np.sum(val))
    if size == 0:
        size = 1
    prob = val / size
    prob=prob[prob>0]
    lim = - np.sum(prob * np.log2(prob))
    if prints:
        print("Valor medio teorico: ", lim)
    return lim

def ex8(size, val, prints=True):
    freq = frequency(val,65536)
    prob = freq / size
    prob = prob[prob > 0]
    codec = HuffmanCodec.from_data(val)
    _, lengths = codec.get_code_len()
    media = np.sum(prob * lengths)
    var = np.sum(prob * np.power(lengths - media, 2))
    if prints:
        print("Valor medio dos bits: ", media)
        print("Variancia: ", var)

def ex9(names, matrix):
    # Calcular os coeficientes de correlação de Pearson entre a variável MPG e as restantes variáveis.
    # Utilize a função corrcoef do Numpy.

    for i in range(0,7):
        print(f"Correlação entre MPG e {names[i]}:", np.corrcoef(matrix[:,6],matrix[:,i])[0,1])

def ex10(names, matrix):
    # Implemente uma função que permita o cálculo da informação mútua (MI) entre a variável MPG e as restantes variáveis.

    #MPG
    freq_mpg = frequency(matrix[:,6],65536)
    infompg = ex7(freq_mpg,False)

    for i in range(0,6):
        if i == 2 or i == 3:
            freq = frequency(binning(matrix[:,i],5),65536)
        elif i == 5:
            freq = frequency(binning(matrix[:,i],40),65536)
        else:
            freq = frequency(matrix[:,i],65536)
        infox = ex7(freq,False)
        infoxy = min_join_entropy(matrix[:,i],matrix[:,6])
        print(f"Informação mútua entre MPG e {names[i]}:",infox+infompg-infoxy)

def ex11(matrix, prints=False):
    # Os valores de MPG podem ser estimados em função das restantes variáveis,
    # utilizando uma relação (simples) como a apresentada a seguir:

    # 𝑀𝑃𝐺𝑝𝑟𝑒𝑑 =−5.5241−0.146∗𝐴𝑐𝑐𝑒𝑙𝑒𝑟𝑎𝑡𝑖𝑜𝑛−0.4909∗𝐶𝑦𝑙𝑖𝑛𝑑𝑒𝑟𝑠
    # +0.0026∗𝐷𝑖𝑠𝑡𝑎𝑛𝑐𝑒−0.0045∗𝐻𝑜𝑟𝑠𝑒𝑝𝑜𝑤𝑒𝑟+0.6725
    # ∗𝑀𝑜𝑑𝑒𝑙−0.0059∗𝑊𝑒𝑖𝑔ℎ𝑡

    mpg = -5.5241 - 0.146*matrix[:,0] - 0.4909*matrix[:,1] + 0.0026*matrix[:,2] - 0.0045*matrix[:,3] + 0.6725*matrix[:,4] - 0.0059*matrix[:,5]
    if prints:
        print(mpg)

    mpg = -5.5241 - 0.146*matrix[:,0] - 0.4909*matrix[:,1] + 0.0026*matrix[:,2] - 0.0045*matrix[:,3] + 0.6725*matrix[:,4]
    if prints:
        print(mpg)

    mpg = -5.5241 - 0.146*matrix[:,0] - 0.4909*matrix[:,1] + 0.0026*matrix[:,2] - 0.0045*matrix[:,3] - 0.0059*matrix[:,5]
    if prints:
        print(mpg)

def main():
    matrix, names = ex1("CarDataset.xlsx")
    ex2(matrix, names)
    _, matrix = ex3(matrix)
    ex4(matrix)
    ex5(names, matrix)
    ex6(matrix)
    ex7(matrix.flatten())
    for i in range(0,7):
        ex8(matrix[:,i].size, matrix[:,i])
        continue
    ex9(names, matrix)
    ex10(names, matrix)
    ex11(matrix, True)

if __name__ == "__main__":
    main()
