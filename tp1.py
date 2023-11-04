import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import huffmancodec

def ex1(excel,prints=False):
    #read the file
    data = pd.read_excel(excel)
    #matrix with all the data
    matrix = data.values
    if prints:
        print(matrix)
    #list with the names of the variables in the table
    varNames=data.columns.values.tolist()
    if prints:
        print(varNames)
    return matrix, varNames

def ex2(matrix, varNames):
    #represent 6 graphs with the first 6 variables as x and the last one is always the y for every graph
    #in a 3x2 grid
    #set hsapce to 0.5 to avoid overlapping
    #set dots to pink
    fig, axs = plt.subplots(3, 2)
    for i in range(0,6):
        ex7(matrix[:,i])
        ex8(matrix[:,i].size,matrix[:,i])
        axs[i//2,i%2].scatter(matrix[:,i],matrix[:,6],c='purple')
        axs[i//2,i%2].set_title(varNames[i]+" vs "+varNames[6])
        axs[i//2,i%2].set_xlabel(varNames[i])
        axs[i//2,i%2].set_ylabel(varNames[6])
    fig.tight_layout()
    plt.show()

def ex3(matrix,prints=False):
    #convert matrix values to uint16
    matrix = matrix.astype(np.uint16)
    if prints:
        print(matrix)
    #define an alphabet for the uint16
    alphabet = np.zeros(65536, dtype=np.uint16)
    return alphabet, matrix


def ex4(matrix):
    #for every value in the matrix, increment the value in the alphabet
    freq = np.zeros(7, dtype=np.ndarray)
    for i in range(0,7):
        freq[i] = frequencia(matrix[:,i],65536)
    return freq


def ex5(varNames,values):
    #Implementar uma função que permita representar um gráfico de barras mediante o resultado obtido no ponto anterior
    #O resultado para cada variável, deve ser apresentado em figuras individuais
    #Representar no gráfico somente elementos do alfabeto com número de ocorrências não nulo
    freq = ex4(values)
    for i in range(0,7):
        plt.xlabel(varNames[i])
        plt.ylabel("Count")
        #remove the values with 0 occurrences
        freq[i] = freq[i][freq[i]>0]
        plt.bar(np.arange(freq[i].size), freq[i], color='red')
        plt.show()

def ex6(values):
    #Fazer agrupamento de símbolos (binning) para as variáveis Weight, Displacement e Horsepower.
    #Na fonte, símbolos dentro de um intervalo predefinido, deverão assumir todos o mesmo valor.
    #A escolha do símbolo mais representativo para cada intervalo, e que irá substituir todos os elementos do intervalo, deverá ser aquele com maior número de ocorrências.
    #Para a variável Weight, deve considerar o agrupamento de 40 símbolos consecutivos, começando pelo primeiro elemento do alfabeto. Este parâmetro deverá ser inserido como variável de entrada da função.
    #Para as variáveis Displacement e Horsepower deverá considerar um agrupamento de 5 símbolos consecutivos, começando pelo primeiro elemento do alfabeto
    #Uma vez feita a substituição, deverá repetir os pontos 4 e 5, para estas três variáveis.
    #Acceleration	Cylinders	Displacement	Horsepower	ModelYear	Weight	MPG

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
    plotvalues("Weight",weight)
    plotvalues("Displacement",displacement)
    plotvalues("Horsepower",horsepower)

def binning(values,window):

    # for every window of values, find the most frequent value and replace all the values in the window with it
    # example: window number is 3, window of values is [0,1,2]
    # values are [1,0,4,2,1]
    # most frequent value is 1
    # replace all the [0,1,2] to 1
    # result is [1,1,4,1,1]
    # if window number is 40, window of values is [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14, ... , 39]

    freq = frequencia(values,65536)

    new_values = np.zeros(values.size, dtype=np.uint16)

    for i in range(0,values.size,window):
        window_values = values[i:i+window]
        window_freq = freq[window_values]
        most_freq = np.argmax(window_freq)
        new_values[i:i+window] = most_freq
    return new_values

def plotvalues(title,values):
    plt.xlabel(title)
    plt.ylabel("Count")
    freq = frequencia(values,65536)
    freq = freq[freq>0]
    plt.bar(np.arange(freq.size), freq, color='red')
    plt.show()

def ex7(val,prints=True):
    size = int(np.sum(val))                                                    # o tamanho corresponde à soma dos elementos de val
    if size == 0:
        size = 1
    prob = val / size                                                          # probabilidade -> freqência dividida pela soma dos números de dados
    prob=prob[prob>0]
    lim = - np.sum(prob * np.log2(prob))                                     # H(x) = - soma de [p(i) * log2(1/p(i))]
    if prints:
        print("Valor medio teorico: ", lim)
    return lim

def ex8(size, val, prints=True): #não sei se está bem
    freq =frequencia(val,65536)
    prob = freq / size
    prob = prob[prob > 0]
    codec = huffmancodec.HuffmanCodec.from_data(val)
    symbols, lengths = codec.get_code_len()
    media = np.sum(prob * lengths)
    var = np.sum(prob * np.power(lengths - media, 2))
    if prints:
        print("Valor medio dos bits: ", media)
        print("Variancia: ", var)

def ex9(varNames,matrix):
    # Calcular os coeficientes de correlação de Pearson entre a variável MPG e as restantes variáveis.
    # Utilize a função corrcoef do Numpy.
    for i in range(0,7):
        print("Correlação entre MPG e ",varNames[i],": ",np.corrcoef(matrix[:,6],matrix[:,i])[0,1])

def ex10(varNames,matrix):
    # Implemente uma função que permita o cálculo da informação mútua (MI) entre a variável MPG e as restantes variáveis.
    # a. Para as variáveis Weight, Distance e Horsepower considerar os dados após o agrupamento.


    #MPG
    freq_mpg = frequencia(matrix[:,6],65536)
    infompg = ex7(freq_mpg,False)

    for i in range(0,6):
        if i == 2 or i == 3:
            freq = frequencia(binning(matrix[:,i],5),65536)
        elif i == 5:
            freq = frequencia(binning(matrix[:,i],40),65536)
        else:
            freq = frequencia(matrix[:,i],65536)
        infox = ex7(freq,False)
        infoxy = LimMinTeoricoConjunto(matrix[:,i],matrix[:,6])
        #MI = H(x) + H(y) - H(x,y)
        print("Informação mútua entre MPG e ",varNames[i],": ",infox+infompg-infoxy)

def LimMinTeoricoConjunto(val1,val2):
    size = len(val1)
    if size == 0:
        size = 1
    joint_prob = {}

    for i in range(size):
        pair = (val1[i], val2[i])
        if pair in joint_prob:
            joint_prob[pair] += 1
        else:
            joint_prob[pair] = 1

    joint_entropy = 0
    for count in joint_prob.values():
        prob = count / size
        joint_entropy += - prob * np.log2(prob)

    return joint_entropy

def frequencia(val,alfabeto):
    freq = np.zeros(alfabeto, dtype=np.uint16)
    for i in np.nditer(val):
        freq[i] += 1
    return freq

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
    matrix, varNames = ex1("CarDataset.xlsx")
    #ex2(matrix, varNames)
    #alphabet, matrix = ex3(matrix)
    #freq = ex4(matrix)
    #ex5(varNames,matrix)
    #ex6(matrix)
    #ex7(matrix.flatten())
    # for i in range(0,7):
    #     ex8(matrix[:,i].size,matrix[:,i])
    #ex9(varNames,matrix)
    #ex10(varNames,matrix)
    #ex11(matrix,True)


if __name__ == "__main__":
    main()
