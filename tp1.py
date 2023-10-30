import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        freq[i] = np.zeros(65536, dtype=np.uint16)
        for j in np.nditer(matrix[:,i]):
            freq[i][j] += 1
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

    freq = np.zeros(65536, dtype=np.uint16)
    for i in np.nditer(values):
        freq[i] += 1

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
    freq = np.zeros(65536, dtype=np.uint16)
    for i in np.nditer(values):
        freq[i] += 1
    freq = freq[freq>0]
    plt.bar(np.arange(freq.size), freq, color='red')
    plt.show()

def main():
    matrix, varNames = ex1("CarDataset.xlsx")
    #ex2(matrix, varNames)
    alphabet, matrix = ex3(matrix)
    freq = ex4(matrix)
    #ex5(varNames,matrix)
    ex6(matrix)

if __name__ == "__main__":
    main()
