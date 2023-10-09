import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ex1():
    #read the file
    data = pd.read_excel("C:\\Users\\gugaa\\Desktop\\TI\\IT-PROJECT\\CarDataset.xlsx") #change the path to the file
    #matrix with all the data
    matrix = data.values
    print(matrix)
    #list with the names of the variables in the table
    varNames=data.columns.values.tolist()
    print(varNames)
    return matrix, varNames
    
def ex2():
    matrix, varNames = ex1()
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
    
def ex3():
    matrix, varNames = ex1()
    df = pd.DataFrame(matrix)
    #convert to uint16
    df = df.astype(np.uint16)
    print(df)

def ex4():
    # # Inicializa um dicionário para armazenar as contagens de ocorrências
    # counts = {}

    # # Itera sobre cada coluna do DataFrame
    # for column in dataframe.columns:
    #     # Verifica se a coluna está no alfabeto
    #     if column in alphabet:
    #         # Conta as ocorrências de cada símbolo na coluna
    #         symbol_counts = dataframe[column].value_counts().to_dict()
    #         counts[column] = symbol_counts

    # return counts

    

def main():
    # ex1()
    # ex2()
    ex3()

if __name__ == "__main__":
    main()
