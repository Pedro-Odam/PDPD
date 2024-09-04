import numpy.matlib
from numpy import linalg as LA
import pandas as pd
import numpy as np

def AlgoritmoDecomposicao(Y, k, alfas, mascaras):
    max_iteracoes = 2000    
    tolx = 1e-4   
    epsilon = np.finfo(float).eps
    sqrt_eps = np.sqrt(epsilon)
    variancia = 0.01 
    
    (n_medicamentos, n_virus) = Y.shape
       
    P0 = np.random.uniform(0, np.sqrt(variancia), (n_medicamentos, k))
    Q0 = np.random.uniform(0, np.sqrt(variancia), (k, n_virus)) 
    
    Q0 = np.divide(Q0, np.matlib.tile(np.array([np.sqrt(np.sum(np.power(Q0, 2), 1))]).transpose(), (1, n_virus)))
    
    J = list()
    for iteracao in range(max_iteracoes):
        numerador = 0
        denominador = 0
        
        for fase in mascaras.keys():
            numerador += alfas[fase] * np.multiply(mascaras[fase], Y)
            denominador += alfas[fase] * np.multiply(mascaras[fase], np.dot(P0, Q0))
            
        numerador = np.dot(numerador, Q0.transpose())
        denominador = np.dot(denominador, Q0.transpose()) + np.spacing(numerador)
        
        P = np.maximum(0, np.multiply(P0, np.divide(numerador, denominador)))
        P.clip(min=0)

        numerador = 0
        denominador = 0
        
        for fase in mascaras.keys():
            numerador += alfas[fase] * np.multiply(mascaras[fase], Y)
            denominador += alfas[fase] * np.multiply(mascaras[fase], np.dot(P, Q0))
            
        numerador = np.dot(P.transpose(), numerador)
        denominador = np.dot(P.transpose(), denominador) + np.spacing(numerador)
        
        Q = np.maximum(0, np.multiply(Q0, np.divide(numerador, denominador)))
        Q.clip(min=0)

        perda = 0
        for fase in mascaras.keys():
            perda += 0.5 * alfas[fase] * LA.norm(np.multiply(mascaras[fase], (Y - np.dot(P, Q))), 'fro')**2
        
        J.append(perda)

        dp = np.amax(np.abs(P - P0)) / (sqrt_eps + np.amax(np.abs(P0)))
        dq = np.amax(np.abs(Q - Q0)) / (sqrt_eps + np.amax(np.abs(Q0)))
        delta = np.maximum(dp, dq)
      
        if iteracao > 1 and delta <= tolx:
            print(f'Convergido na iteração {iteracao} com delta {delta}')
            break

        P0 = P
        Q0 = np.divide(Q, np.matlib.tile(np.array([np.sqrt(np.sum(np.power(Q0, 2), 1))]).transpose(), (1, n_virus)))

    return [P, Q, J]

# Carregamento e preparação dos dados
caminho_arquivo = 'caminho/para/DrugVirus.csv'  # Substitua pelo caminho correto do arquivo CSV
dados = pd.read_csv(caminho_arquivo, encoding='latin1')

fases = ['Fase I', 'Fase II', 'Fase III']
Y = dados.pivot_table(index='Medicamento', columns='Vírus', aggfunc='size', fill_value=0)

mascaras = {fase: (dados.pivot_table(index='Medicamento', columns='Vírus', values=fase, aggfunc=lambda x: 1 if x.notna().any() else 0, fill_value=0) > 0).astype(int) for fase in fases}
alfas = {fase: 1.0 / (idx + 1) for idx, fase in enumerate(fases)}

# Parâmetros do algoritmo de decomposição
k = 10  # Número de fatores latentes

# Execução do algoritmo de decomposição
P, Q, J = AlgoritmoDecomposicao(Y.values, k, alfas, mascaras)

# Exibição dos resultados
print("Matriz P:", P)
print("Matriz Q:", Q)
print("Último valor da função de custo:", J[-1])

