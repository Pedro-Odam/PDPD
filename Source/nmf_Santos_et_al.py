import pandas as pd
import numpy as np

# Carregar o arquivo CSV.
df = pd.read_csv("DrugVirus (1).csv", encoding="Windows-1252")

# Criando uma tabela que diz o peso da fase de cada vírus.
fase_pesos = {
    "Phase I": 0.25,
    "Phase II": 0.5,
    "Phase III": 0.75,
    "Phase IV": 1.0
}

# Função para determinar o peso com base na fase específica
def determinar_fase(linha, fase_especifica):
    if linha.get(fase_especifica, '') == "*":
        return fase_pesos[fase_especifica]
    return 0

# Agrupar as drogas e vírus repetidos, mantendo a fase mais alta caso haja a mesma droga em estágios diferentes.
df_grouped = df.groupby(['Drug', 'Virus']).first().reset_index()

# Criar listas de todas as drogas e vírus que serão usadas na criação da matriz Y
drogas = df_grouped['Drug']
virus = df_grouped['Virus']

# Criar as matrizes de zeros (Máscaras), os pesos serão adicionados mais abaixo no código
M_A = np.zeros((len(drogas), len(virus)))  # Matriz para Phase IV
M_B = np.zeros((len(drogas), len(virus)))  # Matriz para Phase III
M_C = np.zeros((len(drogas), len(virus)))  # Matriz para Phase II
M_D = np.zeros((len(drogas), len(virus)))  # Matriz para Phase I

# Criar uma matriz de zeros
Y = np.zeros((len(drogas), len(virus)))

# Preencher a matriz com os valores das fases
for i, linha in df_grouped.iterrows():
    droga_idx = np.where(drogas == linha['Drug'])[0][0]
    virus_idx = np.where(virus == linha['Virus'])[0][0]

    # Colocar 1 na matriz para indicar a associação da droga com o vírus
    Y[droga_idx, virus_idx] = 1

    # Preencher as matrizes com o peso apenas se a droga estiver na fase correta
    M_A[droga_idx, virus_idx] = determinar_fase(linha, 'Phase IV')
    M_B[droga_idx, virus_idx] = determinar_fase(linha, 'Phase III')
    M_C[droga_idx, virus_idx] = determinar_fase(linha, 'Phase II')
    M_D[droga_idx, virus_idx] = determinar_fase(linha, 'Phase I')

# Transformar as matrizes em DataFrames para facilitar a visualização
M_A_df = pd.DataFrame(M_A, index=drogas, columns=virus)
M_B_df = pd.DataFrame(M_B, index=drogas, columns=virus)
M_C_df = pd.DataFrame(M_C, index=drogas, columns=virus)
M_D_df = pd.DataFrame(M_D, index=drogas, columns=virus)

# Exibir as matrizes de cada fase
print("Matriz M_A (Phase IV):")
print(M_A_df)

print("Matriz M_B (Phase III):")
print(M_B_df)

print("Matriz M_C (Phase II):")
print(M_C_df)

print("Matriz M_D (Phase I):")
print(M_D_df)

# Transformar a matriz em um DataFrame para facilitar a leitura
Y_df = pd.DataFrame(Y, index=drogas, columns=virus)

# Exibir a matriz de associações (drogas e vírus) ponderada pelas fases de desenvolvimento
print("Matriz de Associação entre Drogas e Vírus (ponderada pelas fases):")
print(Y_df)

# Se quiser salvar a matriz em um arquivo CSV
Y_df.to_csv("Matriz_Associacao_Ponderada.csv")

