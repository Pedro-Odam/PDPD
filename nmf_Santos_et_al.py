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

# Olhando cada linha do arquivo e verificando se a droga está na Fase I, II, III ou IV.
def determinar_fase(linha):
    for fase in ['Phase IV', 'Phase III', 'Phase II', 'Phase I']:
        if linha.get(fase, '') == "*":
            return fase_pesos[fase]
        # Se a célula dessa fase tiver um * significa que a droga está nessa fase e ele retorna o peso
        # correspondente da biblioteca, ou zero se não se encontrar em alguma fase.
    return 0

# Abaixo está sendo nova criada a coluna chamada "Fase" no nosso arquivo de dados com o valor da fase mais avançada.
df["Fase"] = df.apply(determinar_fase)

# Para garantir que não temos linhas repetidas, como há no arquivo principal, agrupamos as drogas e vírus repetidos
# através do groupby, o .agg está mantendo a fase mais alta caso haja a mesma droga em estágios diferentes.
# Ja o reset.index está resetando as colunas para seus devídos índices (0, 1, 2, 3, 4...).
df_grouped = df.groupby(['Drug', 'Virus']).agg({'Fase': 'max'}).reset_index()

# Aqui abaixo há a criação de duas listas, de todas as drogas e vírus que serão usadas na criação da matriz Y
drogas = df_grouped['Drug']
virus = df_grouped['Virus']

# Agora estamos criando uma matriz de zeros
# Essa matriz tem uma linha para cada droga e uma coluna para cada vírus
Y = np.zeros((len(drogas), len(virus)))

# Agora preenchendo a matriz com os valores das fases
for i, linha in df_grouped.iterrows():
    # Encontrar a posição da droga e do vírus na lista, usamos linha em vez do df por ela ja ser retornada pelo iterrows
    # assim, não acessamos toda a coluna mas a linha, cortando caminho
    droga_idx = np.where(drogas == linha['Drug'])[0][0]
    virus_idx = np.where(virus == linha['Virus'])[0][0]
    # Agora pegando o valor da fase (que já foi calculado antes) e colocar na matriz
    fase = linha["Fase"]
    Y[droga_idx, virus_idx] = fase  # Preenche a célula correta da matriz com o valor da fase

# Transformando a matriz em um df para facilitar a leitura
Y_df = pd.DataFrame(Y, index=drogas, columns=virus)

# Passo 10: Exibir a matriz de associações (drogas e vírus) ponderada pelas fases de desenvolvimento
print("Matriz de Associação entre Drogas e Vírus (ponderada pelas fases):")
print(Y_df)

# Se quiser salvar a matriz em um arquivo CSV (opcional)
Y_df.to_csv("Matriz_Associacao_Ponderada.csv")

