import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv("tp_final/MapBiomas-Tabela_de_Dados.csv")

# Transpor para formato longo (Ano e Desmatamento como variáveis)
data_long = data.melt(id_vars=["Classe"], 
                      var_name="Ano", 
                      value_name="Desmatamento")

# Converter Ano e Desmatamento para tipos adequados
data_long["Ano"] = pd.to_numeric(data_long["Ano"])
data_long["Desmatamento"] = pd.to_numeric(data_long["Desmatamento"], errors="coerce")

# Agrupar os dados por ano e somar os valores de desmatamento
dados_agrupados = data_long.groupby('Ano')['Desmatamento'].sum().reset_index()

# Dividir os dados em treino (até 2020) e teste (2021–2023)
dados_treino = dados_agrupados[dados_agrupados['Ano'] <= 2020]
dados_teste = dados_agrupados[(dados_agrupados['Ano'] >= 2021) & (dados_agrupados['Ano'] <= 2023)]

# Configurar o dashboard
st.title("Previsão de Desmatamento com ARIMA")

# Slider para ajustar os parâmetros do ARIMA
st.sidebar.header("Parâmetros do Modelo ARIMA")
p = st.sidebar.slider("Ordem AR (p)", 0, 5, 1)
d = st.sidebar.slider("Diferenciação (d)", 0, 2, 1)
q = st.sidebar.slider("Ordem MA (q)", 0, 5, 1)

# Treinar o modelo ARIMA com os parâmetros selecionados
modelo_arima = ARIMA(dados_treino['Desmatamento'], order=(p, d, q))
resultado_arima = modelo_arima.fit()

# Fazer previsões para 2021–2023
previsoes = resultado_arima.forecast(steps=3).values

# Calcular o erro
valores_reais = dados_teste['Desmatamento'].values
erro_mse = mean_squared_error(valores_reais, previsoes)
erro_mae = mean_absolute_error(valores_reais, previsoes)

# Mostrar os resultados
st.subheader("Resultados da Previsão")
st.write(f"Previsões para 2021–2023:\n {previsoes}")
st.write(f"Valores Reais:\n {valores_reais}")
st.write(f"Erro MSE: {erro_mse:.2f}\n Erro MAE: {erro_mae:.2f}")

# Gráfico
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dados_agrupados['Ano'], dados_agrupados['Desmatamento'], label='Dados Históricos', marker='o')
ax.plot(dados_teste['Ano'], previsoes, label='Previsões', marker='x', color='red')
ax.set_title("Previsão de Desmatamento com ARIMA")
ax.set_xlabel("Ano")
ax.set_ylabel("Desmatamento")
ax.legend()
ax.grid(True)
st.pyplot(fig)

###################################################################################################

data2 = pd.read_csv("tp_final/SEEG.csv")

# Transpor para formato longo (Ano e Emissao como variáveis)
data_long2 = data2.melt(id_vars=["Categoria"], 
                      var_name="Ano", 
                      value_name="Emissao")

# Converter Ano e Desmatamento para tipos adequados
data_long2["Ano"] = pd.to_numeric(data_long2["Ano"])
data_long2["Emissao"] = pd.to_numeric(data_long2["Emissao"], errors="coerce")

# Agrupar os dados por ano e somar os valores de emissao
dados_agrupados2 = data_long2.groupby('Ano')['Emissao'].sum().reset_index()

# Dividir os dados em treino (até 2020) e teste (2021–2023)
dados_treino2 = dados_agrupados2[dados_agrupados2['Ano'] <= 2020]
dados_teste2 = dados_agrupados2[(dados_agrupados2['Ano'] >= 2021) & (dados_agrupados2['Ano'] <= 2023)]

# Configurar o dashboard
st.title("\nPrevisão de Emissão de Gases com ARIMA")

# Treinar o modelo ARIMA com os parâmetros selecionados
modelo_arima2 = ARIMA(dados_treino2['Emissao'], order=(p, d, q))
resultado_arima2 = modelo_arima2.fit()

# Fazer previsões para 2021–2023
previsoes2 = resultado_arima2.forecast(steps=3).values

# Calcular o erro
valores_reais2 = dados_teste2['Emissao'].values
erro_mse2 = mean_squared_error(valores_reais2, previsoes2)
erro_mae2 = mean_absolute_error(valores_reais2, previsoes2)

# Mostrar os resultados
st.subheader("Resultados da Previsão")
st.write(f"Previsões para 2021–2023:\n {previsoes2}")
st.write(f"Valores Reais:\n {valores_reais2}")
st.write(f"Erro MSE: {erro_mse2:.2f}\n Erro MAE: {erro_mae2:.2f}")

# Gráfico
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dados_agrupados2['Ano'], dados_agrupados2['Emissao'], label='Dados Históricos', marker='o')
ax.plot(dados_teste2['Ano'], previsoes2, label='Previsões', marker='x', color='red')
ax.set_title("Previsão de Emissão de Gases com ARIMA")
ax.set_xlabel("Ano")
ax.set_ylabel("Emissao")
ax.legend()
ax.grid(True)
st.pyplot(fig)
