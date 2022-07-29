#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio
# 
# - Base de Dados: https://drive.google.com/drive/folders/1o2lpxoi9heyQV1hIlsHXWSfDkBPtze-V?usp=sharing

# ### Passo a Passo de um Projeto de Ciência de Dados
# 
# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
# - Passo 7: Interpretação de Resultados

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio
# - TV, Jornal e Rádio estão em milhares de reais
# - Vendas estão em milhões

# #### Importar a Base de dados

# In[18]:


import pandas as pd

tabela = pd.read_csv("advertising.csv")
display(tabela)


# In[ ]:


print(tabela.info())


# In[35]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install scikit-learn')


# #### Análise Exploratória
# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt

# display(tabela.corr())

# criação de gráfico

sns.heatmap(tabela.corr(), cmap="Wistia", annot=True)


# exibir o gráfico

plt.show()


# #### Com isso, podemos partir para a preparação dos dados para treinarmos o Modelo de Machine Learning
# 
# - Separando em dados de treino e dados de teste

# In[47]:


# y -> Quem você quer prever (vendas)
# x -> É o resto todo 

x = tabela[["TV", "Radio", "Jornal"]]
y = tabela["Vendas"]

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)


# #### Temos um problema de regressão - Vamos escolher os modelos que vamos usar:
# 
# - Regressão Linear
# - RandomForest (Árvore de Decisão)

# In[52]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#criação da inteligência
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()


#treino da inteligência
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)


# #### Teste da AI e Avaliação do Melhor Modelo
# 
# - Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece

# In[53]:


# fazer previsão nos teste

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

from sklearn.metrics import r2_score

print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))


# #### Visualização Gráfica das Previsões

# In[54]:


tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["arvore decisao"] = previsao_arvoredecisao
tabela_auxiliar["regressao linear"] = previsao_regressaolinear

plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()


# #### Como fazer uma nova previsão?

# In[60]:


# Árvore de Decisão

novos = pd.read_csv("novos.csv")
display(novos)


# In[61]:


print(modelo_arvoredecisao.predict(novos))


# In[ ]:




