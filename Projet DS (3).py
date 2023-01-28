#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

prod_vege_2018 = pd.read_csv(r"C:\Users\Hassan\Downloads\2018_donneesgrandescultures.csv", sep=';', header = [0,1])
prod_vege_2018.iloc[:,1:] = prod_vege_2018.iloc[:,1:].astype(float)
prod_vege_2018.rename({'Unnamed: 0_level_0':''}, axis=1, inplace = True)
prod_vege_2018.iloc[:,1:] = prod_vege_2018.iloc[:,1:].round(2)

prod_vege_2018.fillna(0, inplace = True)
prod_vege_2018 = prod_vege_2018.sort_values(by=('', 'Cultures'))
prod_vege_2018.reset_index(drop=True, inplace=True)
prod_vege_2018.insert(0, "Année", '2018')
prod_vege_2018['Année'] = prod_vege_2018['Année'].apply(lambda x: pd.to_datetime(str(x),format='%Y%'))

prod_vege_2018.head()


# In[2]:


prod_vege_2018.isna().sum()


# In[3]:


prod_vege_2019 = pd.read_csv(r"C:\Users\Hassan\Downloads\2019_donneesgrandescultures.csv", sep=';', header = [0,1])

prod_vege_2019.iloc[:,1:] = prod_vege_2019.iloc[:,1:].astype(float)
prod_vege_2019.rename({'Unnamed: 0_level_0':''}, axis=1, inplace = True)
prod_vege_2019.iloc[:,1:] = prod_vege_2019.iloc[:,1:].round(2)

prod_vege_2019.info()

prod_vege_2019.fillna(0, inplace=True)
prod_vege_2019 = prod_vege_2019.sort_values(by=('', 'Cultures'))
prod_vege_2019.reset_index(drop=True, inplace=True)
prod_vege_2019.insert(0, "Année", '2019')
prod_vege_2019['Année'] =prod_vege_2019['Année'].apply(lambda x: pd.to_datetime(str(x),format='%Y%'))
prod_vege_2019.head()


# In[4]:


prod_vege_2020 = pd.read_csv(r"C:\Users\Hassan\Downloads\2020_donneesgrandescultures.csv", sep=';', header=[0,1])

prod_vege_2020.head()

prod_vege_2020.iloc[:,1:] = prod_vege_2020.iloc[:,1:].astype(float)
prod_vege_2020.rename({'Unnamed: 0_level_0':''}, axis=1, inplace = True)
prod_vege_2020.iloc[:,1:] = prod_vege_2020.iloc[:,1:].round(2)

prod_vege_2020.info()

prod_vege_2020.fillna(0, inplace=True)

prod_vege_2020 = prod_vege_2020.sort_values(by=('', 'Cultures'))
prod_vege_2020.reset_index(drop=True, inplace=True)
prod_vege_2020.insert(0, "Année", '2020')
prod_vege_2020['Année'] =prod_vege_2020['Année'].apply(lambda x: pd.to_datetime(str(x),format='%Y%'))
prod_vege_2020.head()


# In[5]:


prod_vege_2021 = pd.read_csv(r"C:\Users\Hassan\Downloads\2021_donneesgrandescultures (1).csv", sep=';', header=[0,1])

prod_vege_2021.iloc[:,1:] = prod_vege_2021.iloc[:,1:].astype(float)
prod_vege_2021.rename({'Unnamed: 0_level_0':''}, axis=1, inplace = True)
prod_vege_2021.iloc[:,1:] = prod_vege_2021.iloc[:,1:].round(2)

prod_vege_2021.info()

prod_vege_2021.fillna(0, inplace=True)

prod_vege_2021.replace({'Rendement(q/h)': 'Rendement(q/ha)'}, inplace=True)

prod_vege_2021 = prod_vege_2021.sort_values(by=('', 'Cultures'))
prod_vege_2021.reset_index(drop=True, inplace=True)
prod_vege_2021.insert(0, "Année", '2021')
prod_vege_2021['Année'] =prod_vege_2021['Année'].apply(lambda x: pd.to_datetime(str(x),format='%Y%'))
prod_vege_2021.head()


# In[6]:


fig, ax = plt.subplots(13, figsize=(30,40))
barWidth = 0.4
x1 = np.arange(len(prod_vege_2018[('','Cultures')]))
x2 = x1 + 0.4
ax[0].bar(x1, prod_vege_2018.loc[:,('France métropolitaine', 'Production(1000 t)')], width = barWidth, label = 'France 2018')
ax[0].bar(x2, prod_vege_2019.loc[:,('France métropolitaine', 'Production(1000 t)')],width = barWidth, label = 'France 2019')
ax[0].text(2, 700, 'Pas de valeur de production des betteraves pour 2018')
ax[0].set_title('Productions grandes cultures 2018 et 2019 France métropolitaine')
ax[0].legend();

ax[1].bar(x1, prod_vege_2018.loc[:,('Occitanie','Production(1000 t)') ], width = barWidth, label = 'Occitanie 2018') 
ax[1].bar(x2,prod_vege_2019.loc[:,('Occitanie', 'Production(1000 t)')], width = barWidth, label = 'Occitanie 2019') 
ax[1].legend();

ax[2].bar(x1, prod_vege_2018.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2018') 
ax[2].bar(x2,prod_vege_2019.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2019') 
ax[2].legend();

ax[3].bar(x1, prod_vege_2018.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2018") 
ax[3].bar(x2,prod_vege_2019.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2019")
ax[3].legend();

ax[4].bar(x1, prod_vege_2018.loc[:,("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2018") 
ax[4].bar(x2,prod_vege_2019.loc[:, ("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2019")
ax[4].legend();

ax[5].bar(x1, prod_vege_2018.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2018") 
ax[5].bar(x2,prod_vege_2019.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2019")
ax[5].legend();

ax[6].bar(x1, prod_vege_2018.loc[:,("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2018") 
ax[6].bar(x2,prod_vege_2019.loc[:, ("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2019")
ax[6].legend();


ax[7].bar(x1, prod_vege_2018.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2018") 
ax[7].bar(x2,prod_vege_2019.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2019")
ax[7].legend();

ax[8].bar(x1, prod_vege_2018.loc[:, ("Nouvelle-Aquitaine",'Production(1000 t)') ], width = barWidth, label = "Nouvelle-Aquitaine 2018") 
ax[8].bar(x2,prod_vege_2019.loc[:,("Nouvelle-Aquitaine",'Production(1000 t)')], width = barWidth, label = "Nouvelle-Aquitaine 2019")
ax[8].legend();

ax[9].bar(x1, prod_vege_2018.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2018") 
ax[9].bar(x2,prod_vege_2019.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2019")
ax[9].legend();

ax[10].bar(x1, prod_vege_2018.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2018") 
ax[10].bar(x2,prod_vege_2019.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2019")
ax[10].legend();

ax[11].bar(x1, prod_vege_2018.loc[:, ("Centre-Val de Loire", 'Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2018") 
ax[11].bar(x2,prod_vege_2019.loc[:, ("Centre-Val de Loire", 'Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2019")
ax[11].legend();

ax[12].bar(x1, prod_vege_2018.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2018") 
ax[12].bar(x2,prod_vege_2019.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2019")
plt.xticks(np.arange(len(prod_vege_2018[('','Cultures')])), prod_vege_2018[('','Cultures')].unique(), rotation = 90);

ax[12].legend();


# In[7]:


"On constate que la variable betteraves industrielles ne montre que des chiffres nuls pour les rendements et productions. Lors du nettoyage des données, les valeurs manquantes ont été remplacées par 0. C'est donc pour cette raison que nous n'avons que des 0"


# In[8]:


"Nous allons donc remplacer ces 0 par des valeurs se basant sur l'année de 2019 et de l'écart moyen entre 2018 et 2019."


# In[9]:


moyenne_2019 = prod_vege_2019.iloc[:,2:].drop(prod_vege_2019.index[5]).mean()
moyenne_2018 = prod_vege_2018.iloc[:,2:].drop(prod_vege_2018.index[5]).mean()
ecart_production_2018_2019 = (moyenne_2019 - moyenne_2018) / moyenne_2019
prod_vege_2018.iloc[5,2:] = prod_vege_2018.iloc[5,2:].replace([prod_vege_2018.iloc[5,2:].values], [np.array(prod_vege_2019.iloc[5,2:] - (moyenne_2019 *ecart_production_2018_2019))])


# In[10]:


fig, ax = plt.subplots(13, figsize=(30,40))
barWidth = 0.4
x1 = np.arange(len(prod_vege_2018[('','Cultures')]))
x2 = x1 + 0.4
ax[0].bar(x1, prod_vege_2018.loc[:,('France métropolitaine', 'Production(1000 t)')], width = barWidth, label = 'France 2018')
ax[0].bar(x2, prod_vege_2019.loc[:,('France métropolitaine', 'Production(1000 t)')],width = barWidth, label = 'France 2019')
ax[0].set_title('Productions grandes cultures 2018 et 2019 France métropolitaine')
ax[0].legend();

ax[1].bar(x1, prod_vege_2018.loc[:,('Occitanie','Production(1000 t)') ], width = barWidth, label = 'Occitanie 2018') 
ax[1].bar(x2,prod_vege_2019.loc[:,('Occitanie', 'Production(1000 t)')], width = barWidth, label = 'Occitanie 2019') 
ax[1].legend();

ax[2].bar(x1, prod_vege_2018.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2018') 
ax[2].bar(x2,prod_vege_2019.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2019') 
ax[2].legend();

ax[3].bar(x1, prod_vege_2018.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2018") 
ax[3].bar(x2,prod_vege_2019.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2019")
ax[3].legend();

ax[4].bar(x1, prod_vege_2018.loc[:,("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2018") 
ax[4].bar(x2,prod_vege_2019.loc[:, ("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2019")
ax[4].legend();

ax[5].bar(x1, prod_vege_2018.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2018") 
ax[5].bar(x2,prod_vege_2019.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2019")
ax[5].legend();

ax[6].bar(x1, prod_vege_2018.loc[:,("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2018") 
ax[6].bar(x2,prod_vege_2019.loc[:, ("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2019")
ax[6].legend();


ax[7].bar(x1, prod_vege_2018.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2018") 
ax[7].bar(x2,prod_vege_2019.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2019")
ax[7].legend();

ax[8].bar(x1, prod_vege_2018.loc[:, ("Nouvelle-Aquitaine",'Production(1000 t)') ], width = barWidth, label = "Nouvelle-Aquitaine 2018") 
ax[8].bar(x2,prod_vege_2019.loc[:,("Nouvelle-Aquitaine",'Production(1000 t)')], width = barWidth, label = "Nouvelle-Aquitaine 2019")
ax[8].legend();

ax[9].bar(x1, prod_vege_2018.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2018") 
ax[9].bar(x2,prod_vege_2019.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2019")
ax[9].legend();

ax[10].bar(x1, prod_vege_2018.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2018") 
ax[10].bar(x2,prod_vege_2019.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2019")
ax[10].legend();

ax[11].bar(x1, prod_vege_2018.loc[:, ("Centre-Val de Loire", 'Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2018") 
ax[11].bar(x2,prod_vege_2019.loc[:, ("Centre-Val de Loire", 'Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2019")
ax[11].legend();

ax[12].bar(x1, prod_vege_2018.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2018") 
ax[12].bar(x2,prod_vege_2019.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2019")
plt.xticks(np.arange(len(prod_vege_2018[('','Cultures')])), prod_vege_2018[('','Cultures')].unique(), rotation = 90);

ax[12].legend();


# In[11]:


fig, ax = plt.subplots(13, figsize=(30,40))
barWidth = 0.4
x1 = np.arange(len(prod_vege_2020[('','Cultures')]))
x2 = x1 + 0.4
ax[0].bar(x1, prod_vege_2019.loc[:,('France métropolitaine', 'Production(1000 t)')], width = barWidth, label = 'France 2019')
ax[0].bar(x2, prod_vege_2020.loc[:,('France métropolitaine', 'Production(1000 t)')],width = barWidth, label = 'France 2020')
ax[0].set_title('Productions grandes cultures 2019 et 2020 France métropolitaine')
ax[0].legend();

ax[1].bar(x1, prod_vege_2019.loc[:,('Occitanie','Production(1000 t)') ], width = barWidth, label = 'Occitanie 2019') 
ax[1].bar(x2,prod_vege_2020.loc[:,('Occitanie', 'Production(1000 t)')], width = barWidth, label = 'Occitanie 2020') 
ax[1].legend();

ax[2].bar(x1, prod_vege_2019.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2019') 
ax[2].bar(x2,prod_vege_2020.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2020') 
ax[2].legend();

ax[3].bar(x1, prod_vege_2019.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2019") 
ax[3].bar(x2,prod_vege_2020.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2020")
ax[3].legend();

ax[4].bar(x1, prod_vege_2019.loc[:,("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2019") 
ax[4].bar(x2,prod_vege_2020.loc[:, ("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2020")
ax[4].legend();

ax[5].bar(x1, prod_vege_2019.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2019") 
ax[5].bar(x2,prod_vege_2020.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2020")
ax[5].legend();

ax[6].bar(x1, prod_vege_2019.loc[:,("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2019") 
ax[6].bar(x2,prod_vege_2020.loc[:, ("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2020")
ax[6].legend();

ax[7].bar(x1, prod_vege_2019.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2019") 
ax[7].bar(x2,prod_vege_2020.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2020")
ax[7].legend();

ax[8].bar(x1, prod_vege_2019.loc[:, ("Nouvelle-Aquitaine",'Production(1000 t)') ], width = barWidth, label = "Nouvelle-Aquitaine 2019") 
ax[8].bar(x2,prod_vege_2020.loc[:,("Nouvelle-Aquitaine",'Production(1000 t)')], width = barWidth, label = "Nouvelle-Aquitaine 2020")
ax[8].legend();

ax[9].bar(x1, prod_vege_2019.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2019") 
ax[9].bar(x2,prod_vege_2020.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2020")
ax[9].legend();

ax[10].bar(x1, prod_vege_2019.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2019") 
ax[10].bar(x2,prod_vege_2020.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2020")
ax[10].legend();

ax[11].bar(x1, prod_vege_2019.loc[:, ("Centre-Val de Loire", 'Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2019") 
ax[11].bar(x2,prod_vege_2020.loc[:, ("Centre-Val de Loire", 'Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2020")
ax[11].legend();

ax[12].bar(x1, prod_vege_2019.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2019") 
ax[12].bar(x2,prod_vege_2020.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2020")
plt.xticks(np.arange(len(prod_vege_2020[('','Cultures')])), prod_vege_2020[('','Cultures')].unique(), rotation = 90);

ax[12].legend();


# In[12]:


'Baisse globale des productions particulièrement des betteraves industrielles en Ile de France. Seule exception : augmentation de la production de maïs au Pays-de-la-Loire et Bretagne.'


# In[13]:


fig, ax = plt.subplots(13, figsize=(30,40))
barWidth = 0.4
x1 = np.arange(len(prod_vege_2020[('','Cultures')]))
x2 = np.arange(len(prod_vege_2021[('','Cultures')])) + 0.4
ax[0].bar(x1, prod_vege_2020.loc[:,('France métropolitaine', 'Production(1000 t)')], width = barWidth, label = 'France 2020')
ax[0].bar(x2, prod_vege_2021.loc[:,('France métropolitaine', 'Production(1000 t)')],width = barWidth, label = 'France 2021')
ax[0].set_title('Productions grandes cultures 2020 et 2021 France métropolitaine')
ax[0].legend();

ax[1].bar(x1, prod_vege_2020.loc[:,('Occitanie','Production(1000 t)') ], width = barWidth, label = 'Occitanie 2020') 
ax[1].bar(x2,prod_vege_2021.loc[:,('Occitanie', 'Production(1000 t)')], width = barWidth, label = 'Occitanie 2021') 
ax[1].legend();

ax[2].bar(x1, prod_vege_2020.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2020') 
ax[2].bar(x2,prod_vege_2021.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2021') 
ax[2].legend();

ax[3].bar(x1, prod_vege_2020.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2020") 
ax[3].bar(x2,prod_vege_2021.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2021")
ax[3].legend();

ax[4].bar(x1, prod_vege_2020.loc[:,("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2020") 
ax[4].bar(x2,prod_vege_2021.loc[:, ("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2021")
ax[4].legend();

ax[5].bar(x1, prod_vege_2020.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2020") 
ax[5].bar(x2,prod_vege_2021.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2021")
ax[5].legend();

ax[6].bar(x1, prod_vege_2020.loc[:,("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2020") 
ax[6].bar(x2,prod_vege_2021.loc[:, ("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2021")
ax[6].legend();

ax[7].bar(x1, prod_vege_2020.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2020") 
ax[7].bar(x2,prod_vege_2021.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2021")
ax[7].legend();

ax[8].bar(x1, prod_vege_2020.loc[:, ("Nouvelle-Aquitaine",'Production(1000 t)') ], width = barWidth, label = "Nouvelle-Aquitaine 2020") 
ax[8].bar(x2,prod_vege_2021.loc[:,("Nouvelle-Aquitaine",'Production(1000 t)')], width = barWidth, label = "Nouvelle-Aquitaine 2021")
ax[8].legend();

ax[9].bar(x1, prod_vege_2020.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2020") 
ax[9].bar(x2,prod_vege_2021.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2021")
ax[9].legend();

ax[10].bar(x1, prod_vege_2020.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2020") 
ax[10].bar(x2,prod_vege_2021.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2021")
ax[10].legend();

ax[11].bar(x1, prod_vege_2020.loc[:, ('Centre-Val de Loire','Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2020") 
ax[11].bar(x2,prod_vege_2021.loc[:, ('Centre-Val de Loire','Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2021")
ax[11].legend();

ax[12].bar(x1, prod_vege_2020.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2020") 
ax[12].bar(x2,prod_vege_2021.loc[:,('Bourgogne-Franche-Comté',  'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2021")
plt.xticks(np.arange(len(prod_vege_2021[('','Cultures')])), prod_vege_2021[('','Cultures')].unique(), rotation = 90);

ax[12].legend();


# In[14]:


'Augmentation globale des productions, particulièrement des betteraves en Ile de France. Seule exception : baisse des productions de maïs au Pays-de-la-Loire et en Bretagne.'


# In[15]:


'Légère augmentation globale des productions, plus marquée pour la production de maïs en Bretagne.'


# In[16]:


"Pour résumer, l'année présentant la baisse la plus importante est 2020. Elle marque les premiers confinement dus à la pandémie de COVID 19. Ces événements pourraient en partie expliquer la baisse de production. Analysons les chiffres entre 2019 et 2021 afin de comprendre si la baisse a été compensée après les levées de restrictions liées à la pandémie. "


# In[17]:


fig, ax = plt.subplots(13, figsize=(30,40))
barWidth = 0.4
x1 = np.arange(len(prod_vege_2019[('','Cultures')]))
x2 = np.arange(len(prod_vege_2021[('','Cultures')])) + 0.4
ax[0].bar(x1, prod_vege_2019.loc[:,('France métropolitaine', 'Production(1000 t)')], width = barWidth, label = 'France 2019')
ax[0].bar(x2, prod_vege_2021.loc[:,('France métropolitaine', 'Production(1000 t)')],width = barWidth, label = 'France 2021')
ax[0].set_title('Productions grandes cultures 2019 et 2021 France métropolitaine')
ax[0].legend();

ax[1].bar(x1, prod_vege_2019.loc[:,('Occitanie','Production(1000 t)') ], width = barWidth, label = 'Occitanie 2019') 
ax[1].bar(x2,prod_vege_2021.loc[:,('Occitanie', 'Production(1000 t)')], width = barWidth, label = 'Occitanie 2021') 
ax[1].legend();

ax[2].bar(x1, prod_vege_2019.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2019') 
ax[2].bar(x2,prod_vege_2021.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2021') 
ax[2].legend();

ax[3].bar(x1, prod_vege_2019.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2019") 
ax[3].bar(x2,prod_vege_2021.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2021")
ax[3].legend();

ax[4].bar(x1, prod_vege_2019.loc[:,("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2019") 
ax[4].bar(x2,prod_vege_2021.loc[:, ("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2021")
ax[4].legend();

ax[5].bar(x1, prod_vege_2019.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2019") 
ax[5].bar(x2,prod_vege_2021.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2021")
ax[5].legend();

ax[6].bar(x1, prod_vege_2019.loc[:,("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2019") 
ax[6].bar(x2,prod_vege_2021.loc[:, ("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2021")
ax[6].legend();

ax[7].bar(x1, prod_vege_2019.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2019") 
ax[7].bar(x2,prod_vege_2021.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2021")
ax[7].legend();

ax[8].bar(x1, prod_vege_2019.loc[:, ("Nouvelle-Aquitaine",'Production(1000 t)') ], width = barWidth, label = "Nouvelle-Aquitaine 2019") 
ax[8].bar(x2,prod_vege_2021.loc[:,("Nouvelle-Aquitaine",'Production(1000 t)')], width = barWidth, label = "Nouvelle-Aquitaine 2021")
ax[8].legend();

ax[9].bar(x1, prod_vege_2019.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2019") 
ax[9].bar(x2,prod_vege_2021.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2021")
ax[9].legend();

ax[10].bar(x1, prod_vege_2019.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2019") 
ax[10].bar(x2,prod_vege_2021.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2021")
ax[10].legend();

ax[11].bar(x1, prod_vege_2019.loc[:, ('Centre-Val de Loire','Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2019") 
ax[11].bar(x2,prod_vege_2021.loc[:, ('Centre-Val de Loire','Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2021")
ax[11].legend();

ax[12].bar(x1, prod_vege_2019.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2019") 
ax[12].bar(x2,prod_vege_2021.loc[:,('Bourgogne-Franche-Comté',  'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2021")
plt.xticks(np.arange(len(prod_vege_2021[('','Cultures')])), prod_vege_2021[('','Cultures')].unique(), rotation = 90);

ax[12].legend();


# In[18]:


"Les baisses de l'année 2020 n'ont pas été complétement compensées en 2021 "


# In[19]:


'Faisons à présent une synthèse chiffrée des variations de surface, production et rendement de 2018 à 2021'


# In[20]:


moyenne_2018 = prod_vege_2018.iloc[:, prod_vege_2018.columns.get_level_values(1)=='Production(1000 t)'].mean()
moyenne_2019 = prod_vege_2019.iloc[:, prod_vege_2019.columns.get_level_values(1)=='Production(1000 t)'].mean()

ecart_production_2018_2019 = (moyenne_2019 - moyenne_2018) / moyenne_2019
ecart_production_2018_2019


# In[21]:


moyenne_2020 = prod_vege_2020.iloc[:, prod_vege_2020.columns.get_level_values(1)=='Production(1000 t)'].mean()
ecart_production_2019_2020 = (moyenne_2020 - moyenne_2019) / moyenne_2020
ecart_production_2019_2020


# In[22]:


moyenne_2021 = prod_vege_2021.iloc[:, prod_vege_2021.columns.get_level_values(1)=='Production(1000 t)'].mean()
ecart_production_2020_2021 = (moyenne_2021 - moyenne_2020) / moyenne_2021
ecart_production_2020_2021


# In[23]:


prod_2018_2019 = pd.DataFrame(ecart_production_2018_2019.values.reshape(14,1), columns = ['2018-2019'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
prod_2019_2020 = pd.DataFrame(ecart_production_2019_2020.values.reshape(14,1), columns = ['2019-2020'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
prod_2020_2021 = pd.DataFrame(ecart_production_2020_2021.values.reshape(14,1), columns = ['2020-2021'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())


# In[24]:


variations_prod_2018_2021 = pd.concat([prod_2018_2019, prod_2019_2020, prod_2020_2021], axis = 1)


# In[25]:


variations_prod_2018_2021 = (variations_prod_2018_2021 * 100).round(2).astype(str) + '%'


# In[26]:


moyenne_2018 = prod_vege_2018.iloc[:, prod_vege_2018.columns.get_level_values(1)=='Rendement(q/ha)'].mean()
moyenne_2019 = prod_vege_2019.iloc[:, prod_vege_2019.columns.get_level_values(1)=='Rendement(q/ha)'].mean()
ecart_rendement_2018_2019 = (moyenne_2019 - moyenne_2018) / moyenne_2019

moyenne_2020 = prod_vege_2020.iloc[:, prod_vege_2020.columns.get_level_values(1)=='Rendement(q/ha)'].mean()
ecart_rendement_2019_2020 = (moyenne_2020 - moyenne_2019) / moyenne_2020

moyenne_2021 = prod_vege_2021.iloc[:, prod_vege_2021.columns.get_level_values(1)=='Rendement(q/ha)'].mean()
ecart_rendement_2020_2021 = (moyenne_2021 - moyenne_2020) / moyenne_2021

rend_2018_2019 = pd.DataFrame(ecart_rendement_2018_2019.values.reshape(14,1), columns = ['2018-2019'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
rend_2019_2020 = pd.DataFrame(ecart_rendement_2019_2020.values.reshape(14,1), columns = ['2019-2020'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
rend_2020_2021 = pd.DataFrame(ecart_rendement_2020_2021.values.reshape(14,1), columns = ['2020-2021'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())

variations_rend_2018_2021 = pd.concat([rend_2018_2019, rend_2019_2020, rend_2020_2021], axis = 1)

variations_rend_2018_2021 = (variations_rend_2018_2021 * 100).round(2).astype(str) + '%'


# In[27]:


moyenne_2018 = prod_vege_2018.iloc[:, prod_vege_2018.columns.get_level_values(1)=='Superficie(1000 ha)'].mean()
moyenne_2019 = prod_vege_2019.iloc[:, prod_vege_2019.columns.get_level_values(1)=='Superficie(1000 ha)'].mean()
ecart_surface_2018_2019 = (moyenne_2019 - moyenne_2018) / moyenne_2019

moyenne_2020 = prod_vege_2020.iloc[:, prod_vege_2020.columns.get_level_values(1)=='Superficie(1000 ha)'].mean()
ecart_surface_2019_2020 = (moyenne_2020 - moyenne_2019) / moyenne_2020

moyenne_2021 = prod_vege_2021.iloc[:, prod_vege_2021.columns.get_level_values(1)=='Superficie(1000 ha)'].mean()
ecart_surface_2020_2021 = (moyenne_2021 - moyenne_2020) / moyenne_2021

surf_2018_2019 = pd.DataFrame(ecart_surface_2018_2019.values.reshape(14,1), columns = ['2018-2019'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
surf_2019_2020 = pd.DataFrame(ecart_surface_2019_2020.values.reshape(14,1), columns = ['2019-2020'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
surf_2020_2021 = pd.DataFrame(ecart_surface_2020_2021.values.reshape(14,1), columns = ['2020-2021'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())

variations_surf_2018_2021 = pd.concat([surf_2018_2019, surf_2019_2020, surf_2020_2021], axis = 1)

variations_surf_2018_2021 = (variations_surf_2018_2021 * 100).round(2).astype(str) + '%'


# In[28]:


variations_rend_2018_2021


# In[29]:


variations_surf_2018_2021


# In[30]:


variations_prod_2018_2021


# In[324]:


"On observe sans grande surprise que l'année 2020 présente la baisse la plus grande en rendement et production. La surface exploitée a été épargnée, c'est donc bien la production et le travail réalisé sur les surfaces agricoles qui ont baissées. La région la plus impactée est l'Occitanie."


# In[32]:


from bs4 import BeautifulSoup as bs
from urllib.request import urlopen

page = urlopen('https://reseauactionclimat.org/quels-impacts-du-changement-climatique-sur-lagriculture/')
soup = bs(page, 'html.parser')
soup.title.string


# In[33]:


soup.findAll("div") 


# In[34]:


texte = soup.find('div', {'class':'fc'}).text


# In[35]:


from nltk.tokenize import PunktSentenceTokenizer


# In[36]:


tokenizer = PunktSentenceTokenizer()
texte = tokenizer.tokenize(texte)


# In[37]:


texte[:14]


# In[38]:


texte[19]


# In[39]:


"Les risques liés au réchauffement climatique sur les exploitations sont donc très importants. Ce qu'on peut retenir c'est que les déréglements climatiques (sécheresse, précipitations importantes et autres phénomènes météorlogiques extrêmes) entraînent une destruction des récoltes ou une modification des dates de récoltes" 


# In[40]:


'Pour analyser plus précisément si les récoltes ont bien été avancées entre 2018 et 2021 et montreraient donc un risque de baisse de production, intéressons nous à la récolte du blé.'


# In[41]:


page = urlopen('https://vert-lavenir.com/ble/')
soup = bs(page, 'html.parser', from_encoding='utf-8')


# In[42]:


soup.findAll("div")


# In[43]:


texts = soup.find_all('p')
texte = []
for text in texts:
    texte.append(text.get_text())


# In[44]:


texte


# In[45]:


to_remember = texte[4:8]


# In[46]:


to_remember = []
for i in texte[4:8]:
    to_remember.append(i.replace(u'\xa0', u' '))


# In[47]:


to_remember


# In[48]:


"Pour bien comprendre les récoltes de blé et les impacts de la météo sur ceux-ci nous devons nous pencher sur la phénologie du blé. C'est à dire, l'étude du cycle de vie de la plante. Ainsi, nous pourrons constater d'une éventuelle modification de la croissance du végétal et donc des récoltes"


# In[422]:


pheno_ble = pd.read_csv(r"C:\Users\Hassan\Documents\datascientest\phenologie blé 2018 2021(3).csv", error_bad_lines = False, sep = ';', encoding="ISO-8859-1")


# In[423]:


pheno_ble.head()


# In[424]:


pheno_ble.info()


# In[425]:


pheno_ble.drop(['kingdom', 'data_source', 'scale', 'genus', 'binomial_name'], axis=1, inplace=True)


# In[426]:


pheno_ble.duplicated().sum()


# In[427]:


pheno_ble.drop_duplicates(inplace=True)


# In[428]:


pheno_ble.reset_index(drop=True, inplace=True)


# In[429]:


pheno_ble.head()


# In[430]:


pheno_ble['date'] = pd.to_datetime(pheno_ble['date'],format='%d/%m/%Y', utc = True)


# In[431]:


pheno_ble.head()


# In[432]:


pheno_ble.sort_values(by=['site_id','date'], inplace=True)


# In[433]:


pheno_ble.reset_index(drop=True, inplace=True)


# In[434]:


new_names = {'Centre':'Centre-Val de Loire','Languedoc-Roussillon':'Occitanie', 'Nord-Pas-de-Calais': 'Hauts-de-France', 'Limousin':'Nouvelle-Aquitaine','Poitou-Charentes':'Nouvelle-Aquitaine','Franche-Comté':'Bourgogne-Franche-Comté', 'Bourgogne':'Bourgogne-Franche-Comté','Auvergne':'Auvergne-Rhône-Alpes', 'Rhône-Alpes':'Auvergne-Rhône-Alpes','Champagne-Ardenne':'Grand Est','Alsace':'Grand Est','Midi-Pyrénées':'Occitanie', 'Picardie':'Hauts-de-France','Lorraine':'Grand Est', 'Aquitaine': 'Nouvelle-Aquitaine'}
pheno_ble = pheno_ble.replace(new_names)


# In[435]:


pheno_ble = pheno_ble.rename({'species':'taxon'}, axis=1)


# In[436]:


pheno_ble.head(40)


# In[437]:


stage_description = pheno_ble[pheno_ble['phenological_main_event_code'].isin(pheno_ble['phenological_main_event_code'].unique())][['phenological_main_event_code', 'phenological_main_event_description','stage_code', 'stage_description' ]].drop_duplicates()


# In[438]:


stage_description = stage_description.sort_values(by=['phenological_main_event_code', 'stage_code']).set_index('phenological_main_event_code')


# In[439]:


stage_description


# In[440]:


'Céréale d’automne le blé est semé d’octobre à novembre (stade 0 et 1) pour être récolté en juillet(stade 8) de l’année suivante, soit un cycle de 9 mois environ. Alors que l’automne se caractérise par la germination des graines et la pousse de petites plantes dites plantules, le printemps marque le développement du feuillage, la multiplication des tiges (ou talles) puis l’allongement de celles-ci jusqu’à la constitution puis le remplissage des grains.',


# In[441]:


from PIL import Image
from skimage import io
img = io.imread(r"C:\Users\Hassan\Documents\datascientest\projet datascientest PROD AGRICOLES\CycleBle-750x320-1.png")


# In[442]:


plt.figure(figsize=(20,20))
plt.imshow(img)
plt.show()


# In[443]:


"Selon le cours normal du cycle, on devrait avoir le début de vie de la plante aux alentours d'octobre, novembre, le tallage (Le tallage est un phénomène naturel qui permet d'obtenir plusieurs tiges à partir d'une seule.) en hiver, la montaison début printemps et le remplissage des grains en juin, qui s'en suit de la récolte en juillet. Analysons les données présentes afin de déterminer si c'est bien le calendrier suivi pour les années 2018, 2019, 2020 et 2021."


# In[444]:


stage_description


# In[445]:


stage_description['phenological_main_event_description'].unique()


# In[446]:


"Selon l'image décrivant le calendrier typique de croissance du blé, les stades 0 et 1 sont d'octobre à novembre, le stade 2 de novembre à février, le stade 3 de février à mai, les stades 5 et 6 de mai à juin et les stades 7 et 8 de juin à juillet (mois de récolte). Le stade 4 (gonflement et moment où la tête de la graine émerge de la gaine) n'est pas présent dans la description du calendrier."


# In[74]:


import datetime


# In[75]:


fig, ax = plt.subplots(1, figsize=(20,8))
ax = sns.scatterplot(pheno_ble['date'], pheno_ble['phenological_main_event_code'], hue = pheno_ble['grid_label'])
ax_twiny = ax.twiny()
plt.xticks(pheno_ble['date'],pheno_ble['date'].dt.month)
ax.legend(bbox_to_anchor=(1.35, 1));


# In[79]:


fig, ax = plt.subplots(1, figsize=(10,8))
ax = sns.scatterplot(pheno_ble['date'][pheno_ble['year']==2018], pheno_ble['phenological_main_event_code'][pheno_ble['taxon'] == 'aestivum'][pheno_ble['year']==2018], hue = pheno_ble['grid_label'][pheno_ble['year']==2018])
ax_twiny = ax.twiny()
plt.xticks(pheno_ble['date'][pheno_ble['year']==2018],pheno_ble['date'][pheno_ble['year']==2018].dt.month)
plt.title('stades du cycle du blé 2018')
ax.legend(bbox_to_anchor=(1.35, 1));


# In[80]:


"Stades 0, 1 et 2 début à la mi-septembre, octobre et fin octobre pour la récolte de 2019. Pour la récolte de 2018 les stades 1 et 2 se terminent respectivement en avril (à part quelques exceptions en mai) et en mai, avec donc environ 3 mois de retard sur le calendrier décrit plus haut. Le stade 3 débute en février et finit en juin. Les stades 5 et 6 commencent fin avril et finissent fin juin et juillet.Les stades 7 et 8 débutent en juin et se terminent en juillet, donc avec un mois d'avance par rapport au calendrier typique."


# In[81]:


fig, ax = plt.subplots(1, figsize=(10,8))
ax = sns.scatterplot(pheno_ble['date'][pheno_ble['year']==2019], pheno_ble['phenological_main_event_code'][pheno_ble['year']==2019], hue = pheno_ble['grid_label'][pheno_ble['year']==2019])
ax_twiny = ax.twiny()
plt.xticks(pheno_ble['date'][pheno_ble['year']==2019],pheno_ble['date'][pheno_ble['year']==2019].dt.month)
plt.title('stades du cycle du blé 2019')
ax.legend(bbox_to_anchor=(1.35, 1));


# In[82]:


"Les stades 1 et 2 débutés en octobre et fin octobre / mi-novembre 2018 se terminent en avril et mai avec donc environ 3 mois de retard sur le calendrier. Le stade 3 débute en février et finit fin juin. Les stades débutent 5 et 6 fin avril / mai et finissent en juillet. Les stades 7 et 8 commencent en mai et juin et se terminent fin juillet, donc avec un mois d'avance par rapport au calendrier typique. Stades 0, 1 et 2 pour la récolte de 2020 commencent en octobre."


# In[83]:


fig, ax = plt.subplots(1, figsize=(10,8))
ax = sns.scatterplot(pheno_ble['date'][pheno_ble['year']==2020], pheno_ble['phenological_main_event_code'][pheno_ble['year']==2020], hue = pheno_ble['grid_label'][pheno_ble['year']==2020])
ax_twiny = ax.twiny()
plt.xticks(pheno_ble['date'][pheno_ble['year']==2020],pheno_ble['date'][pheno_ble['year']==2020].dt.month)
plt.title('stades du cycle du blé 2020')

ax.legend(bbox_to_anchor=(1.35, 1));


# In[84]:


'Calendrier similaire à 2019'


# In[85]:


fig, ax = plt.subplots(1, figsize=(10,8))
ax = sns.scatterplot(pheno_ble['date'][pheno_ble['year']==2021], pheno_ble['phenological_main_event_code'][pheno_ble['year']==2021], hue = pheno_ble['grid_label'][pheno_ble['year']==2021])
ax_twiny = ax.twiny()
plt.xticks(pheno_ble['date'][pheno_ble['year']==2021],pheno_ble['date'][pheno_ble['year']==2021].dt.month)
plt.title('stades du cycle du blé 2021')
ax.legend(bbox_to_anchor=(1.35, 1));


# In[87]:


pheno_ble[(pheno_ble['phenological_main_event_code']== 2) & (pheno_ble['grid_label'] == 'Pays de la Loire') & (pheno_ble['taxon']=='aestivum')].sort_values(by=['site_id', 'date'])


# In[88]:


pheno_ble[pheno_ble['site_id']== 41812].sort_values(by=['phenological_main_event_code','date'])


# In[89]:


"Si nous prenons l'exemple du stade 2 en région Pays de la Loire pour le site 41812, nous observons que les dates vont du 23/11/2020 au 29/01/2021. Ce qui signifie que le stade 2 débute en novembre et finit en janvier. C'est pourquoi, nous observons sur les graphs que le stade 2 revient 2 fois dans l'année. Il en est de même pour d'autres sites et pour le stade de croissance 1."


# In[90]:


"Nous avons visiblement moins de données pour l'année 2021, les stades 5 et supérieurs ne sont pas représentés. (éventuellement utiliser l'année 2021 comme test, pour prédictions/analyse en ML de la relation météo et phénologie.)"


# In[91]:


pheno_ble[pheno_ble['year']==2021].shape


# In[92]:


pheno_ble[pheno_ble['year']==2020].shape


# In[93]:


pheno_ble[pheno_ble['year']==2019].shape


# In[94]:


pheno_ble[pheno_ble['year']==2018].shape


# In[447]:


percent_category_2018 = []
for i in pheno_ble[pheno_ble['year']== 2018]['phenological_main_event_code'].value_counts():
    percent_category_2018.append((i/len(pheno_ble[pheno_ble['year']== 2018]['phenological_main_event_code']))*100)
percent_category_2018 = pd.DataFrame(np.array(percent_category_2018), columns = ['phenological_main_event_code 2018'], index = pheno_ble[pheno_ble['year']== 2018]['phenological_main_event_code'].value_counts().index)
percent_category_2018 = percent_category_2018.round(2).astype(str) + '%'


# In[448]:


percent_category_2018


# In[449]:


rend2018 = pd.DataFrame(prod_vege_2018.iloc[6:12, prod_vege_2018.columns.get_level_values(1)=='Rendement(q/ha)'].mean(), columns = ['2018'])


# In[450]:


rend2018.iloc[0,:]


# In[451]:


percent_category_2019 = []
for i in pheno_ble[pheno_ble['year']== 2019]['phenological_main_event_code'].value_counts():
    percent_category_2019.append((i/len(pheno_ble[pheno_ble['year']== 2019]['phenological_main_event_code']))*100)
percent_category_2019 = pd.DataFrame(np.array(percent_category_2019), columns = ['phenological_main_event_code 2019'], index = pheno_ble[pheno_ble['year']== 2019]['phenological_main_event_code'].value_counts().index)
percent_category_2019 = percent_category_2019.round(2).astype(str) + '%'


# In[452]:


percent_category_2019


# In[453]:


rend2019 = pd.DataFrame(prod_vege_2019.iloc[6:12, prod_vege_2019.columns.get_level_values(1)=='Rendement(q/ha)'].mean(), columns = ['2019'])


# In[454]:


rend2019.iloc[0,:]


# In[455]:


percent_category_2020 = []
for i in pheno_ble[pheno_ble['year']== 2020]['phenological_main_event_code'].value_counts():
    percent_category_2020.append((i/len(pheno_ble[pheno_ble['year']== 2020]['phenological_main_event_code']))*100)
percent_category_2020 = pd.DataFrame(np.array(percent_category_2020), columns = ['phenological_main_event_code 2020'], index = pheno_ble[pheno_ble['year']== 2020]['phenological_main_event_code'].value_counts().index)
percent_category_2020 = percent_category_2020.round(2).astype(str) + '%'


# In[456]:


percent_category_2020


# In[457]:


rend2020 = pd.DataFrame(prod_vege_2020.iloc[6:12, prod_vege_2020.columns.get_level_values(1)=='Rendement(q/ha)'].mean(), columns = ['2020'])


# In[458]:


rend2020.iloc[0,:]


# In[459]:


"On constate, en observant le rendement moyen sur tout le territoire et la répartition des différents stades de croissance du blé qu'il est difficile de trouver une corrélation satisfaisante entre les deux. Le stade 8 qui annonce la récolte est proportionellement plus important en 2018 et 2020 bien que les rendements soient plus faibles qu'en 2019. C'est donc sur un autre critère que l'on doit s'orienter pour estimer le rendement de la culture par rapport aux stades phénologiques."


# In[108]:


page = urlopen('https://www.agro.basf.fr/fr/cultures/ble/implantation_du_ble/les_composantes_du_rendement_du_ble/')
soup = bs(page, 'html.parser', from_encoding='utf-8')


# In[109]:


soup.find('main')


# In[110]:


soup.find('div', {'class':'col text-left'}).text.strip()


# In[111]:


"Le facteur favorable à un bon rendement est une montaison longue qui permet d'augmenter le volume de grains. En d'autres termes, on veut une plante qui produit beaucoup de grains pour assurer un bon rendement. Cela correspond au stade 3 de croissance du blé. Pour l'année 2019, on remarque que le stade 3 est majoritaire. C'est donc probablement une des raisons pour lesquelles le rendement est plus important."


# In[112]:


pheno_ble[pheno_ble['phenological_main_event_code'] ==3].head(30)


# In[113]:


display(pheno_ble[pheno_ble['site_id'] == 25903])
"En prenant l'exemple de la plante 25903, on s'aperçoit que le stade phénologique 3 dure plus longtemps que les autres stades."


# In[114]:


import PyPDF2
   
pdfFileObj = open(r"C:\Users\Hassan\Documents\datascientest\projet datascientest PROD AGRICOLES\impact-climatique-phéno-blé.pdf", 'rb')
   
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
   
print(pdfReader.numPages)
   
page1 = pdfReader.getPage(0)
page2 = pdfReader.getPage(1)
text1 = page1.extractText()
text2 = page2.extractText()
print(text1)
print(text2)


# In[115]:


meteo_2018_2021 = pd.read_csv(r"C:\Users\Hassan\Documents\datascientest\projet datascientest PROD AGRICOLES\projet Datascientest météo\meteo 2018 2021 (2).csv", sep=';', error_bad_lines = False)


# In[116]:


meteo_2018_2021.head()


# In[117]:


meteo_2018_2021.info()


# In[118]:


meteo_2018_2021['Date'] = pd.to_datetime(meteo_2018_2021['Date'], utc = True)


# In[119]:


meteo_2018_2021['region (name)'].value_counts()


# In[120]:


meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Guyane'].index, inplace=True)
meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Terres australes et antarctiques françaises'].index, inplace=True)
meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Guadeloupe'].index, inplace=True)
meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Saint-Pierre-et-Miquelon'].index, inplace=True)
meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Mayotte'].index, inplace=True)
meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'La Réunion'].index, inplace=True)
meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Martinique'].index, inplace=True)


# In[121]:


meteo_2018_2021 = meteo_2018_2021.sort_values(by=['Date','region (name)'])
meteo_2018_2021.reset_index(drop =True, inplace = True)


# In[122]:


meteo_2018_2021.head()


# In[123]:


meteo_2018_2021.rename({'Visibilité horizontale': 'Visibilité horizontale (en mètre)', 'region (name)' : 'nom'}, axis=1, inplace=True)


# In[124]:


meteo_2018_2021.head()


# In[125]:


columns_to_drop = meteo_2018_2021.columns[(meteo_2018_2021.notna().sum() < 81000) == True]
meteo_2018_2021.drop(columns_to_drop, axis=1, inplace = True)


# In[126]:


meteo_2018_2021.head()


# In[127]:


meteo_2018_2021.drop(['Type de tendance barométrique.1','region (code)','communes (code)', 'mois_de_l_annee', 'EPCI (name)', 'EPCI (code)', 'Temps présent', 'Nom', 'communes (name)','ID OMM station' ], axis=1, inplace = True)


# In[128]:


meteo_2018_2021.info()


# In[129]:


meteo_2018_2021 = meteo_2018_2021.fillna(method="ffill")


# In[130]:


meteo_2018_2021['Latitude'].unique()


# In[131]:


'Deux outliers : -49.352333 et -66.663167'


# In[132]:


meteo_2018_2021 = meteo_2018_2021.query("Latitude != -49.352333")


# In[133]:


meteo_2018_2021 = meteo_2018_2021.query("Latitude != -66.663167")


# In[134]:


sns.heatmap(meteo_2018_2021.corr())


# In[338]:


"Nous allons à présent créer une nouvelle colonne de la moyenne des températures les 5 derniers jours. Elle nous sera utile pour les prédictions futures."


# In[339]:


meteo_2018_2021['rolling_avg_temp'] = meteo_2018_2021['Température (°C)'].rolling(window=5, min_periods=1).mean()


# In[340]:


meteo_2018_2021.describe()


# In[137]:


plt.figure(figsize=(10,10))
plt.plot_date(meteo_2018_2021['Date'], meteo_2018_2021['Température (°C)'])


# In[138]:


'Températures froides records en 2018 et 2021 et chaudes en 2019 et 2020'


# In[139]:


fig, axes = plt.subplots(2, 2, figsize=(10,10))

sns.boxplot(meteo_2018_2021['Température (°C)'][meteo_2018_2021['Date'].dt.year == 2018], color = 'orange', ax = axes[0,0]).set(title='2018')
sns.boxplot(meteo_2018_2021['Température (°C)'][meteo_2018_2021['Date'].dt.year == 2019], ax = axes[0,1]).set(title = '2019')
sns.boxplot(meteo_2018_2021['Température (°C)'][meteo_2018_2021['Date'].dt.year == 2020], color= 'green', ax = axes[1,0]).set(title = '2020')
sns.boxplot(meteo_2018_2021['Température (°C)'][meteo_2018_2021['Date'].dt.year == 2021], color = 'red', ax = axes[1,1]).set(title = '2021')
plt.legend();


# In[140]:


"Dans le détail, on constate que la moyenne des températures est assez stable mais que l'année 2018 présente plusieurs valeurs de températures extrêmes froides et 2019 et 2020 de températures extrêmes chaudes. Les températures hivernales de l'année 2020 semblent plus douces, avec des valeurs n'allant pas au-delà de -5°."


# In[141]:


"L'autre facteur déterminant en agriculture est la précipitation. Voyons donc l'évolution de 2018 à 2021"


# In[142]:


plt.figure(figsize=(10,10))
plt.plot_date(meteo_2018_2021['Date'], meteo_2018_2021['Précipitations dans les 24 dernières heures'])
plt.title('Précipitations de 2018 à 2021')
plt.legend();


# In[143]:


fig, axes = plt.subplots(2, 2, figsize=(10,10))

sns.boxplot(meteo_2018_2021['Précipitations dans les 24 dernières heures'][meteo_2018_2021['Date'].dt.year == 2018], color = 'orange', ax = axes[0,0]).set(title='2018')
sns.boxplot(meteo_2018_2021['Précipitations dans les 24 dernières heures'][meteo_2018_2021['Date'].dt.year == 2019], ax = axes[0,1]).set(title = '2019')
sns.boxplot(meteo_2018_2021['Précipitations dans les 24 dernières heures'][meteo_2018_2021['Date'].dt.year == 2020], color= 'green', ax = axes[1,0]).set(title = '2020')
sns.boxplot(meteo_2018_2021['Précipitations dans les 24 dernières heures'][meteo_2018_2021['Date'].dt.year == 2021], color = 'red', ax = axes[1,1]).set(title = '2021')
plt.legend();


# In[144]:


"L'année la plus pluvieuse est 2018, la plus sèche 2019. 2020 et 2021 sont relativement similaires."


# In[145]:


pip install geopandas


# In[146]:


import geopandas as gpd
shapefile = gpd.read_file('https://france-geojson.gregoiredavid.fr/repo/regions.geojson')

merged1 = shapefile.merge(meteo_2018_2021[meteo_2018_2021['Date'].dt.year == 2018].groupby('nom').mean(), on = 'nom')
merged2 = shapefile.merge(meteo_2018_2021[meteo_2018_2021['Date'].dt.year == 2019].groupby('nom').mean(), on = 'nom')
merged3 = shapefile.merge(meteo_2018_2021[meteo_2018_2021['Date'].dt.year == 2020].groupby('nom').mean(), on = 'nom')
merged4 = shapefile.merge(meteo_2018_2021[meteo_2018_2021['Date'].dt.year == 2021].groupby('nom').mean(), on = 'nom')

fig, ax = plt.subplots(4, figsize=(20, 20))

merged1.plot('Humidité', cmap='YlGnBu', ax=ax[0], legend=True)
ax[0].set_axis_off()
ax[0].set_title("Distribution spatiale de l'humidité moyenne \n \n 2018")

merged2.plot('Humidité', cmap='YlGnBu', ax=ax[1], legend=True)
ax[1].set_axis_off()
ax[1].set_title("2019")

merged3.plot('Humidité', cmap='YlGnBu', ax=ax[2], legend=True)
ax[2].set_axis_off()
ax[2].set_title("2020")

merged4.plot('Humidité', cmap='YlGnBu', ax=ax[3], legend=True)
ax[3].set_axis_off()
ax[3].set_title("2021")




plt.show()


# In[147]:


fig, ax = plt.subplots(4, figsize=(20, 20))

merged1.plot('Température (°C)', cmap='OrRd', ax=ax[0], legend=True)
ax[0].set_axis_off()
ax[0].set_title("Distribution spatiale des t° moyennes \n \n 2018")

merged2.plot('Température (°C)', cmap='OrRd', ax=ax[1], legend=True)
ax[1].set_axis_off()
ax[1].set_title("2019")

merged3.plot('Température (°C)', cmap='OrRd', ax=ax[2], legend=True)
ax[2].set_axis_off()
ax[2].set_title("2020")

merged4.plot('Température (°C)', cmap='OrRd', ax=ax[3], legend=True)
ax[3].set_axis_off()
ax[3].set_title("2021")




plt.show()


# In[148]:


"En observant les moyennes des températures sur le territoire, l'année la plus chaude semble être 2020 et l'année la plus froide 2021. La répartition des températures est néanmoins inégale pour cette dernière année, les régions du Sud et l'Ile de France semblent montrer des températures bien plus élevées que le reste du pays. "


# In[149]:


fig, ax = plt.subplots(4, figsize=(20, 20))

merged1.plot('Précipitations dans les 24 dernières heures', cmap='YlGnBu', ax=ax[0], legend=True)
ax[0].set_axis_off()
ax[0].set_title("Distribution spatiale des précipitations moyennes \n \n 2018")

merged2.plot('Précipitations dans les 24 dernières heures', cmap='YlGnBu', ax=ax[1], legend=True)
ax[1].set_axis_off()
ax[1].set_title("2019")

merged3.plot('Précipitations dans les 24 dernières heures', cmap='YlGnBu', ax=ax[2], legend=True)
ax[2].set_axis_off()
ax[2].set_title("2020")

merged4.plot('Précipitations dans les 24 dernières heures', cmap='YlGnBu', ax=ax[3], legend=True)
ax[3].set_axis_off()
ax[3].set_title("2021")




plt.show()


# In[332]:


"Les données visualisées sur la carte confirment bien que 2018 était une année pluvieuse et 2019 une année plutôt sèche(à l'exception de l'ouest). On constate néanmoins que la répartition des précipitations est différente entre 2020 et 2021. En 2020, il a plumajoritairement dans l'ouest tandis qu'en 2021, les précipitations sont plus homogènes sur le territoire, à l'exception de l'Ile de France, où les pluies ont été abondantes."


# In[460]:


train_data_pheno = pheno_ble.query("year != 2021")
train_data_pheno = pheno_ble[['stage_code', 'date','grid_label','site_latitude','site_longitude','phenological_main_event_code']]


# In[461]:


train_data_meteo = meteo_2018_2021.drop(['department (name)','department (code)', 'Coordonnees','Altitude'], axis=1)


# In[462]:


train_data_meteo.head()


# In[463]:


train_data_meteo.info()


# In[464]:


train_data_meteo = train_data_meteo.loc[train_data_meteo['Température minimale sur 12 heures (°C)'].notna(), :]


# In[465]:


train_data_pheno.rename({'grid_label':'nom', 'date':'Date'}, axis=1, inplace = True)


# In[466]:


train_data = pd.merge(train_data_meteo,train_data_pheno, on=['nom', 'Date'])


# In[467]:


train_data.drop(['site_latitude','site_longitude'], axis= 1, inplace = True)


# In[468]:


train_data.duplicated().sum()


# In[469]:


train_data.drop_duplicates(inplace=True)


# In[470]:


train_data.head()


# In[471]:


train_data.corr()


# In[472]:


sns.heatmap(train_data.corr())


# In[473]:


'Selon le tableau de corrélation, la variable météorologique la plus corrélée au stade de développement du blé est la température'


# In[474]:


from scipy.stats import chi2_contingency
print("Hypothèse H0 : il n'y a pas d'influence de la variable 'Température (°C)' sur les stades de croissance du blé 'phenological_main_event_code'")

table = pd.crosstab(train_data['phenological_main_event_code'], train_data['Température (°C)'])

chi2, p, dof, expected = chi2_contingency(table)

print(chi2) 
print(p)  
print('p-value inférieure à 0.05, donc H0 réfutée')


# In[475]:


from scipy.stats import chi2_contingency
print("Hypothèse H0 : il n'y a pas d'influence de la variable 'rolling_avg_temp' sur les stades de croissance du blé 'phenological_main_event_code'")

table = pd.crosstab(train_data['phenological_main_event_code'], train_data['rolling_avg_temp'])

chi2, p, dof, expected = chi2_contingency(table)

print(chi2) 
print(p)  
print('p-value inférieure à 0.05, donc H0 réfutée')


# In[476]:


print("Hypothèse H0 : il n'y a pas d'influence de la variable 'Précipitations dans les 24 dernières heures' sur les stades de croissance du blé 'phenological_main_event_code'")

table = pd.crosstab(train_data['phenological_main_event_code'], train_data['Précipitations dans les 24 dernières heures'])

chi2, p, dof, expected = chi2_contingency(table)

print(chi2) 
print(p)  
print("p-value inférieure à 0.05, donc H0 réfutée, la statistique chi-2 étant plus faible que pour le test avec les températures, on en déduit que cette variable joue moins d'importance")


# In[477]:


from scipy.stats import chi2_contingency
print("Hypothèse H0 : il n'y a pas d'influence de la variable 'Humidité' sur les stades de croissance du blé 'phenological_main_event_code'")

table = pd.crosstab(train_data['phenological_main_event_code'], train_data['Humidité'])

chi2, p, dof, expected = chi2_contingency(table)

print(chi2) 
print(p)  
print('p-value inférieure à 0.05, donc H0 réfutée')
"L'hypothèse est bien réfutée mais le résultat du test est nettement inférieur aux deux précédents, donc la variable a peu d'influence."


# In[478]:


train_data = pd.concat([pd.get_dummies(train_data.nom), train_data], axis =1)


# In[479]:


train_data.drop(['nom','Pression au niveau mer','Variation de pression en 3 heures','Type de tendance barométrique','Direction du vent moyen 10 mn','Vitesse du vent moyen 10 mn', 'Point de rosée','Visibilité horizontale (en mètre)',"Nebulosité totale","Nébulosité  des nuages de l' étage inférieur","Hauteur de la base des nuages de l'étage inférieur",'Pression station','Variation de pression en 24 heures', 'Rafale sur les 10 dernières minutes','Rafales sur une période','Periode de mesure de la rafale','Etat du sol','Hauteur totale de la couche de neige, glace, autre au sol', 'Nébulosité couche nuageuse 1','Hauteur de base 1','Nébulosité couche nuageuse 2','Hauteur de base 2', 'Température','Température minimale sur 12 heures','Température maximale sur 12 heures','Température minimale du sol sur 12 heures'], axis=1, inplace = True)


# In[480]:


train_data['year'] = train_data.Date.dt.year
train_data['month'] = train_data.Date.dt.month
train_data['day'] = train_data.Date.dt.day


# In[481]:


train_data.drop('Date', axis=1, inplace=True)


# In[482]:


train_data.columns


# In[483]:


data = train_data.drop(['stage_code','phenological_main_event_code'], axis=1)
target = train_data['phenological_main_event_code']


# In[484]:


data.head()


# In[485]:


test_2021 = meteo_2018_2021[meteo_2018_2021['Date'].dt.year == 2021].drop(['Pression au niveau mer','Variation de pression en 3 heures','Type de tendance barométrique','Direction du vent moyen 10 mn','Vitesse du vent moyen 10 mn', 'Point de rosée','Visibilité horizontale (en mètre)',"Nebulosité totale","Nébulosité  des nuages de l' étage inférieur","Hauteur de la base des nuages de l'étage inférieur",'Pression station','Variation de pression en 24 heures', 'Rafale sur les 10 dernières minutes','Rafales sur une période','Periode de mesure de la rafale','Etat du sol','Hauteur totale de la couche de neige, glace, autre au sol', 'Nébulosité couche nuageuse 1','Hauteur de base 1','Nébulosité couche nuageuse 2','Hauteur de base 2', 'Température','Température minimale sur 12 heures','Température maximale sur 12 heures','Température minimale du sol sur 12 heures', 'Coordonnees','department (name)','department (code)','Altitude'], axis=1)
test_2021 = test_2021.loc[train_data_meteo['Température minimale sur 12 heures (°C)'].notna(), :]


# In[486]:


test_2021.shape


# In[487]:


test_2021 = pd.concat([test_2021, pd.get_dummies(test_2021['nom'])], axis=1)


# In[488]:


test_2021['rolling_avg_temp'] = test_2021['Température (°C)'].rolling(window=5, min_periods=1).mean()


# In[489]:


test_2021.head()


# In[490]:


test_2021 = test_2021.groupby(['Date','nom']).head(2)


# In[491]:


test_2021.drop('nom', axis=1, inplace=True)


# In[492]:


test_2021['year'] = test_2021.Date.dt.year
test_2021['month'] = test_2021.Date.dt.month
test_2021['day'] = test_2021.Date.dt.day
test_2021.drop('Date', axis=1, inplace=True)


# In[493]:


test_2021.drop('Corse', axis=1, inplace=True)


# In[494]:


test_2021.head()


# In[495]:


test_2021.shape


# In[496]:


test_2021.drop_duplicates(inplace=True)


# In[497]:


from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split


# In[498]:


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 123)


# In[499]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[500]:


param_grid = {'n_estimators': [10, 50, 100],
              'max_depth': [5, 10, 20],
              'min_samples_split': [2, 5, 10]
              }
rf = RandomForestClassifier()

grid_search = GridSearchCV(rf, param_grid, cv=7, scoring='accuracy')

grid_search.fit(X_train_scaled, y_train)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best score: {:.2f}".format(grid_search.best_score_))


# In[501]:


rf = RandomForestClassifier(max_depth =  10, min_samples_split = 10, n_estimators = 50)


# In[502]:


rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
print(pd.crosstab(y_pred, y_test))
rf.score(X_test_scaled, y_test)


# In[503]:


features = data.columns
features_importance = {}
sorted_features = {}
for x,j in zip(features, rf.feature_importances_):
    features_importance[x] = j
sorted_features = sorted(features_importance.items(), key=lambda x:x[1], reverse=True) 
print(sorted_features[:8])


# In[504]:


precision_score(y_pred, y_test, average = "weighted")


# In[505]:



print(f'precision score : {precision_score(y_pred, y_test, average = "weighted")}')


# In[506]:


test_2021_scaled = scaler.transform(test_2021)


# In[507]:


y_pred_2021 = rf.predict(test_2021_scaled)
np.unique(y_pred_2021)


# In[508]:


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators = 100, max_depth = 7, learning_rate = 0.01, subsample = 1)


# In[509]:


gb.fit(X_train_scaled, y_train)
y_pred = gb.predict(X_test_scaled)
print(pd.crosstab(y_pred, y_test))
gb.score(X_test_scaled, y_test)


# In[510]:


features = data.columns
features_importance = {}
sorted_features = {}
for x,j in zip(features, gb.feature_importances_):
    features_importance[x] = j
sorted_features = sorted(features_importance.items(), key=lambda x:x[1], reverse=True) 
print(sorted_features[:8])


# In[511]:


print(f'precision score : {precision_score(y_pred, y_test, average = "weighted")}')


# In[512]:


y_pred_2021 = gb.predict(test_2021_scaled)
np.unique(y_pred_2021)


# In[513]:


clf = DecisionTreeClassifier()
param_grid = {'max_depth': range(1,10),
              'min_samples_split': [2, 3, 4, 5, 6, 7],
              'criterion': ['gini','entropy'],
              }

grid_search = GridSearchCV(clf, param_grid, cv=8, scoring='accuracy')

grid_search.fit(X_train_scaled, y_train)

print(grid_search.best_params_)

print(grid_search.best_score_)
print(grid_search.best_estimator_)


# In[524]:


clf_entr = DecisionTreeClassifier(criterion = 'entropy', max_depth = 9,min_samples_split = 7)
clf_entr.fit(X_train_scaled, y_train)
y_pred = clf_entr.predict(X_test_scaled)
pd.crosstab(y_pred, y_test)


# In[525]:


clf_entr.score(X_test_scaled, y_test)


# In[526]:


features = data.columns
features_importance = {}
sorted_features = {}
for x,j in zip(features, clf_entr.feature_importances_):
    features_importance[x] = j
sorted_features = sorted(features_importance.items(), key=lambda x:x[1], reverse=True) 
print(sorted_features[:8])


# In[527]:


print(f'precision score : {precision_score(y_pred, y_test, average = "weighted")}')


# In[528]:


y_pred_2021 = clf_entr.predict(test_2021_scaled)


# In[529]:


np.unique(y_pred_2021)


# In[520]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print("Accuracy: {:.2f}".format(knn.score(X_test_scaled, y_test)))


# In[521]:


print(f'precision score : {precision_score(y_pred, y_test, average = "weighted")}')


# In[522]:


y_pred_knn_2021 = knn.predict(test_2021_scaled)
np.unique(y_pred_knn_2021)


# In[523]:


'modèle plus performant : decision tree, metric : entropy'


# In[530]:


test_2021['phenological_main_event_code'] = y_pred_2021


# In[531]:


test_2021.head()


# In[532]:


pheno_meteo = pd.concat([train_data, test_2021])
pheno_meteo['régions'] = pheno_meteo[['Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté','Bretagne','Centre-Val de Loire','Grand Est','Hauts-de-France','Normandie','Nouvelle-Aquitaine','Occitanie','Pays de la Loire',"Provence-Alpes-Côte d'Azur",'Île-de-France']].idxmax(axis=1)


# In[533]:


pheno_meteo.drop(['Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté','Bretagne','Centre-Val de Loire','Grand Est','Hauts-de-France','Normandie','Nouvelle-Aquitaine','Occitanie','Pays de la Loire',"Provence-Alpes-Côte d'Azur",'Île-de-France'], axis=1, inplace=True)


# In[534]:


pheno_meteo['date'] = pd.to_datetime(pheno_meteo[['year', 'month', 'day']])


# In[535]:


pheno_meteo.head()


# In[300]:


fig, ax = plt.subplots(1, figsize=(30, 12))
ax = sns.scatterplot(pheno_meteo['date'], pheno_meteo['phenological_main_event_code'], hue = pheno_meteo['régions'])
ax_twiny = ax.twiny()
plt.xticks(pheno_meteo['date'],pheno_meteo['date'].dt.month)
ax.legend(bbox_to_anchor=(1.35, 1));


# In[301]:


fig, ax = plt.subplots(1, figsize=(20,8))
ax = sns.scatterplot(pheno_meteo['date'][pheno_meteo['year'] == 2021], pheno_meteo['phenological_main_event_code'][pheno_meteo['year'] == 2021], hue = pheno_meteo['régions'][pheno_meteo['year'] == 2021])
ax_twiny = ax.twiny()
plt.xticks(pheno_meteo['date'][pheno_meteo['year'] == 2021],pheno_meteo['date'][pheno_meteo['year'] == 2021].dt.month)
ax.legend(bbox_to_anchor=(1.35, 1));


# In[222]:


page = urlopen('https://www.linfodurable.fr/environnement/38-degres-en-2100-rechauffement-climatique-pire-que-prevu-en-france-34833')
soup = bs(page, 'html.parser')


# In[302]:


percent_category_2021 = []
for i in pheno_meteo[pheno_meteo['year']== 2021]['phenological_main_event_code'].value_counts():
    percent_category_2021.append((i/len(pheno_meteo[pheno_meteo['year']== 2021]['phenological_main_event_code']))*100)
percent_category_2021 = pd.DataFrame(np.array(percent_category_2021), columns = ['phenological_main_event_code 2021'], index = pheno_meteo[pheno_meteo['year']== 2021]['phenological_main_event_code'].value_counts().index)
percent_category_2021 = percent_category_2021.round(2).astype(str) + '%'


# In[303]:


rend2021 = pd.DataFrame(prod_vege_2021.iloc[6:12, prod_vege_2021.columns.get_level_values(1)=='Rendement(q/ha)'].mean(), columns = ['2021'])


# In[304]:


percent_category_2021


# In[305]:


rend2021.iloc[0,:]


# In[308]:


"Le stade 3 représente une partie relativement faible des stades observés (de très peu majoritaire), ce qui peut expliquer un rendement moins important pour cette année. Ce sont des prédictions avec plus de dates que pour les autres années, c'est pourquoi la répartition des stades diffère des années précemment étudiées."


# In[228]:


soup.find_all('div')


# In[229]:


soup.find('div', {'class':'col-8_sm-12'})


# In[230]:


texte1 = soup.find('h2',{'class':'font-medium fs-20 node-20'}).text.strip()


# In[231]:


tokenizer = PunktSentenceTokenizer()
texte = tokenizer.tokenize(texte1)


# In[232]:


texte2 = tokenizer.tokenize(soup.find('div', {'class':'clearfix text-formatted field field--name-field-article-body field--type-text-with-summary field--label-hidden field__item'}).text.strip())


# In[233]:


texte2


# In[234]:


print(texte1, texte2[10:18])


# In[235]:


"Nous allons maintenant utiliser les informations du rapport du GIEC concernant les prédictions de hausses de températures. En 2100 les températures augmenteront en moyenne de 3.8°C, mais en hiver de 3.2°C et en été de 5.1°C. Nous allons donc augmenter en conséquence les températures de notre dataset météo et prédire les stades de croissance du blé."


# In[236]:


pheno_meteo.head()


# In[237]:


"Pour réaliser les prédictions, prenons comme référence l'année 2018. C'est celle pour laquelle nous avons le plus de données."


# In[309]:


pheno_meteo_pred_train = pheno_meteo[pheno_meteo['year'] == 2018].drop(['date','stage_code','phenological_main_event_code'], axis=1)


# In[310]:


pheno_meteo_pred_train['year'] = pheno_meteo_pred_train['year'].replace(2018, 2100)


# In[311]:


pheno_meteo_pred_train = pd.concat([pd.get_dummies(pheno_meteo_pred_train['régions']), pheno_meteo_pred_train], axis =1).drop('régions', axis=1)


# In[312]:


pheno_meteo_pred_train.head()


# In[313]:


pheno_meteo_pred_train.loc[(pheno_meteo_pred_train['month'] <= 4) | (pheno_meteo_pred_train['month'] >= 9),'Température (°C)'] += 3.2
pheno_meteo_pred_train.loc[(pheno_meteo_pred_train['month'] > 4) & (pheno_meteo_pred_train['month'] < 9),'Température (°C)'] += 5.1
pheno_meteo_pred_train.loc[(pheno_meteo_pred_train['month'] <= 4) | (pheno_meteo_pred_train['month'] >= 9),'Température minimale sur 12 heures (°C)'] += 3.2
pheno_meteo_pred_train.loc[(pheno_meteo_pred_train['month'] > 4) & (pheno_meteo_pred_train['month'] < 9),'Température minimale sur 12 heures (°C)'] += 5.1
pheno_meteo_pred_train.loc[(pheno_meteo_pred_train['month'] <= 4) | (pheno_meteo_pred_train['month'] >= 9),'Température maximale sur 12 heures (°C)'] += 3.2
pheno_meteo_pred_train.loc[(pheno_meteo_pred_train['month'] > 4) & (pheno_meteo_pred_train['month'] < 9),'Température maximale sur 12 heures (°C)'] += 5.1
pheno_meteo_pred_train.loc[(pheno_meteo_pred_train['month'] <= 4) | (pheno_meteo_pred_train['month'] >= 9),'Température minimale du sol sur 12 heures (en °C)'] += 3.2
pheno_meteo_pred_train.loc[(pheno_meteo_pred_train['month'] > 4) & (pheno_meteo_pred_train['month'] < 9),'Température minimale du sol sur 12 heures (en °C)'] += 5.1


# In[314]:


pheno_meteo_pred_train.head()


# In[315]:


pheno_meteo_pred_train_scaled = scaler.transform(pheno_meteo_pred_train)


# In[316]:


y_pred_2100 = clf_entr.predict(pheno_meteo_pred_train_scaled)


# In[317]:


np.unique(y_pred_2100)


# In[318]:


pheno_meteo_pred_train['phenological_main_event_code'] = y_pred_2100


# In[319]:


pheno_meteo_pred_train.head()


# In[320]:


pheno_meteo_pred_train['régions'] = pheno_meteo_pred_train[['Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté','Bretagne','Centre-Val de Loire','Grand Est','Hauts-de-France','Normandie','Nouvelle-Aquitaine','Occitanie','Pays de la Loire',"Provence-Alpes-Côte d'Azur",'Île-de-France']].idxmax(axis=1)
pheno_meteo_pred_train.drop(['Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté','Bretagne','Centre-Val de Loire','Grand Est','Hauts-de-France','Normandie','Nouvelle-Aquitaine','Occitanie','Pays de la Loire',"Provence-Alpes-Côte d'Azur",'Île-de-France'], axis=1, inplace=True)
pheno_meteo_pred_train['date'] = pd.to_datetime(pheno_meteo_pred_train[['year', 'month', 'day']])


# In[250]:


fig, ax = plt.subplots(2, figsize=(15,12))

sns.scatterplot(pheno_meteo_pred_train['date'], pheno_meteo_pred_train['phenological_main_event_code'], hue = pheno_meteo_pred_train['régions'], ax=ax[0], alpha = 0.5)

sns.scatterplot(pheno_meteo['date'][pheno_meteo['year']==2018], pheno_meteo['phenological_main_event_code'][pheno_meteo['year']==2018],  hue = pheno_meteo['régions'], ax=ax[1], alpha = 0.5)
ax_twiny = ax[0].twiny()

plt.xticks(pheno_meteo_pred_train['date'],pheno_meteo_pred_train['date'].dt.month)
plt.xticks(pheno_meteo['date'][pheno_meteo['year']==2018],pheno_meteo['date'][pheno_meteo['year']==2018].dt.month)

ax[0].legend(bbox_to_anchor=(1.35, 1))
ax[1].legend(bbox_to_anchor=(1.35, 1))


# In[321]:


percent_category_2100 = []
for i in pheno_meteo_pred_train['phenological_main_event_code'].value_counts():
    percent_category_2100.append((i/len(pheno_meteo_pred_train['phenological_main_event_code']))*100)
percent_category_2100 = pd.DataFrame(np.array(percent_category_2100), columns = ['phenological_main_event_code 2100'], index = pheno_meteo_pred_train['phenological_main_event_code'].value_counts().index)
percent_category_2100 = percent_category_2100.round(2).astype(str) + '%'


# In[322]:


percent_category_2100


# In[323]:


percent_category_2018


# In[251]:


"Comme démontré précedemment, le stade déterminant à une bonne récolte est le stade de montaison à 1cm d'épi qui correspond au stade 3. On en compte moins en 2100 qu'en 2018. On peut donc supposer que selon ces prédictions, le rendement aura tendance à être plus faible en 2100."


# In[252]:


"En conclusion, nous avons exploré les données de productions agricoles de 2018 à 2021 et avons constaté une baisse de rendement en 2020. Sachant que cette année a été fortement impactée par les confinements, cette baisse pourrait s'expliquer par le manque de main d'oeuvre pour travailler les terres agricoles. Le but étant d'établir un lien entre les données météorologiques et la production agricole, les données telles quelles étaient insuffisantes. Il fallait comprendre le comportement du végétal dans son développement étape par étape. J'ai choisi de cibler le blé car il représente une denrée alimentaire essentielle pour l'humanité et un bon rendement est donc crucial. L'étude phénologique du blé se base sur la mise en relation des dates attendues des différents stades par rapport aux dates observées. Pour les 4 années concernées, on observe déjà un décalage, souvent les stades sont précoces(ils commencent trop tôt ou finissent plus tôt que prévus), ce qui est communément signe de températures plus élevées que la moyenne de saison. En cherchant plus loin la signification et l'importance de chaque stade, j'ai pu apprendre qu'en agriculture, le stade de montaison est déterminant pour la récolte. Si ce stade est plus court, les épis seront moins denses donc produiront moins de grains. C'est comme ça que j'ai pu établir le lien entre les rendements de 2018 à 2020 en constatant que 2019 était l'année la plus productive car le stade 3 était plus long pour la majorité des plantes observées. Ceci étant fait, j'ai analysé les données météo et l'année la plus froide fut 2021 et la plus chaude 2020. La plus pluvieuse 2018 et la plus sèche 2019. Ceci dit, ce sont des observations des moyennes des différentes données, qui ne prennent donc pas en compte les amplitudes. J'ai ensuite mergé les données phénologiques et météo pour ensuite effectué des tests statistiques afin de démontrer une corrélation entre ces deux types de variables. Selon le test chi-2, les températures et précipitations sont bien corrélées aux stades de croissance du blé. On a donc pu établir un modèle de ML avec en features les données météo et target les stades phénologiques. Premièrement, j'ai prédit les stades de l'année 2021 car incomplets dans le dataset d'origine, puis j'ai finalement fait les prédictions de l'année 2100 en utilisant les hausses de température estimées par le dernier rapport du GIEC. Les résultats montrent une diminution de la période de montaison (stade 3), qui pourrait donc amener à une récolte plus faible. Les variables déterminantes sont en réalité plus complexes et diverses, du recensement de la main d'oeuvre agricole à l'utilisation de pesticides, en passant par les irrigations artificielles, les facteurs sont très nombreux. C'est pourquoi, ces prédictions restent aproximatives et ne prennent pas en compte tout le champs possible des conditions de l'exploitation agricole."


# In[ ]:




