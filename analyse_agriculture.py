# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:29:41 2023

@author: Sara
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

import streamlit as st
st.set_page_config(page_title = 'Projet DS meteo agriculture', )
st.set_option('deprecation.showPyplotGlobalUse', False)
def analyse_agriculture_function():
    prod_vege_2018 = pd.read_csv("Resources/2018_donneesgrandescultures.csv", sep=';', header = [0,1])
    prod_vege_2018.iloc[:,1:] = prod_vege_2018.iloc[:,1:].astype(float)
    prod_vege_2018.rename({'Unnamed: 0_level_0':''}, axis=1, inplace = True)
    prod_vege_2018.iloc[:,1:] = prod_vege_2018.iloc[:,1:].round(2)
    
    prod_vege_2018.fillna(0, inplace = True)
    prod_vege_2018 = prod_vege_2018.sort_values(by=('', 'Cultures'))
    prod_vege_2018.reset_index(drop=True, inplace=True)
    prod_vege_2018.insert(0, "Année", '2018')
    prod_vege_2018['Année'] = prod_vege_2018['Année'].apply(lambda x: pd.to_datetime(str(x),format='%Y%'))
    
    
    
    # In[2]:
    
    
    #prod_vege_2018.isna().sum()
    
    
    # In[3]:
    st.header('Analyse des productions, surfaces et rendements agricoles de 2018 à 2021')
    
    prod_vege_2019 = pd.read_csv("Resources/2019_donneesgrandescultures.csv", sep=';', header = [0,1])
    
    prod_vege_2019.iloc[:,1:] = prod_vege_2019.iloc[:,1:].astype(float)
    prod_vege_2019.rename({'Unnamed: 0_level_0':''}, axis=1, inplace = True)
    prod_vege_2019.iloc[:,1:] = prod_vege_2019.iloc[:,1:].round(2)
    
    
    prod_vege_2019.fillna(0, inplace=True)
    prod_vege_2019 = prod_vege_2019.sort_values(by=('', 'Cultures'))
    prod_vege_2019.reset_index(drop=True, inplace=True)
    prod_vege_2019.insert(0, "Année", '2019')
    prod_vege_2019['Année'] =prod_vege_2019['Année'].apply(lambda x: pd.to_datetime(str(x),format='%Y%'))
    
    
    # In[4]:
    
    
    prod_vege_2020 = pd.read_csv("Resources/2020_donneesgrandescultures.csv", sep=';', header=[0,1])
    
    prod_vege_2020.head()
    
    prod_vege_2020.iloc[:,1:] = prod_vege_2020.iloc[:,1:].astype(float)
    prod_vege_2020.rename({'Unnamed: 0_level_0':''}, axis=1, inplace = True)
    prod_vege_2020.iloc[:,1:] = prod_vege_2020.iloc[:,1:].round(2)
    
    
    prod_vege_2020.fillna(0, inplace=True)
    
    prod_vege_2020 = prod_vege_2020.sort_values(by=('', 'Cultures'))
    prod_vege_2020.reset_index(drop=True, inplace=True)
    prod_vege_2020.insert(0, "Année", '2020')
    prod_vege_2020['Année'] =prod_vege_2020['Année'].apply(lambda x: pd.to_datetime(str(x),format='%Y%'))
    
    
    
    # In[5]:
    
    
    prod_vege_2021 = pd.read_csv("Resources/2021_donneesgrandescultures (1).csv", sep=';', header=[0,1])
    
    prod_vege_2021.iloc[:,1:] = prod_vege_2021.iloc[:,1:].astype(float)
    prod_vege_2021.rename({'Unnamed: 0_level_0':''}, axis=1, inplace = True)
    prod_vege_2021.iloc[:,1:] = prod_vege_2021.iloc[:,1:].round(2)
    
    
    prod_vege_2021.fillna(0, inplace=True)
    
    prod_vege_2021.replace({'Rendement(q/h)': 'Rendement(q/ha)'}, inplace=True)
    
    prod_vege_2021 = prod_vege_2021.sort_values(by=('', 'Cultures'))
    prod_vege_2021.reset_index(drop=True, inplace=True)
    prod_vege_2021.insert(0, "Année", '2021')
    prod_vege_2021['Année'] =prod_vege_2021['Année'].apply(lambda x: pd.to_datetime(str(x),format='%Y%'))
    
    
    
    # In[6]:
    
    annee = st.selectbox('Années de productions agricoles',['2018-2019','2019-2020', '2020-2021'])
    
    
    
    # In[31]:
    
    
    "On constate que la variable 'betteraves industrielles' ne montre que des chiffres nuls pour les rendements et productions. Lors du nettoyage des données, les valeurs manquantes ont été remplacées par 0. C'est donc pour cette raison que nous n'avons que des 0"
    
    
    # In[8]:
    
    
    "Nous allons donc remplacer ces 0 par des valeurs se basant sur l'année de 2019 et de l'écart moyen entre 2018 et 2019."
    
    
    # In[9]:
    
    
    moyenne_2019 = prod_vege_2019.iloc[:,2:].drop(prod_vege_2019.index[5]).mean()
    moyenne_2018 = prod_vege_2018.iloc[:,2:].drop(prod_vege_2018.index[5]).mean()
    ecart_production_2018_2019 = (moyenne_2019 - moyenne_2018) / moyenne_2019
    prod_vege_2018.iloc[5,2:] = prod_vege_2018.iloc[5,2:].replace([prod_vege_2018.iloc[5,2:].values], [np.array(prod_vege_2019.iloc[5,2:] - (moyenne_2019 *ecart_production_2018_2019))])
    
    
    # In[10]:
    
    if annee == '2018-2019':
        fig, ax = plt.subplots(13, figsize=(30,40))
        
        barWidth = 0.4
        x1 = np.arange(len(prod_vege_2018[('','Cultures')]))
        x2 = x1 + 0.4
        ax[0].bar(x1, prod_vege_2018.loc[:,('France métropolitaine', 'Production(1000 t)')], width = barWidth, label = 'France 2018')
        ax[0].bar(x2, prod_vege_2019.loc[:,('France métropolitaine', 'Production(1000 t)')],width = barWidth, label = 'France 2019')
        ax[0].set_title('Productions grandes cultures 2018 et 2019 France métropolitaine', fontsize = 40)
        ax[0].set_xticks([])
        ax[0].legend();
        
        ax[1].bar(x1, prod_vege_2018.loc[:,('Occitanie','Production(1000 t)') ], width = barWidth, label = 'Occitanie 2018') 
        ax[1].bar(x2,prod_vege_2019.loc[:,('Occitanie', 'Production(1000 t)')], width = barWidth, label = 'Occitanie 2019') 
        ax[1].set_xticks([])
        ax[1].legend();
        
        ax[2].bar(x1, prod_vege_2018.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2018') 
        ax[2].bar(x2,prod_vege_2019.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2019') 
        ax[2].set_xticks([])
        ax[2].legend();
        
        ax[3].bar(x1, prod_vege_2018.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2018") 
        ax[3].bar(x2,prod_vege_2019.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2019")
        ax[3].set_xticks([])
        ax[3].legend();
        
        ax[4].bar(x1, prod_vege_2018.loc[:,("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2018") 
        ax[4].bar(x2,prod_vege_2019.loc[:, ("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2019")
        ax[4].set_xticks([])
        ax[4].legend();
        
        ax[5].bar(x1, prod_vege_2018.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2018") 
        ax[5].bar(x2,prod_vege_2019.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2019")
        ax[5].set_xticks([])
        ax[5].legend();
        
        ax[6].bar(x1, prod_vege_2018.loc[:,("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2018") 
        ax[6].bar(x2,prod_vege_2019.loc[:, ("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2019")
        ax[6].set_xticks([])
        ax[6].legend();
        
        
        ax[7].bar(x1, prod_vege_2018.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2018") 
        ax[7].bar(x2,prod_vege_2019.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2019")
        ax[7].set_xticks([])
        ax[7].legend();
        
        ax[8].bar(x1, prod_vege_2018.loc[:, ("Nouvelle-Aquitaine",'Production(1000 t)') ], width = barWidth, label = "Nouvelle-Aquitaine 2018") 
        ax[8].bar(x2,prod_vege_2019.loc[:,("Nouvelle-Aquitaine",'Production(1000 t)')], width = barWidth, label = "Nouvelle-Aquitaine 2019")
        ax[8].set_xticks([])
        ax[8].legend();
        
        ax[9].bar(x1, prod_vege_2018.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2018") 
        ax[9].bar(x2,prod_vege_2019.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2019")
        ax[9].set_xticks([])
        ax[9].legend();
        
        ax[10].bar(x1, prod_vege_2018.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2018") 
        ax[10].bar(x2,prod_vege_2019.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2019")
        ax[10].set_xticks([])
        ax[10].legend();
        
        ax[11].bar(x1, prod_vege_2018.loc[:, ("Centre-Val de Loire", 'Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2018") 
        ax[11].bar(x2,prod_vege_2019.loc[:, ("Centre-Val de Loire", 'Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2019")
        ax[11].set_xticks([])
        ax[11].legend();
        
        ax[12].bar(x1, prod_vege_2018.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2018") 
        ax[12].bar(x2,prod_vege_2019.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2019")
        plt.xticks(np.arange(len(prod_vege_2018[('','Cultures')])), prod_vege_2018[('','Cultures')].unique(), rotation = 90)
        xticks = plt.gca().get_xticklabels()
        plt.setp(xticks, fontsize=25)
        
        ax[12].legend();
        st.pyplot(clear_figure = True)
        st.markdown('_Observations : Légère augmentation globale des différentes cultures sauf maïs._')
    
    if annee == '2019-2020':
    
        fig, ax = plt.subplots(13, figsize=(30,40))
        barWidth = 0.4
        x1 = np.arange(len(prod_vege_2020[('','Cultures')]))
        x2 = x1 + 0.4
        ax[0].bar(x1, prod_vege_2019.loc[:,('France métropolitaine', 'Production(1000 t)')], width = barWidth, label = 'France 2019')
        ax[0].bar(x2, prod_vege_2020.loc[:,('France métropolitaine', 'Production(1000 t)')],width = barWidth, label = 'France 2020')
        ax[0].set_title('Productions grandes cultures 2019 et 2020 France métropolitaine', fontsize = 40)
        ax[0].set_xticks([])
        ax[0].legend();
        
        ax[1].bar(x1, prod_vege_2019.loc[:,('Occitanie','Production(1000 t)') ], width = barWidth, label = 'Occitanie 2019') 
        ax[1].bar(x2,prod_vege_2020.loc[:,('Occitanie', 'Production(1000 t)')], width = barWidth, label = 'Occitanie 2020') 
        ax[1].set_xticks([])
        ax[1].legend();
        
        ax[2].bar(x1, prod_vege_2019.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2019') 
        ax[2].bar(x2,prod_vege_2020.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2020') 
        ax[2].set_xticks([])
        ax[2].legend();
        
        ax[3].bar(x1, prod_vege_2019.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2019") 
        ax[3].bar(x2,prod_vege_2020.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2020")
        ax[3].set_xticks([])
        ax[3].legend();
        
        ax[4].bar(x1, prod_vege_2019.loc[:,("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2019") 
        ax[4].bar(x2,prod_vege_2020.loc[:, ("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2020")
        ax[4].set_xticks([])
        ax[4].legend();
        
        ax[5].bar(x1, prod_vege_2019.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2019") 
        ax[5].bar(x2,prod_vege_2020.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2020")
        ax[5].set_xticks([])
        ax[5].legend();
        
        ax[6].bar(x1, prod_vege_2019.loc[:,("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2019") 
        ax[6].bar(x2,prod_vege_2020.loc[:, ("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2020")
        ax[6].set_xticks([])
        ax[6].legend();
        
        ax[7].bar(x1, prod_vege_2019.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2019") 
        ax[7].bar(x2,prod_vege_2020.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2020")
        ax[7].set_xticks([])
        ax[7].legend();
        
        ax[8].bar(x1, prod_vege_2019.loc[:, ("Nouvelle-Aquitaine",'Production(1000 t)') ], width = barWidth, label = "Nouvelle-Aquitaine 2019") 
        ax[8].bar(x2,prod_vege_2020.loc[:,("Nouvelle-Aquitaine",'Production(1000 t)')], width = barWidth, label = "Nouvelle-Aquitaine 2020")
        ax[8].set_xticks([])
        ax[8].legend();
        
        ax[9].bar(x1, prod_vege_2019.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2019") 
        ax[9].bar(x2,prod_vege_2020.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2020")
        ax[9].set_xticks([])
        ax[9].legend();
        
        ax[10].bar(x1, prod_vege_2019.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2019") 
        ax[10].bar(x2,prod_vege_2020.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2020")
        ax[10].set_xticks([])
        ax[10].legend();
        
        ax[11].bar(x1, prod_vege_2019.loc[:, ("Centre-Val de Loire", 'Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2019") 
        ax[11].bar(x2,prod_vege_2020.loc[:, ("Centre-Val de Loire", 'Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2020")
        ax[11].set_xticks([])
        ax[11].legend();
        
        ax[12].bar(x1, prod_vege_2019.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2019") 
        ax[12].bar(x2,prod_vege_2020.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2020")
        plt.xticks(np.arange(len(prod_vege_2020[('','Cultures')])), prod_vege_2020[('','Cultures')].unique(), rotation = 90);
        xticks = plt.gca().get_xticklabels()
        plt.setp(xticks, fontsize=25)
        ax[12].legend();
        st.pyplot()
        
    
        st.markdown('_Observations : Baisse globale des productions particulièrement des betteraves industrielles en Ile de France. Seule exception : augmentation de la production de maïs au Pays-de-la-Loire et Bretagne._')
    
    
    # In[13]:
    if annee == '2020-2021':
    
        fig, ax = plt.subplots(13, figsize=(30,40))
        barWidth = 0.4
        x1 = np.arange(len(prod_vege_2020[('','Cultures')]))
        x2 = np.arange(len(prod_vege_2021[('','Cultures')])) + 0.4
        ax[0].bar(x1, prod_vege_2020.loc[:,('France métropolitaine', 'Production(1000 t)')], width = barWidth, label = 'France 2020')
        ax[0].bar(x2, prod_vege_2021.loc[:,('France métropolitaine', 'Production(1000 t)')],width = barWidth, label = 'France 2021')
        ax[0].set_title('Productions grandes cultures 2020 et 2021 France métropolitaine', fontsize = 40)
        ax[0].set_xticks([])
        ax[0].legend();
        
        ax[1].bar(x1, prod_vege_2020.loc[:,('Occitanie','Production(1000 t)') ], width = barWidth, label = 'Occitanie 2020') 
        ax[1].bar(x2,prod_vege_2021.loc[:,('Occitanie', 'Production(1000 t)')], width = barWidth, label = 'Occitanie 2021') 
        ax[1].set_xticks([])
        ax[1].legend();
        
        ax[2].bar(x1, prod_vege_2020.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2020') 
        ax[2].bar(x2,prod_vege_2021.loc[:,('Auvergne-Rhône-Alpes', 'Production(1000 t)')], width = barWidth, label = 'Auvergne-Rhône-Alpes 2021') 
        ax[2].set_xticks([])
        ax[2].legend();
        
        ax[3].bar(x1, prod_vege_2020.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2020") 
        ax[3].bar(x2,prod_vege_2021.loc[:,("Provence-Alpes-Côte-d'Azur",'Production(1000 t)')], width = barWidth, label = "PACA 2021")
        ax[3].set_xticks([])
        ax[3].legend();
        
        ax[4].bar(x1, prod_vege_2020.loc[:,("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2020") 
        ax[4].bar(x2,prod_vege_2021.loc[:, ("Bretagne", 'Production(1000 t)')], width = barWidth, label = "Bretagne 2021")
        ax[4].set_xticks([])
        ax[4].legend();
        
        ax[5].bar(x1, prod_vege_2020.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2020") 
        ax[5].bar(x2,prod_vege_2021.loc[:, ("Hauts de France", 'Production(1000 t)')], width = barWidth, label = "Hauts-de-France 2021")
        ax[5].set_xticks([])
        ax[5].legend();
        
        ax[6].bar(x1, prod_vege_2020.loc[:,("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2020") 
        ax[6].bar(x2,prod_vege_2021.loc[:, ("Ile-de-France", 'Production(1000 t)')], width = barWidth, label = "Ile-de-France 2021")
        ax[6].set_xticks([])
        ax[6].legend();
        
        ax[7].bar(x1, prod_vege_2020.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2020") 
        ax[7].bar(x2,prod_vege_2021.loc[:,("Pays-de-la-Loire", 'Production(1000 t)')], width = barWidth, label = "Pays-de-la-Loire 2021")
        ax[7].set_xticks([])
        ax[7].legend();
        
        ax[8].bar(x1, prod_vege_2020.loc[:, ("Nouvelle-Aquitaine",'Production(1000 t)') ], width = barWidth, label = "Nouvelle-Aquitaine 2020") 
        ax[8].bar(x2,prod_vege_2021.loc[:,("Nouvelle-Aquitaine",'Production(1000 t)')], width = barWidth, label = "Nouvelle-Aquitaine 2021")
        ax[8].set_xticks([])
        ax[8].legend();
        
        ax[9].bar(x1, prod_vege_2020.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2020") 
        ax[9].bar(x2,prod_vege_2021.loc[:, ("Grand Est", 'Production(1000 t)')], width = barWidth, label = "Grand-Est 2021")
        ax[9].set_xticks([])
        ax[9].legend();
        
        ax[10].bar(x1, prod_vege_2020.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2020") 
        ax[10].bar(x2,prod_vege_2021.loc[:,("Normandie",'Production(1000 t)')], width = barWidth, label = "Normandie 2021")
        ax[10].set_xticks([])
        ax[10].legend();
        
        ax[11].bar(x1, prod_vege_2020.loc[:, ('Centre-Val de Loire','Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2020") 
        ax[11].bar(x2,prod_vege_2021.loc[:, ('Centre-Val de Loire','Production(1000 t)')], width = barWidth, label = "Centre-Val de Loire 2021")
        ax[11].set_xticks([])
        ax[11].legend();
        
        ax[12].bar(x1, prod_vege_2020.loc[:,("Bourgogne-Franche-Comté",'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2020") 
        ax[12].bar(x2,prod_vege_2021.loc[:,('Bourgogne-Franche-Comté',  'Production(1000 t)')], width = barWidth, label = "Bourgogne-Franche-Comté 2021")
        plt.xticks(np.arange(len(prod_vege_2021[('','Cultures')])), prod_vege_2021[('','Cultures')].unique(), rotation = 90);
        xticks = plt.gca().get_xticklabels()
        plt.setp(xticks, fontsize=25)
        ax[12].legend();
        
        st.pyplot(clear_figure = True)
    
    
    
        st.markdown('_Observations : Légère augmentation globale des productions, plus marquée pour la production de maïs en Bretagne._')
    
    
    # In[16]:
    
    
    st.markdown("**Pour résumer, l'année présentant la baisse la plus importante est 2020. Elle marque les premiers confinement dus à la pandémie de COVID 19. Ces événements pourraient en partie expliquer la baisse de production.**")
    
    
    
    
    
    # In[19]:
    
    
    st.header('Synthèse chiffrée des variations de surface, production et rendement de 2018 à 2021')
    
    
    # In[20]:
    
    
    moyenne_2018 = prod_vege_2018.iloc[:, prod_vege_2018.columns.get_level_values(1)=='Production(1000 t)'].mean()
    moyenne_2019 = prod_vege_2019.iloc[:, prod_vege_2019.columns.get_level_values(1)=='Production(1000 t)'].mean()
    
    ecart_production_2018_2019 = pd.DataFrame((moyenne_2019 - moyenne_2018) / moyenne_2019, columns = ['ecart_production_2018_2019'])
    
    
    # In[21]:
    
    
    moyenne_2020 = prod_vege_2020.iloc[:, prod_vege_2020.columns.get_level_values(1)=='Production(1000 t)'].mean()
    ecart_production_2019_2020 = pd.DataFrame((moyenne_2020 - moyenne_2019) / moyenne_2020, columns = ['ecart_production_2019_2020'])
    
    
    # In[22]:
    
    
    moyenne_2021 = prod_vege_2021.iloc[:, prod_vege_2021.columns.get_level_values(1)=='Production(1000 t)'].mean()
    ecart_production_2020_2021 = pd.DataFrame((moyenne_2021 - moyenne_2020) / moyenne_2021, columns = ['ecart_production_2020_2021'])
    
    
    # In[23]:
    
    
    prod_2018_2019 = pd.DataFrame(ecart_production_2018_2019.values.reshape(14,1), columns = ['écart production 2018-2019'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    prod_2019_2020 = pd.DataFrame(ecart_production_2019_2020.values.reshape(14,1), columns = ['écart production 2019-2020'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    prod_2020_2021 = pd.DataFrame(ecart_production_2020_2021.values.reshape(14,1), columns = ['écart production 2020-2021'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    
    
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
    
    rend_2018_2019 = pd.DataFrame(ecart_rendement_2018_2019.values.reshape(14,1), columns = ['écart rendement 2018-2019'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    rend_2019_2020 = pd.DataFrame(ecart_rendement_2019_2020.values.reshape(14,1), columns = ['écart rendement 2019-2020'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    rend_2020_2021 = pd.DataFrame(ecart_rendement_2020_2021.values.reshape(14,1), columns = ['écart rendement 2020-2021'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    
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
    
    surf_2018_2019 = pd.DataFrame(ecart_surface_2018_2019.values.reshape(14,1), columns = ['écart surface 2018-2019'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    surf_2019_2020 = pd.DataFrame(ecart_surface_2019_2020.values.reshape(14,1), columns = ['écart surface 2019-2020'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    surf_2020_2021 = pd.DataFrame(ecart_surface_2020_2021.values.reshape(14,1), columns = ['écart surface 2020-2021'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    
    variations_surf_2018_2021 = pd.concat([surf_2018_2019, surf_2019_2020, surf_2020_2021], axis = 1)
    
    variations_surf_2018_2021 = (variations_surf_2018_2021 * 100).round(2).astype(str) + '%'
    
    
    # In[28]:
    
    variation = st.selectbox('Evolution des caractéristiques agricoles de 2018 à 2021', ['Production', 'Rendement', 'Surface'])
    if variation == 'Rendement':
        st.write(variations_rend_2018_2021)
    
    
    # In[29]:
    
    if variation == 'Surface':
        st.write(variations_surf_2018_2021)
    
    
    # In[30]:
    
    if variation == 'Production':
        st.write(variations_prod_2018_2021)
    
    
    # In[324]:
    
    
    st.markdown("**On observe sans grande surprise que l'année 2020 présente la baisse la plus grande en rendement et production. La surface exploitée a été relativement épargnée, c'est donc bien la production et le travail réalisé sur les surfaces agricoles qui ont baissées. La région la plus impactée est l'Occitanie.**")
    
    
    # In[2]:
    
    
    from bs4 import BeautifulSoup as bs
    from urllib.request import urlopen
    
    page = urlopen('https://reseauactionclimat.org/quels-impacts-du-changement-climatique-sur-lagriculture/')
    soup = bs(page, 'html.parser')
    soup.title.string
    
    
    # In[21]:
    
    
    
    
    # In[4]:
    
    
    texte = soup.find('div', {'class':'fc'}).text
    
    
    # In[5]:
    
    
    from nltk.tokenize import PunktSentenceTokenizer
    
    
    # In[6]:
    
    
    tokenizer = PunktSentenceTokenizer()
    texte = tokenizer.tokenize(texte)
    
    
    # In[19]:
    
    
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    giec_agri_pred = TreebankWordDetokenizer().detokenize(texte[:10]),TreebankWordDetokenizer().detokenize(texte[11:14]) , texte[19]
    
    
    # In[20]:
    if st.button('Rapport du GIEC sur les impacts sur changement climatique sur les récoltes.'):
    
        st.write(giec_agri_pred)
    
    
    # In[39]:
    
    
    st.markdown("Les risques liés au réchauffement climatique sur les exploitations sont donc très importants. Ce qu'on peut retenir c'est que les déréglements climatiques (sécheresse, précipitations importantes et autres phénomènes météorlogiques extrêmes) entraînent une destruction des récoltes ou une modification des dates de récoltes") 