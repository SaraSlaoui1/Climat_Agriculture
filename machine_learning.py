# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:34:05 2023

@author: Sara
"""
#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import zipfile
from pathlib import Path


def machine_learning_function():    
    #st.title("Impact du changement climatique sur l'agriculture française")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    prod_vege_2018 = pd.read_csv("Resources/2018_donneesgrandescultures.csv", sep=';', header = [0,1])
    prod_vege_2018.iloc[:,1:] = prod_vege_2018.iloc[:,1:].astype(float)
    prod_vege_2018.rename({'Unnamed: 0_level_0':''}, axis=1, inplace = True)
    prod_vege_2018.iloc[:,1:] = prod_vege_2018.iloc[:,1:].round(2)
    
    prod_vege_2018.fillna(0, inplace = True)
    prod_vege_2018 = prod_vege_2018.sort_values(by=('', 'Cultures'))
    prod_vege_2018.reset_index(drop=True, inplace=True)
    prod_vege_2018.insert(0, "Année", '2018')
    prod_vege_2018['Année'] = prod_vege_2018['Année'].apply(lambda x: pd.to_datetime(str(x),format='%Y%'))
    
    
    
 
    prod_vege_2019 = pd.read_csv("Resources/2019_donneesgrandescultures.csv", sep=';', header = [0,1])
    
    prod_vege_2019.iloc[:,1:] = prod_vege_2019.iloc[:,1:].astype(float)
    prod_vege_2019.rename({'Unnamed: 0_level_0':''}, axis=1, inplace = True)
    prod_vege_2019.iloc[:,1:] = prod_vege_2019.iloc[:,1:].round(2)
    
    prod_vege_2019.info()
    
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
    
    prod_vege_2020.info()
    
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
    
    prod_vege_2021.info()
    
    prod_vege_2021.fillna(0, inplace=True)
    
    prod_vege_2021.replace({'Rendement(q/h)': 'Rendement(q/ha)'}, inplace=True)
    
    prod_vege_2021 = prod_vege_2021.sort_values(by=('', 'Cultures'))
    prod_vege_2021.reset_index(drop=True, inplace=True)
    prod_vege_2021.insert(0, "Année", '2021')
    prod_vege_2021['Année'] =prod_vege_2021['Année'].apply(lambda x: pd.to_datetime(str(x),format='%Y%'))
    
    
    
    
    
    
    # In[9]:
    
    
    moyenne_2019 = prod_vege_2019.iloc[:,2:].drop(prod_vege_2019.index[5]).mean()
    moyenne_2018 = prod_vege_2018.iloc[:,2:].drop(prod_vege_2018.index[5]).mean()
    ecart_production_2018_2019 = (moyenne_2019 - moyenne_2018) / moyenne_2019
    prod_vege_2018.iloc[5,2:] = prod_vege_2018.iloc[5,2:].replace([prod_vege_2018.iloc[5,2:].values], [np.array(prod_vege_2019.iloc[5,2:] - (moyenne_2019 *ecart_production_2018_2019))])
    
    
    # In[10]:
    
    
    
    # In[20]:
    
    
    moyenne_2018 = prod_vege_2018.iloc[:, prod_vege_2018.columns.get_level_values(1)=='Production(1000 t)'].mean()
    moyenne_2019 = prod_vege_2019.iloc[:, prod_vege_2019.columns.get_level_values(1)=='Production(1000 t)'].mean()
    
    ecart_production_2018_2019 = pd.DataFrame((moyenne_2019 - moyenne_2018) / moyenne_2019, columns = ['ecart_production_2018_2019'])
    #ecart_production_2018_2019
    
    
    # In[21]:
    
    
    moyenne_2020 = prod_vege_2020.iloc[:, prod_vege_2020.columns.get_level_values(1)=='Production(1000 t)'].mean()
    ecart_production_2019_2020 = pd.DataFrame((moyenne_2020 - moyenne_2019) / moyenne_2020, columns = ['ecart_production_2019_2020'])
    #ecart_production_2019_2020
    
    
    # In[22]:
    
    
    moyenne_2021 = prod_vege_2021.iloc[:, prod_vege_2021.columns.get_level_values(1)=='Production(1000 t)'].mean()
    ecart_production_2020_2021 = pd.DataFrame((moyenne_2021 - moyenne_2020) / moyenne_2021, columns = ['ecart_production_2020_2021'])
    #ecart_production_2020_2021
    
    
    # In[23]:
    
    
    prod_2018_2019 = pd.DataFrame(ecart_production_2018_2019.values.reshape(14,1), columns = ['ecart production 2018-2019'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    prod_2019_2020 = pd.DataFrame(ecart_production_2019_2020.values.reshape(14,1), columns = ['ecart production 2019-2020'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    prod_2020_2021 = pd.DataFrame(ecart_production_2020_2021.values.reshape(14,1), columns = ['ecart production 2020-2021'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    
    
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
    
    rend_2018_2019 = pd.DataFrame(ecart_rendement_2018_2019.values.reshape(14,1), columns = ['ecart rendement 2018-2019'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    rend_2019_2020 = pd.DataFrame(ecart_rendement_2019_2020.values.reshape(14,1), columns = ['ecart rendement 2019-2020'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    rend_2020_2021 = pd.DataFrame(ecart_rendement_2020_2021.values.reshape(14,1), columns = ['ecart rendement 2020-2021'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    
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
    
    surf_2018_2019 = pd.DataFrame(ecart_surface_2018_2019.values.reshape(14,1), columns = ['ecart surface 2018-2019'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    surf_2019_2020 = pd.DataFrame(ecart_surface_2019_2020.values.reshape(14,1), columns = ['ecart surface 2019-2020'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    surf_2020_2021 = pd.DataFrame(ecart_surface_2020_2021.values.reshape(14,1), columns = ['ecart surface 2020-2021'], index = prod_vege_2018.columns[2:].get_level_values(0).unique())
    
    variations_surf_2018_2021 = pd.concat([surf_2018_2019, surf_2019_2020, surf_2020_2021], axis = 1)
    
    variations_surf_2018_2021 = (variations_surf_2018_2021 * 100).round(2).astype(str) + '%'
    
    
 
    
    
    
    # In[23]:
    if not Path("Resources/phenologie blé 2018 2021(3).csv").is_file():
        with zipfile.ZipFile("Resources/phenologie blé 2018 2021(3).zip", 'r') as zip_ref:
            zip_ref.extractall("Resources")
     
    
    pheno_ble = pd.read_csv("Resources/phenologie blé 2018 2021(3).csv", error_bad_lines = False, sep = ';', encoding="ISO-8859-1")
    
    
    

    
    # In[425]:
    
    
    pheno_ble.drop(['kingdom', 'data_source', 'scale', 'genus', 'binomial_name'], axis=1, inplace=True)
    
    
    # In[426]:
    
    
    
    
    # In[427]:
    
    
    pheno_ble.drop_duplicates(inplace=True)
    
    
    # In[428]:
    
    
    pheno_ble.reset_index(drop=True, inplace=True)
    
    
  
    
    # In[430]:
    
    
    pheno_ble['date'] = pd.to_datetime(pheno_ble['date'],format='%d/%m/%Y', utc = True)
    
    

    
    
    # In[432]:
    
    
    pheno_ble.sort_values(by=['site_id','date'], inplace=True)
    
    
    # In[433]:
    
    
    pheno_ble.reset_index(drop=True, inplace=True)
    
    
    # In[434]:
    
    
    new_names = {'Centre':'Centre-Val de Loire','Languedoc-Roussillon':'Occitanie', 'Nord-Pas-de-Calais': 'Hauts-de-France', 'Limousin':'Nouvelle-Aquitaine','Poitou-Charentes':'Nouvelle-Aquitaine','Franche-Comté':'Bourgogne-Franche-Comté', 'Bourgogne':'Bourgogne-Franche-Comté','Auvergne':'Auvergne-Rhône-Alpes', 'Rhône-Alpes':'Auvergne-Rhône-Alpes','Champagne-Ardenne':'Grand Est','Alsace':'Grand Est','Midi-Pyrénées':'Occitanie', 'Picardie':'Hauts-de-France','Lorraine':'Grand Est', 'Aquitaine': 'Nouvelle-Aquitaine'}
    pheno_ble = pheno_ble.replace(new_names)
    
    
    # In[435]:
    
    
    pheno_ble = pheno_ble.rename({'species':'taxon'}, axis=1)
    
    

    
    # In[437]:
    
    
    stage_description = pheno_ble[pheno_ble['phenological_main_event_code'].isin(pheno_ble['phenological_main_event_code'].unique())][['phenological_main_event_code', 'phenological_main_event_description','stage_code', 'stage_description' ]].drop_duplicates()
    
    
    # In[438]:
    
    
    stage_description = stage_description.sort_values(by=['phenological_main_event_code', 'stage_code']).set_index('phenological_main_event_code')
    
    
 
    # In[447]:
    
    
    percent_category_2018 = []
    for i in pheno_ble[pheno_ble['year']== 2018]['phenological_main_event_code'].value_counts():
        percent_category_2018.append((i/len(pheno_ble[pheno_ble['year']== 2018]['phenological_main_event_code']))*100)
    percent_category_2018 = pd.DataFrame(np.array(percent_category_2018), columns = ['phenological_main_event_code 2018'], index = pheno_ble[pheno_ble['year']== 2018]['phenological_main_event_code'].value_counts().index)
    percent_category_2018 = percent_category_2018.round(2).astype(str) + '%'
    
    
    
    
    # In[449]:
    
    
    rend2018 = pd.DataFrame(prod_vege_2018.iloc[6:12, prod_vege_2018.columns.get_level_values(1)=='Rendement(q/ha)'].mean(), columns = ['2018'])
    
    

    
    
    # In[451]:
    
    
    percent_category_2019 = []
    for i in pheno_ble[pheno_ble['year']== 2019]['phenological_main_event_code'].value_counts():
        percent_category_2019.append((i/len(pheno_ble[pheno_ble['year']== 2019]['phenological_main_event_code']))*100)
    percent_category_2019 = pd.DataFrame(np.array(percent_category_2019), columns = ['phenological_main_event_code 2019'], index = pheno_ble[pheno_ble['year']== 2019]['phenological_main_event_code'].value_counts().index)
    percent_category_2019 = percent_category_2019.round(2).astype(str) + '%'
    
    
    # In[452]:
    
    
   
    
    
    # In[453]:
    
    
    rend2019 = pd.DataFrame(prod_vege_2019.iloc[6:12, prod_vege_2019.columns.get_level_values(1)=='Rendement(q/ha)'].mean(), columns = ['2019'])
    
    
  
    
    
    # In[455]:
    
    
    percent_category_2020 = []
    for i in pheno_ble[pheno_ble['year']== 2020]['phenological_main_event_code'].value_counts():
        percent_category_2020.append((i/len(pheno_ble[pheno_ble['year']== 2020]['phenological_main_event_code']))*100)
    percent_category_2020 = pd.DataFrame(np.array(percent_category_2020), columns = ['phenological_main_event_code 2020'], index = pheno_ble[pheno_ble['year']== 2020]['phenological_main_event_code'].value_counts().index)
    percent_category_2020 = percent_category_2020.round(2).astype(str) + '%'
    
    
   
    
    
    # In[457]:
    
    
    rend2020 = pd.DataFrame(prod_vege_2020.iloc[6:12, prod_vege_2020.columns.get_level_values(1)=='Rendement(q/ha)'].mean(), columns = ['2020'])
    
    
    
    
    # In[459]:
    
    # In[8]:
    #st.header('Analyse des variables météorologiques de 2018 à 2021')
    if not Path("Resources/meteo 2018 2021 (2).csv").is_file():
        with zipfile.ZipFile("Resources/meteo 2018 2021 (2).zip", 'r') as zip_ref:
            zip_ref.extractall("Resources")
    meteo_2018_2021 = pd.read_csv("Resources/meteo 2018 2021 (2).csv", sep=';', error_bad_lines = False)
    st.dataframe(meteo_2018_2021)
    
    
    
    
    # In[11]:
    
    
    meteo_2018_2021['Date'] = pd.to_datetime(meteo_2018_2021['Date'], utc = True)
    
    
    # In[12]:
    
    
    
    
    # In[13]:
    
    
    meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Guyane'].index, inplace=True)
    meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Terres australes et antarctiques françaises'].index, inplace=True)
    meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Guadeloupe'].index, inplace=True)
    meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Saint-Pierre-et-Miquelon'].index, inplace=True)
    meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Mayotte'].index, inplace=True)
    meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'La Réunion'].index, inplace=True)
    meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == 'Martinique'].index, inplace=True)
    
    
    # In[14]:
    
    
    meteo_2018_2021 = meteo_2018_2021.sort_values(by=['Date','region (name)'])
    meteo_2018_2021.reset_index(drop =True, inplace = True)
    
    
 
    
    
    # In[15]:
    
    
    meteo_2018_2021.rename({'Visibilité horizontale': 'Visibilité horizontale (en mètre)', 'region (name)' : 'nom'}, axis=1, inplace=True)
    
    
    
    
    
    # In[17]:
    
    
    columns_to_drop = meteo_2018_2021.columns[(meteo_2018_2021.notna().sum() < 81000) == True]
    meteo_2018_2021.drop(columns_to_drop, axis=1, inplace = True)
    
    
    
    
    
    # In[19]:
    
    
    meteo_2018_2021.drop(['Type de tendance barométrique.1','region (code)','communes (code)', 'mois_de_l_annee', 'EPCI (name)', 'EPCI (code)', 'Temps présent', 'Nom', 'communes (name)','ID OMM station' ], axis=1, inplace = True)
    
    
  
    
    
    # In[21]:
    
    
    meteo_2018_2021 = meteo_2018_2021.fillna(method="ffill")
    
    
 
    
    # In[23]:
    
    
    'Deux outliers : -49.352333 et -66.663167'
    
    
    # In[24]:
    
    
    meteo_2018_2021 = meteo_2018_2021.query("Latitude != -49.352333")
    
    
    # In[25]:
    
    
    meteo_2018_2021 = meteo_2018_2021.query("Latitude != -66.663167")
    
    
    
    
    
    meteo_2018_2021['rolling_avg_temp'] = meteo_2018_2021['Température (°C)'].rolling(window=5, min_periods=1).mean()
    
    
  
    
    # In[460]:
    st.header("Prédictions pour L'année manquante 2021")
    
    train_data_pheno = pheno_ble.query("year != 2021")
    train_data_pheno = pheno_ble[['stage_code', 'date','grid_label','site_latitude','site_longitude','phenological_main_event_code']]
    
    
    # In[461]:
    
    
    train_data_meteo = meteo_2018_2021.drop(['department (name)','department (code)', 'Coordonnees','Altitude'], axis=1)
    
    
    # In[462]:
    
    
    train_data_meteo.head()
    
    
   
    
    
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
    
    st.markdown("Merging de météo et phénologie sans l'année 2021 et en supprimant les variables latitude, longitude")
    st.write(train_data.head())
    
    
    # In[471]:
    if st.button('Table de corrélation'):
    
        st.write(train_data.corr())
    
    
    # In[472]:
    
    
    
    
    
    # In[473]:
    
    
    st.markdown('**Selon le tableau de corrélation, la variable météorologique la plus corrélée au stade de développement du blé est la température**')
    
    
    # In[474]:
    st.markdown('Test Chi-2 pour déterminer la corrélation entre une variable catégorielle et une variable numérique continue')
    variable = st.selectbox('Variable catégorielle',['Température (°C)','rolling_avg_temp','Précipitations dans les 24 dernières heures','Humidité'])
    if variable == 'Température (°C)':
        from scipy.stats import chi2_contingency
        st.write("Hypothèse H0 : il n'y a pas d'influence de la variable 'Température (°C)' sur les stades de croissance du blé 'phenological_main_event_code'")
        
        table = pd.crosstab(train_data['phenological_main_event_code'], train_data['Température (°C)'])
        
        chi2, p, dof, expected = chi2_contingency(table)
        
        st.write('chi2 = ',chi2) 
        st.write('p = ', p)  
        st.write('p-value inférieure à 0.05, donc H0 réfutée')
        
    
    # In[475]:
    
    if variable == 'rolling_avg_temp' : 
        from scipy.stats import chi2_contingency
        st.write("Hypothèse H0 : il n'y a pas d'influence de la variable 'rolling_avg_temp' sur les stades de croissance du blé 'phenological_main_event_code'")
        
        table = pd.crosstab(train_data['phenological_main_event_code'], train_data['rolling_avg_temp'])
        
        chi2, p, dof, expected = chi2_contingency(table)
        
        st.write('chi2 = ',chi2) 
        st.write('p = ', p)  
        st.write('p-value inférieure à 0.05, donc H0 réfutée')
    
    
    # In[476]:
    
    if variable == 'Précipitations dans les 24 dernières heures' :
        st.write("Hypothèse H0 : il n'y a pas d'influence de la variable 'Précipitations dans les 24 dernières heures' sur les stades de croissance du blé 'phenological_main_event_code'")
        
        table = pd.crosstab(train_data['phenological_main_event_code'], train_data['Précipitations dans les 24 dernières heures'])
        
        chi2, p, dof, expected = chi2_contingency(table)
        
        st.write('chi2 = ',chi2) 
        st.write('p = ', p)  
        st.write("p-value inférieure à 0.05, donc H0 réfutée, la statistique chi-2 étant plus faible que pour le test avec les températures, on en déduit que cette variable joue moins d'importance")
        
        
    # In[477]:
    
    
    if variable == 'Humidité':
        st.write("Hypothèse H0 : il n'y a pas d'influence de la variable 'Humidité' sur les stades de croissance du blé 'phenological_main_event_code'")
        
        table = pd.crosstab(train_data['phenological_main_event_code'], train_data['Humidité'])
        
        chi2, p, dof, expected = chi2_contingency(table)
        
        st.write('chi2 = ',chi2) 
        st.write('p = ', p)  
        st.write('p-value inférieure à 0.05, donc H0 réfutée')
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
    
    
    #train_data.columns
    
    
    # In[483]:
    
    
    data = train_data.drop(['stage_code','phenological_main_event_code'], axis=1)
    target = train_data['phenological_main_event_code']
    
    
    # In[484]:
    
    
    data.head()
    
    
    # In[485]:
    
    
    test_2021 = meteo_2018_2021[meteo_2018_2021['Date'].dt.year == 2021].drop(['Pression au niveau mer','Variation de pression en 3 heures','Type de tendance barométrique','Direction du vent moyen 10 mn','Vitesse du vent moyen 10 mn', 'Point de rosée','Visibilité horizontale (en mètre)',"Nebulosité totale","Nébulosité  des nuages de l' étage inférieur","Hauteur de la base des nuages de l'étage inférieur",'Pression station','Variation de pression en 24 heures', 'Rafale sur les 10 dernières minutes','Rafales sur une période','Periode de mesure de la rafale','Etat du sol','Hauteur totale de la couche de neige, glace, autre au sol', 'Nébulosité couche nuageuse 1','Hauteur de base 1','Nébulosité couche nuageuse 2','Hauteur de base 2', 'Température','Température minimale sur 12 heures','Température maximale sur 12 heures','Température minimale du sol sur 12 heures', 'Coordonnees','department (name)','department (code)','Altitude'], axis=1)
    test_2021 = test_2021.loc[train_data_meteo['Température minimale sur 12 heures (°C)'].notna(), :]
    
    
    # In[486]:
    
    
    #test_2021.shape
    
    
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
    
    
    #test_2021.shape
    
    
    # In[496]:
    
    
    test_2021.drop_duplicates(inplace=True)
    st.subheader("Recherche de l'algorithme le plus efficient pour prédire les stades de croissance du blé de 2021")
    
    # In[497]:
    
    
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score
    from sklearn.model_selection import train_test_split
    
    
    # In[498]:
   
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 123)
    
    X_train.sort_index(axis=1, inplace=True)
    test_2021.sort_index(axis=1, inplace=True)
    X_test.sort_index(axis=1, inplace=True)
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
    
    "Best parameters RF: {}".format(grid_search.best_params_)
    "Best score RF: {:.2f}".format(grid_search.best_score_)
    test_2021_scaled = scaler.transform(test_2021)

    
    # In[501]:
    
    alg = st.selectbox('Algorithme de Classification', ['Random Forest','Gradient Boosting','Decision Tree','KNN'])
    if alg == 'Random Forest':
        rf = RandomForestClassifier(max_depth =  10, min_samples_split = 10, n_estimators = 50)
        
        
        # In[502]:
        
        
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)
        st.write(pd.crosstab(y_pred_rf, y_test))
        st.write('score RF : ', rf.score(X_test_scaled, y_test))
        
        
        # In[503]:
        
        
        features = data.columns
        features_importance = {}
        sorted_features = {}
        for x,j in zip(features, rf.feature_importances_):
            features_importance[x] = j
        sorted_features = sorted(features_importance.items(), key=lambda x:x[1], reverse=True) 
        print(sorted_features[:8])
        
        
        # In[504]:
        
        
        precision_score(y_pred_rf, y_test, average = "weighted")
        
        
        # In[505]:
        
        
        
        st.write(f'precision score RF : {precision_score(y_pred_rf, y_test, average = "weighted")}')
        
        
        # In[506]:
        
        
        test_2021_scaled = scaler.transform(test_2021)
        
        
        # In[507]:
        
        
        y_pred_2021_rf = rf.predict(test_2021_scaled)
        st.write('prédictions 2021 RF : ', np.unique(y_pred_2021_rf))
        

    # In[508]:
    if alg == 'Gradient Boosting':
    
        from sklearn.ensemble import GradientBoostingClassifier
        
        gb = GradientBoostingClassifier(n_estimators = 100, max_depth = 7, learning_rate = 0.01, subsample = 1)
        
        
        # In[509]:
        
        
        gb.fit(X_train_scaled, y_train)
        y_pred_gb = gb.predict(X_test_scaled)
        st.write(pd.crosstab(y_pred_gb, y_test))
        st.write('score GB : ', gb.score(X_test_scaled, y_test))
        
        
        # In[510]:
        
        
        features = data.columns
        features_importance = {}
        sorted_features = {}
        for x,j in zip(features, gb.feature_importances_):
            features_importance[x] = j
        sorted_features = sorted(features_importance.items(), key=lambda x:x[1], reverse=True) 
        print(sorted_features[:8])
        
        
        # In[511]:
        
        
        print(f'precision score GB: {precision_score(y_pred_gb, y_test, average = "weighted")}')
        
        
        # In[512]:
        
        
        y_pred_2021_gb = gb.predict(test_2021_scaled)
        st.write('prédictions 2021 GB : ', np.unique(y_pred_2021_gb))
        
        
    # In[513]:
    if alg == 'Decision Tree':
    
        #clf = DecisionTreeClassifier()
        #param_grid = {'max_depth': range(1,10),
         #             'min_samples_split': [2, 3, 4, 5, 6, 7],
          #            'criterion': ['gini','entropy'],
           #           }
        
        #grid_search = GridSearchCV(clf, param_grid, cv=8, scoring='accuracy')
        
        #grid_search.fit(X_train_scaled, y_train)
        #print(grid_search.best_params_)
        
        #print(grid_search.best_score_)
        #print(grid_search.best_estimator_)
        
        
        # In[524]:
        
        
        clf_entr = DecisionTreeClassifier(criterion = 'entropy', max_depth = 9,min_samples_split = 7)
        clf_entr.fit(X_train_scaled, y_train)
        y_pred = clf_entr.predict(X_test_scaled)
        st.write(pd.crosstab(y_pred, y_test))
        
        
        # In[525]:
        
        
        st.write('accuracy score DT : ', clf_entr.score(X_test_scaled, y_test))
        
        
        # In[526]:
        
        
        features = data.columns
        features_importance = {}
        sorted_features = {}
        for x,j in zip(features, clf_entr.feature_importances_):
            features_importance[x] = j
        sorted_features = sorted(features_importance.items(), key=lambda x:x[1], reverse=True) 
        print(sorted_features[:8])
        
        
        # In[527]:
        
        
        st.write(f'precision score DT : {precision_score(y_pred, y_test, average = "weighted")}')
        
        
        # In[528]:
        
        
        y_pred_2021 = clf_entr.predict(test_2021_scaled)
        
        
        # In[529]:
        
        
        st.write('prédictions 2021 DT : ',np.unique(y_pred_2021))
    
    
    # In[520]:
    if alg == 'KNN':
    
        from sklearn.neighbors import KNeighborsClassifier
        
        knn = KNeighborsClassifier(n_neighbors=5)
        
        knn.fit(X_train_scaled, y_train)
        
        y_pred_knn = knn.predict(X_test_scaled)
        
        st.write("Accuracy score KNN: {:.2f}".format(knn.score(X_test_scaled, y_test)))
        
        st.write(pd.crosstab(y_pred_knn, y_test))

        # In[521]:
        
        
        st.write(f'precision score KNN: {precision_score(y_pred_knn, y_test, average = "weighted")}')
        
        
        # In[522]:
        
        
        y_pred_knn_2021 = knn.predict(test_2021_scaled)
        st.write('prédictions 2021 KNN : ',np.unique(y_pred_knn_2021))
    
    
    # In[523]:
    
    
    st.markdown('Modèle le plus performant : Gradient Boosting')
    
    
    # In[530]:
    from sklearn.ensemble import GradientBoostingClassifier
    
    gb = GradientBoostingClassifier(n_estimators = 100, max_depth = 7, learning_rate = 0.01, subsample = 1)
    
    gb.fit(X_train_scaled, y_train)
    y_pred_gb = gb.predict(X_test_scaled)
    y_pred_2021_gb = gb.predict(test_2021_scaled)


    test_2021['phenological_main_event_code'] = y_pred_2021_gb
    
    
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
    
    
  
    
    # In[301]:
    
    
    # In[302]:
    
    
    percent_category_2021 = []
    for i in pheno_meteo[pheno_meteo['year']== 2021]['phenological_main_event_code'].value_counts():
        percent_category_2021.append((i/len(pheno_meteo[pheno_meteo['year']== 2021]['phenological_main_event_code']))*100)
    percent_category_2021 = pd.DataFrame(np.array(percent_category_2021), columns = ['phenological_main_event_code 2021'], index = pheno_meteo[pheno_meteo['year']== 2021]['phenological_main_event_code'].value_counts().index)
    percent_category_2021 = percent_category_2021.round(2).astype(str) + '%'
    
    
    # In[303]:
    
    
    rend2021 = pd.DataFrame(prod_vege_2021.iloc[6:12, prod_vege_2021.columns.get_level_values(1)=='Rendement(q/ha)'].mean(), columns = ['2021'])
    
    
    # In[304]:
    button = st.button('Comparaison de la distribution des différents stades de croissance et du rendement pour 2021')
    if button :     
        st.write(percent_category_2021)
        st.write(rend2021.iloc[0,:])
        
    
    # In[308]:
    
    
    st.markdown("**Le stade 3 représente une partie relativement faible des stades observés** (de très peu majoritaire), ce qui peut expliquer **un rendement moins important** pour cette année. Ce sont des prédictions avec plus de dates que pour les autres années, c'est pourquoi la répartition des stades diffère des années précemment étudiées.")
    
    
    # In[73]:
    st.header('Prédictions pour 2100')
    
    page = urlopen('https://www.linfodurable.fr/environnement/38-degres-en-2100-rechauffement-climatique-pire-que-prevu-en-france-34833')
    soup = bs(page, 'html.parser')
    
    
    # In[74]:
    
    
    
    # In[76]:
    
    st.subheader("Prédictions des stades phénologiques du blé en 2100 à partir des prédictions sur les températures du rapport du GIEC pour l'année 2100")
    texte1 = soup.find('h2',{'class':'font-medium fs-20 node-20'}).text.strip()
    
    
    # In[77]:
    
    
    tokenizer = PunktSentenceTokenizer()
    texte = tokenizer.tokenize(texte1)
    
    
    # In[78]:
    
    
    texte2 = tokenizer.tokenize(soup.find('div', {'class':'clearfix text-formatted field field--name-field-article-body field--type-text-with-summary field--label-hidden field__item'}).text.strip())
    
    
    # In[84]:
    
    
    #texte2
    
    
    # In[80]:
    
    
    pred_giec_meteo = texte,TreebankWordDetokenizer().detokenize(texte2[10:18])
    
    pred_giec_meteo = str(pred_giec_meteo)
    
    pred_giec_meteo
    
    import re
    pred_giec_meteo = re.sub(r"\\", "", pred_giec_meteo)
    button = st.button('**Rapport GIEC météo**')
    if button :
        st.write(pred_giec_meteo)
    
    
    # In[235]:
    
    
    st.markdown("Nous allons maintenant utiliser les informations du rapport du GIEC concernant les prédictions de hausses de températures. En 2100 les températures augmenteront en moyenne de 3.8°C, mais en hiver de 3.2°C et en été de 5.1°C. Nous allons donc augmenter en conséquence les températures de notre dataset météo et prédire les stades de croissance du blé.")
    
    
    # In[236]:
    
    
    pheno_meteo.head()
    
    
    # In[237]:
    
    
    st.markdown("Pour réaliser les prédictions, prenons comme référence l'année 2018. C'est celle pour laquelle nous avons le plus de données (du dataframe pheno_ble).")
    
    
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
    
    pheno_meteo_pred_train.sort_index(axis=1, inplace=True)
    # In[315]:
    
    
    pheno_meteo_pred_train_scaled = scaler.transform(pheno_meteo_pred_train)
    
    
    # In[316]:
    st.markdown("**Algorithme Decision Tree (utilisé précedemment) pour prédire les stades de l'année 2100**")

    gb = GradientBoostingClassifier(n_estimators = 100, max_depth = 7, learning_rate = 0.01, subsample = 1)
    
    gb.fit(X_train_scaled, y_train)
    y_pred_2100 = gb.predict(pheno_meteo_pred_train_scaled)
    
    
    # In[317]:
    
    
    st.write(f'**Stades prédits : {np.unique(y_pred_2100)}**')
    
    
    # In[318]:
    
    
    pheno_meteo_pred_train['phenological_main_event_code'] = y_pred_2100
    
    
    # In[319]:
    
    
    pheno_meteo_pred_train.head()
    
    
    # In[320]:
    
    
    pheno_meteo_pred_train['régions'] = pheno_meteo_pred_train[['Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté','Bretagne','Centre-Val de Loire','Grand Est','Hauts-de-France','Normandie','Nouvelle-Aquitaine','Occitanie','Pays de la Loire',"Provence-Alpes-Côte d'Azur",'Île-de-France']].idxmax(axis=1)
    pheno_meteo_pred_train.drop(['Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté','Bretagne','Centre-Val de Loire','Grand Est','Hauts-de-France','Normandie','Nouvelle-Aquitaine','Occitanie','Pays de la Loire',"Provence-Alpes-Côte d'Azur",'Île-de-France'], axis=1, inplace=True)
    pheno_meteo_pred_train['date'] = pd.to_datetime(pheno_meteo_pred_train[['year', 'month', 'day']])
    
    
    
    # In[321]:
    
    
    percent_category_2100 = []
    for i in pheno_meteo_pred_train['phenological_main_event_code'].value_counts():
        percent_category_2100.append((i/len(pheno_meteo_pred_train['phenological_main_event_code']))*100)
    percent_category_2100 = pd.DataFrame(np.array(percent_category_2100), columns = ['phenological_main_event_code 2100'], index = pheno_meteo_pred_train['phenological_main_event_code'].value_counts().index)
    percent_category_2100 = percent_category_2100.round(2).astype(str) + '%'
    
    
    # In[322]:
    st.markdown('Comparaison des distributions des années 2018 et 2100 des stades de croissance du blé.')
    dist = st.selectbox('Année',['2018','2100'])
    if dist == '2018':
        st.write(percent_category_2018)
    
    
    # In[323]:
    
    if dist == '2100':
        st.write(percent_category_2100)
    
    
    # In[251]:
    
    
    st.markdown("Comme démontré précedemment, le stade déterminant à une bonne récolte est le stade de montaison à 1cm d'épi qui correspond au stade 3. On en compte moins en 2100 qu'en 2018. On peut donc supposer que selon ces prédictions, le rendement aura tendance à être plus faible en 2100.")
    
    
    # In[252]:
   
    
    
    
    
