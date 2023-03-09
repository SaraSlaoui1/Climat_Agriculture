# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:31:23 2023

@author: Sara
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import geopandas as gpd
import zipfile
from pathlib import Path

def analyse_meteo_function():
    st.header('Analyse des variables météorologiques de 2018 à 2021')
    
    if not Path("Resources/meteo 2018 2021 (2).csv").is_file():
        with zipfile.ZipFile("Resources/meteo 2018 2021 (2).zip", 'r') as zip_ref:
            zip_ref.extractall("Resources")
    meteo_2018_2021 = pd.read_csv("Resources/meteo 2018 2021 (2).csv", sep=';', error_bad_lines = False)
    
    
    # In[9]:
    
    
    
    
    # In[10]:
    
    
    
    
    # In[11]:
    
    
    meteo_2018_2021['Date'] = pd.to_datetime(meteo_2018_2021['Date'], utc = True)
    
    
    # In[12]:
    
    
    meteo_2018_2021['region (name)'].value_counts()
    
    
    # In[13]:
    
    
    regions_to_remove = ['Guyane', 'Terres australes et antarctiques françaises', 'Guadeloupe', 'Saint-Pierre-et-Miquelon', 'Mayotte', 'La Réunion', 'Martinique']

    for region in regions_to_remove:
        meteo_2018_2021.drop(meteo_2018_2021[meteo_2018_2021['region (name)'] == region].index, inplace=True)
    
    
    # In[14]:
    
    
    meteo_2018_2021 = meteo_2018_2021.sort_values(by=['Date','region (name)'])
    meteo_2018_2021.reset_index(drop =True, inplace = True)
    
    
    # In[9]:
    
    
    
    
    # In[15]:
    
    
    meteo_2018_2021.rename({'Visibilité horizontale': 'Visibilité horizontale (en mètre)', 'region (name)' : 'nom'}, axis=1, inplace=True)
    
    
    # In[16]:
    
    
    
    
    # In[17]:
    
    
    columns_to_drop = meteo_2018_2021.columns[(meteo_2018_2021.notna().sum() < 81000) == True]
    meteo_2018_2021.drop(columns_to_drop, axis=1, inplace = True)
    
    
    # In[18]:
    
    
    
    
    # In[19]:
    
    
    meteo_2018_2021.drop(['Type de tendance barométrique.1','region (code)','communes (code)', 'mois_de_l_annee', 'EPCI (name)', 'EPCI (code)', 'Temps présent', 'Nom', 'communes (name)','ID OMM station' ], axis=1, inplace = True)
    
    
    # In[20]:
    
    
    
    
    # In[21]:
    
    
    meteo_2018_2021 = meteo_2018_2021.fillna(method="ffill")
    
    
    # In[22]:
    
    
    meteo_2018_2021['Latitude'].unique()
    
    
    # In[23]:
    
    
    'Deux outliers : -49.352333 et -66.663167'
    
    
    # In[24]:
    
    
    meteo_2018_2021 = meteo_2018_2021.query("Latitude != -49.352333")
    
    
    # In[25]:
    
    
    meteo_2018_2021 = meteo_2018_2021.query("Latitude != -66.663167")
    meteo_2018_2021.dropna(inplace=True)
    
    # In[26]:
    st.markdown('DataFrame meteo 2018 à 2021')
    
    st.write(meteo_2018_2021.head())

    
    # In[27]:
    
    
    
    
    # In[28]:
    
    
    meteo_2018_2021['rolling_avg_temp'] = meteo_2018_2021['Température (°C)'].rolling(window=5, min_periods=1).mean()
    
    
    # In[29]:
    
    st.markdown("En moyenne, l'année la plus froide fut 2021 et la plus chaude 2020. La plus pluvieuse 2018 et la plus sèche 2019.")

    shapefile = gpd.read_file("Resources/jsonvalidator.json")
    
    merged1 = shapefile.merge(meteo_2018_2021[meteo_2018_2021['Date'].dt.year == 2018].groupby('nom').mean(), on = 'nom')
    merged2 = shapefile.merge(meteo_2018_2021[meteo_2018_2021['Date'].dt.year == 2019].groupby('nom').mean(), on = 'nom')
    merged3 = shapefile.merge(meteo_2018_2021[meteo_2018_2021['Date'].dt.year == 2020].groupby('nom').mean(), on = 'nom')
    merged4 = shapefile.merge(meteo_2018_2021[meteo_2018_2021['Date'].dt.year == 2021].groupby('nom').mean(), on = 'nom')
    
    # In[30]:


    option = ['Température', 'Précipitations et humidité']
    
    select_option = st.selectbox('Caractéristiques météo 2018 à 2021', option)
    typegraph = st.multiselect('Type de Vizualisation', ['Scatterplot', 'Boxplot','Map'])
    if select_option == 'Température':
    
        if 'Scatterplot' in typegraph :
            plt.figure(figsize=(12,10))
            plt.plot_date(meteo_2018_2021['Date'], meteo_2018_2021['Température (°C)'])
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Temperature (°C)', fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=14)
        
            plt.title('Evolution Température (°C) ', fontsize=18)
            st.pyplot()
        if 'Boxplot'in typegraph:
            fig, axes = plt.subplots(2, 2, figsize=(20,24))
    
            sns.boxplot(x=meteo_2018_2021['Température (°C)'][meteo_2018_2021['Date'].dt.year == 2018], color = 'orange', ax = axes[0,0]).set(title='2018')
            sns.boxplot(x=meteo_2018_2021['Température (°C)'][meteo_2018_2021['Date'].dt.year == 2019], ax = axes[0,1]).set(title = '2019')
            sns.boxplot(x=meteo_2018_2021['Température (°C)'][meteo_2018_2021['Date'].dt.year == 2020], color= 'green', ax = axes[1,0]).set(title = '2020')
            sns.boxplot(x=meteo_2018_2021['Température (°C)'][meteo_2018_2021['Date'].dt.year == 2021], color = 'red', ax = axes[1,1]).set(title = '2021')
            plt.tick_params(axis='both', which='major', labelsize=14)
        
            plt.legend()
            st.pyplot();
            st.markdown("_Températures froides records en 2018 et 2021 et chaudes en 2019 et 2020. Dans le détail, on constate que la moyenne des températures est assez stable mais que l'année 2018 présente plusieurs valeurs de températures extrêmes froides et 2019 et 2020 de températures extrêmes chaudes. Les températures hivernales de l'année 2020 semblent plus douces, avec des valeurs n'allant pas au-delà de -5°._")
    
        if 'Map' in typegraph:
            fig, ax = plt.subplots(2,2, figsize=(20, 20))

            merged1.plot('Température (°C)', cmap='OrRd', ax=ax[0,0], legend=True)
            ax[0,0].set_axis_off()
            ax[0,0].set_title("Distribution spatiale des t° moyennes \n \n 2018")
            
            merged2.plot('Température (°C)', cmap='OrRd', ax=ax[0,1], legend=True)
            ax[0,1].set_axis_off()
            ax[0,1].set_title("2019")
            
            merged3.plot('Température (°C)', cmap='OrRd', ax=ax[1,0], legend=True)
            ax[1,0].set_axis_off()
            ax[1,0].set_title("2020")
            
            merged4.plot('Température (°C)', cmap='OrRd', ax=ax[1,1], legend=True)
            ax[1,1].set_axis_off()
            ax[1,1].set_title("2021")
            plt.show()

            st.pyplot()
            st.markdown("_En observant les moyennes des températures sur le territoire, l'année la plus chaude semble être 2020 et l'année la plus froide 2021. La répartition des températures est néanmoins inégale pour cette dernière année, les régions du Sud et l'Ile de France semblent montrer des températures bien plus élevées que le reste du pays._")
        
        
    elif select_option == 'Précipitations et humidité':
    
        if  'Scatterplot' in typegraph:
            
       
    
    # In[142]:
    
    
            plt.figure(figsize=(12,10))
            plt.plot_date(meteo_2018_2021['Date'], meteo_2018_2021['Précipitations dans les 24 dernières heures'])
            plt.title('Précipitations de 2018 à 2021')
            plt.tick_params(axis='both', which='major', labelsize=14)
        
            plt.legend()
            st.pyplot();
        
        
        # In[143]:
        
        if 'Boxplot' in typegraph:
            fig, axes = plt.subplots(2, 2, figsize=(20,24))
            
            sns.boxplot(x=meteo_2018_2021['Précipitations dans les 24 dernières heures'][meteo_2018_2021['Date'].dt.year == 2018], color = 'orange', ax = axes[0,0]).set(title='2018')
            sns.boxplot(x=meteo_2018_2021['Précipitations dans les 24 dernières heures'][meteo_2018_2021['Date'].dt.year == 2019], ax = axes[0,1]).set(title = '2019')
            sns.boxplot(x=meteo_2018_2021['Précipitations dans les 24 dernières heures'][meteo_2018_2021['Date'].dt.year == 2020], color= 'green', ax = axes[1,0]).set(title = '2020')
            sns.boxplot(x=meteo_2018_2021['Précipitations dans les 24 dernières heures'][meteo_2018_2021['Date'].dt.year == 2021], color = 'red', ax = axes[1,1]).set(title = '2021')
            plt.tick_params(axis='both', which='major', labelsize=14)
        
            plt.legend()
            st.pyplot();
            
            
            # In[144]:
            
            
            st.markdown("_L'année la plus pluvieuse est 2018, la plus sèche 2019. 2020 et 2021 sont relativement similaires._")
        
        
        # In[145]:
    
        
        
        if 'Map' in typegraph:
            fig, ax = plt.subplots(2,2, figsize=(20, 20))
            
            merged1.plot('Humidité', cmap='YlGnBu', ax=ax[0,0], legend=True)
            ax[0,0].set_axis_off()
            ax[0,0].set_title("Distribution spatiale de l'humidité moyenne \n \n 2018")
            
            merged2.plot('Humidité', cmap='YlGnBu', ax=ax[0,1], legend=True)
            ax[0,1].set_axis_off()
            ax[0,1].set_title("2019")
            
            merged3.plot('Humidité', cmap='YlGnBu', ax=ax[1,0], legend=True)
            ax[1,0].set_axis_off()
            ax[1,0].set_title("2020")
            
            merged4.plot('Humidité', cmap='YlGnBu', ax=ax[1,1], legend=True)
            ax[1,1].set_axis_off()
            ax[1,1].set_title("2021")
            plt.show()

            st.pyplot()
            fig, ax = plt.subplots(2,2, figsize=(20, 20))

            merged1.plot('Précipitations dans les 24 dernières heures', cmap='YlGnBu', ax=ax[0,0], legend=True)
            ax[0,0].set_axis_off()
            ax[0,0].set_title("Distribution spatiale des précipitations moyennes \n \n 2018")
            
            merged2.plot('Précipitations dans les 24 dernières heures', cmap='YlGnBu', ax=ax[0,1], legend=True)
            ax[0,1].set_axis_off()
            ax[0,1].set_title("2019")
            
            merged3.plot('Précipitations dans les 24 dernières heures', cmap='YlGnBu', ax=ax[1,0], legend=True)
            ax[1,0].set_axis_off()
            ax[1,0].set_title("2020")
            
            merged4.plot('Précipitations dans les 24 dernières heures', cmap='YlGnBu', ax=ax[1,1], legend=True)
            ax[1,1].set_axis_off()
            ax[1,1].set_title("2021")
            
            plt.show()
            st.pyplot()
                
            st.markdown("_Les données visualisées sur la carte confirment bien que 2018 était une année pluvieuse et 2019 une année plutôt sèche(à l'exception de l'ouest). On constate néanmoins que la répartition des précipitations est différente entre 2020 et 2021. En 2020, il a plu majoritairement dans l'ouest tandis qu'en 2021, les précipitations sont plus homogènes sur le territoire, à l'exception de l'Ile de France, où les pluies ont été abondantes._")
            
        