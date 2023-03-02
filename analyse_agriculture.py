# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:29:41 2023

@author: Sara
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 22})

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
    data_dict = {"2018-2019": (prod_vege_2018, prod_vege_2019),
                 "2019-2020": (prod_vege_2019, prod_vege_2020),
                 "2020-2021": (prod_vege_2020, prod_vege_2021)}
    barWidth = 0.4
        
        # Define the text content for each graph
    text_dict = {
             "2018-2019": "_Observations : Légère augmentation globale des différentes cultures sauf maïs._",
             "2019-2020": "_Observations : Baisse globale des productions particulièrement des betteraves industrielles en Ile de France. Seule exception : augmentation de la production de maïs au Pays-de-la-Loire et Bretagne._",
            "2020-2021": "_Observations : Légère augmentation globale des productions, plus marquée pour la production de maïs en Bretagne._",
            }
        
        # Create the select boxes for the region and year range
    regions = ["France métropolitaine", "Occitanie", "Auvergne-Rhône-Alpes", "Provence-Alpes-Côte-d'Azur", 
                   "Bretagne", "Hauts de France", "Ile-de-France", "Pays-de-la-Loire", "Nouvelle-Aquitaine", 
                   "Centre-Val de Loire", "Grand Est", "Normandie", "Bourgogne-Franche-Comté"]
    region = st.selectbox("Région", regions)
    year_range = st.selectbox("Années", list(data_dict.keys()))
        
        # Get the data for the selected year range
    data = data_dict[year_range]
        
        # Set the x-axis labels and positions
    x = np.arange(len(data[0][('','Cultures')]))
        
        # Create the bar chart
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(x, data[0].loc[:, (region, 'Production(1000 t)')], width=barWidth,label=f'{year_range.split("-")[0]}-{year_range.split("-")[1]}')
    ax.bar(x + barWidth, data[1].loc[:, (region, 'Production(1000 t)')], width=barWidth,label=f'{year_range.split("-")[1]}-{year_range.split("-")[1]}')
    ax.set_title(f'Productions grandes cultures en 1000T')
    ax.legend()
    plt.xticks(x + barWidth/2, data[0][('', 'Cultures')].unique(), rotation=90)
        
        # Get the text content for the selected graph
    text = text_dict.get((year_range), "")
        
        # Display the chart with the markdown content
    st.markdown(text)
    st.pyplot(fig)
        
    
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