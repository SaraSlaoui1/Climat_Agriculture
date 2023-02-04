# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:23:44 2023

@author: Sara

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
def phenologie_ble_function():
    st.header('Analyse phénologique du blé de 2018 à 2021')
    
    
    
    # In[41]:
    
    
    page = urlopen('https://vert-lavenir.com/ble/')
    soup = bs(page, 'html.parser', from_encoding='utf-8')
    
    
    # In[22]:
    
    
    #soup.findAll("div")
    
    
    # In[43]:
    
    
    texts = soup.find_all('p')
    texte = []
    for text in texts:
        texte.append(text.get_text())
    
    
 
  
    # In[45]:
    
    
    to_remember = texte[4:8]
    
    
    # In[46]:
    
    
    to_remember = []
    for i in texte[4:8]:
        to_remember.append(i.replace(u'\xa0', u' '))
    
    
    # In[47]:
    
    
    st.markdown(to_remember)
    
    
    # In[48]:
    
    
    st.markdown("Pour bien comprendre les récoltes de blé et les impacts de la météo sur ceux-ci nous devons nous pencher sur la phénologie du blé. C'est à dire, l'étude du cycle de vie de la plante. Ainsi, nous pourrons constater d'une éventuelle modification de la croissance du végétal et donc des récoltes")
    
    
    # In[23]:
    
    if not Path("Resources/phenologie blé 2018 2021(3).csv").is_file():
        with zipfile.ZipFile("Resources/phenologie blé 2018 2021(3).zip", 'r') as zip_ref:
            zip_ref.extractall("Resources")
    pheno_ble = pd.read_csv("Resources/phenologie blé 2018 2021(3).csv", error_bad_lines = False, sep = ';', encoding="ISO-8859-1")
    
    
    
    # In[423]:
    
    
    
    
    # In[424]:
    
    
    
    
    # In[425]:
    
    
    pheno_ble.drop(['kingdom', 'data_source', 'scale', 'genus', 'binomial_name'], axis=1, inplace=True)
    
    
    # In[426]:
    
    
    pheno_ble.duplicated().sum()
    
    
    # In[427]:
    
    
    pheno_ble.drop_duplicates(inplace=True)
    
    
    # In[428]:
    
    
    pheno_ble.reset_index(drop=True, inplace=True)
    
    
    # In[429]:
    
    
    
    
    # In[430]:
    
    
    pheno_ble['date'] = pd.to_datetime(pheno_ble['date'],format='%d/%m/%Y', utc = True)
    
    
    # In[431]:
    
    
    
    
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
    
    st.markdown('**Dataframe des différents stades du blé avec dates et caractéristiques**')
    st.write(pheno_ble.head())
    
    # In[437]:
    
    
    stage_description = pheno_ble[pheno_ble['phenological_main_event_code'].isin(pheno_ble['phenological_main_event_code'].unique())][['phenological_main_event_code', 'phenological_main_event_description','stage_code', 'stage_description' ]].drop_duplicates()
    
    
    # In[438]:
    
    
    stage_description = stage_description.sort_values(by=['phenological_main_event_code', 'stage_code']).set_index('phenological_main_event_code')
    
    
    # In[439]:
    button = st.button("**Correspondance de _stage_code_ et _phenological_main_event_code_**")
    if button :
        st.write(stage_description)
    
    
    # In[440]:
    
    st.markdown('**Description du cycle du blé**')
    
    
    # In[441]:
    
    from skimage import io
    img = io.imread('Resources/vivescia_cycle-ble_0.jpg')
    
    
    # In[442]:
    
    
    plt.figure(figsize=(20,20))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    st.pyplot()
    
    # In[443]:
    
    
    st.markdown("Selon le cours normal du cycle, on devrait voir le début de vie de la plante aux alentours d'octobre, novembre, le tallage (_Le tallage est un phénomène naturel qui permet d'obtenir plusieurs tiges à partir d'une seule._) en hiver, la montaison début printemps et le remplissage des grains en juin, qui s'en suit de la récolte en juillet. Analysons les données présentes afin de déterminer si c'est bien le calendrier suivi pour les années 2018, 2019, 2020 et 2021.")
    
    
 
    
    # In[79]:
    annee = st.selectbox("Stades de croissance du blé selon leur mois d'apparition",['2018','2019','2020','2021'])
    import datetime as dt
    import matplotlib.cm as cm


    if annee == '2018':
        fig, ax = plt.subplots(1, figsize=(10,8))
        grid_labels = pheno_ble[pheno_ble['year']==2018]['grid_label']
        unique_labels = grid_labels.unique()
        color_map = cm.get_cmap("viridis", len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            label_data = pheno_ble[(pheno_ble['year']==2018) & (pheno_ble['grid_label']==label)]
            color = color_map(i)
            ax.scatter(x=label_data['date'], y=label_data['phenological_main_event_code'], c=color, label=label)
        
        plt.xlabel('mois')
        plt.xticks([])
        ax2 = ax.twinx()
        month_dates = [dt.datetime(2018, i, 1) for i in range(1, 13)]
        ax2.set_xticks(month_dates)
        ax2.set_xticklabels(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
        ax2.set_yticks([])
        plt.title('Stades du cycle du blé et mois : 2018')
        ax.legend(bbox_to_anchor=(1.35, 1))
        st.pyplot()


        st.markdown("_Stades 0, 1 et 2 début à la mi-septembre, octobre et fin octobre pour la récolte de 2019. Pour la récolte de 2018 les stades 1 et 2 se terminent respectivement en avril (à part quelques exceptions en mai) et en mai, avec donc environ 3 mois de retard sur le calendrier décrit plus haut. Le stade 3 débute en février et finit en juin. Les stades 5 et 6 commencent fin avril et finissent fin juin et juillet.Les stades 7 et 8 débutent en juin et se terminent en juillet, donc avec un mois d'avance par rapport au calendrier typique._")

    # In[80]:
    
    
    
    
    # In[81]:
    
    if annee == '2019':
        fig, ax = plt.subplots(1, figsize=(10,8))

        grid_labels = pheno_ble[pheno_ble['year']==2019]['grid_label']
        unique_labels = grid_labels.unique()
        color_map = cm.get_cmap("viridis", len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            label_data = pheno_ble[(pheno_ble['year']==2019) & (pheno_ble['grid_label']==label)]
            color = color_map(i)
            ax.scatter(x=label_data['date'], y=label_data['phenological_main_event_code'], c=color, label=label)
        
        plt.xlabel('mois')

        ax2 = ax.twinx()
        month_dates = [dt.datetime(2019, i, 1) for i in range(1, 13)]
        ax2.set_xticks(month_dates)
        ax2.set_xticklabels(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
        ax2.set_yticks([])

        plt.title('Stades du cycle du blé et mois : 2019')
        ax.legend(bbox_to_anchor=(1.35, 1))
        st.pyplot()

        st.markdown("_Observations : Les stades 1 et 2 débutés en octobre et fin octobre / mi-novembre 2018 se terminent en avril et mai avec donc environ 3 mois de retard sur le calendrier. Le stade 3 débute en février et finit fin juin. Les stades débutent 5 et 6 fin avril / mai et finissent en juillet. Les stades 7 et 8 commencent en mai et juin et se terminent fin juillet, donc avec un mois d'avance par rapport au calendrier typique. Stades 0, 1 et 2 pour la récolte de 2020 commencent en octobre._")
        
    
    # In[83]:
    
    if annee == '2020':
        fig, ax = plt.subplots(1, figsize=(10,8))

        grid_labels = pheno_ble[pheno_ble['year']==2020]['grid_label']
        unique_labels = grid_labels.unique()
        color_map = cm.get_cmap("viridis", len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            label_data = pheno_ble[(pheno_ble['year']==2020) & (pheno_ble['grid_label']==label)]
            color = color_map(i)
            ax.scatter(x=label_data['date'], y=label_data['phenological_main_event_code'], c=color, label=label)
        plt.xlabel('mois')

        ax2 = ax.twinx()
        month_dates = [dt.datetime(2020, i, 1) for i in range(1, 13)]
        ax2.set_xticks(month_dates)
        ax2.set_xticklabels(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
        ax2.set_yticks([])

        plt.title('Stades du cycle du blé et mois : 2020')
        ax.legend(bbox_to_anchor=(1.35, 1))
        st.pyplot()

        
        st.markdown('_Observations : Calendrier similaire à 2019 mais stade 3 visiblement plus court_')
    
    
    # In[85]:
    if annee == '2021':
        fig, ax = plt.subplots(1, figsize=(10,8))

        grid_labels = pheno_ble[pheno_ble['year']==2021]['grid_label']
        unique_labels = grid_labels.unique()
        color_map = cm.get_cmap("viridis", len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            label_data = pheno_ble[(pheno_ble['year']==2021) & (pheno_ble['grid_label']==label)]
            color = color_map(i)
            ax.scatter(x=label_data['date'], y=label_data['phenological_main_event_code'], c=color, label=label)
        
        plt.xlabel('mois')
        ax2 = ax.twinx()
        month_dates = [dt.datetime(2021, i, 1) for i in range(1, 13)]
        ax2.set_xticks(month_dates)
        ax2.set_xticklabels(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
        ax2.set_yticks([])
        plt.title('Stades du cycle du blé et mois : 2021')
        ax.legend(bbox_to_anchor=(1.35, 1))
        st.pyplot()

        st.markdown("_Observations : Nous avons visiblement moins de données pour l'année 2021, les stades 5 et supérieurs ne sont pas représentés. (On utilisera l'année 2021 comme test, pour prédictions/analyse en ML de la relation météo et phénologie.)_")
    
    # In[87]:
    
    
    #pheno_ble[(pheno_ble['phenological_main_event_code']== 2) & (pheno_ble['grid_label'] == 'Pays de la Loire') & (pheno_ble['taxon']=='aestivum')].sort_values(by=['site_id', 'date'])
    
    
    # In[88]:
    
    
    #pheno_ble[pheno_ble['site_id']== 41812].sort_values(by=['phenological_main_event_code','date'])
    
    
    # In[89]:
    
    #"Si nous prenons l'exemple du stade 2 en région Pays de la Loire pour le site 41812, nous observons que les dates vont du 23/11/2020 au 29/01/2021. Ce qui signifie que le stade 2 débute en novembre et finit en janvier. C'est pourquoi, nous observons sur les graphs que le stade 2 revient 2 fois dans l'année. Il en est de même pour d'autres sites et pour le stade de croissance 1."
     
    
    # In[447]:
    
    
    
    
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
    
    
    # In[449]:
    
    
    rend2018 = pd.DataFrame(prod_vege_2018.iloc[6:12, prod_vege_2018.columns.get_level_values(1)=='Rendement(q/ha)'].mean(), columns = ['2018'])
    
    
    # In[450]:
    percent_category_2018 = []
    for i in pheno_ble[pheno_ble['year']== 2018]['phenological_main_event_code'].value_counts():
        percent_category_2018.append((i/len(pheno_ble[pheno_ble['year']== 2018]['phenological_main_event_code']))*100)
    percent_category_2018 = pd.DataFrame(np.array(percent_category_2018), columns = ['Répartition des stades phénologiques 2018'], index = pheno_ble[pheno_ble['year']== 2018]['phenological_main_event_code'].value_counts().index)
    percent_category_2018 = percent_category_2018.round(2).astype(str) + '%'
    
    
    # In[448]:
    
    rep_rend = st.selectbox('Répartitions des différents stades du blé et rendement pour la même année', ['2018','2019','2020'])
    
    if rep_rend == '2018':
        st.write(percent_category_2018)
        
        st.write(rend2018.iloc[0,:])
    
    
    # In[451]:
    
    
    percent_category_2019 = []
    for i in pheno_ble[pheno_ble['year']== 2019]['phenological_main_event_code'].value_counts():
        percent_category_2019.append((i/len(pheno_ble[pheno_ble['year']== 2019]['phenological_main_event_code']))*100)
    percent_category_2019 = pd.DataFrame(np.array(percent_category_2019), columns = ['Répartition des stades phénologiques 2019'], index = pheno_ble[pheno_ble['year']== 2019]['phenological_main_event_code'].value_counts().index)
    percent_category_2019 = percent_category_2019.round(2).astype(str) + '%'
    
 
    
    # In[453]:
    
    
    rend2019 = pd.DataFrame(prod_vege_2019.iloc[6:12, prod_vege_2019.columns.get_level_values(1)=='Rendement(q/ha)'].mean(), columns = ['2019'])
    
    
    # In[454]:
    if rep_rend == '2019':
        st.write(percent_category_2019)
    
        st.write(rend2019.iloc[0,:])
    
    
    # In[455]:
    
    
    percent_category_2020 = []
    for i in pheno_ble[pheno_ble['year']== 2020]['phenological_main_event_code'].value_counts():
        percent_category_2020.append((i/len(pheno_ble[pheno_ble['year']== 2020]['phenological_main_event_code']))*100)
    percent_category_2020 = pd.DataFrame(np.array(percent_category_2020), columns = ['Répartition des stades phénologiques 2020'], index = pheno_ble[pheno_ble['year']== 2020]['phenological_main_event_code'].value_counts().index)
    percent_category_2020 = percent_category_2020.round(2).astype(str) + '%'
    
    
    
    
    # In[457]:
    
    
    rend2020 = pd.DataFrame(prod_vege_2020.iloc[6:12, prod_vege_2020.columns.get_level_values(1)=='Rendement(q/ha)'].mean(), columns = ['2020'])
    
    
    # In[458]:
    if rep_rend == '2020':
        st.write(percent_category_2020)

        st.write(rend2020.iloc[0,:])
    
    
    # In[459]:
    
    
    st.markdown("On constate, en observant le rendement moyen sur tout le territoire et la répartition des différents stades de croissance du blé qu'il est difficile de trouver une corrélation satisfaisante entre les deux. Le stade 8 qui annonce la récolte est proportionellement plus important en 2018 et 2020 bien que les rendements soient plus faibles qu'en 2019. C'est donc sur un autre critère que l'on doit s'orienter pour estimer le rendement de la culture par rapport aux stades phénologiques.")
    
    
    # In[26]:
    
    
    page = urlopen('https://www.agro.basf.fr/fr/cultures/ble/implantation_du_ble/les_composantes_du_rendement_du_ble/')
    soup = bs(page, 'html.parser', from_encoding='utf-8')
    
    
    # In[24]:
    
    
    #soup.find('main')
    
    
    # In[27]:
    
    
    rend_ble = soup.find('div', {'class':'col text-left'}).text.strip()
    
    
    # In[28]:
    
    st.markdown('**Elements caractérisant un bon rendement de blé**')
    st.write(rend_ble)
    
    
    # In[111]:
    
    
    st.markdown("**Le facteur favorable à un bon rendement est une montaison longue qui permet d'augmenter le volume de grains.** En d'autres termes, on veut une plante qui produit beaucoup de grains pour assurer un bon rendement. **Cela correspond au stade 3 de croissance du blé.** Pour l'année 2019, on remarque que le stade 3 est majoritaire. C'est donc probablement une des raisons pour lesquelles le rendement est plus important.")
    
    
    # In[112]:
    
    
    pheno_ble[pheno_ble['phenological_main_event_code'] ==3].head(30)
    
    
    # In[113]:
    
    
    pheno_ble[pheno_ble['site_id'] == 25903]
    "En prenant l'exemple de la plante 25903, on s'aperçoit que le stade phénologique 3 dure plus longtemps que les autres stades."
    
    
    # In[8]:
    