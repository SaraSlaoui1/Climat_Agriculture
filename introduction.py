# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 22:19:14 2023

@author: Sara
"""
import streamlit as st
from skimage import io
import matplotlib.pyplot as plt

def intro_function():
    import platform


    img = io.imread('Resources/nature-friendly.png')  
    st.image(img, width=250)

    st.header('Introduction')
    st.markdown("Etant sensible à la cause écologique et très à l'écoute de l'actualité, j'ai voulu réaliser ce projet pour appliquer les connaissances acquises lors de la formation et en apprendre d'avantage sur l'agriculture, ses caractéristiques et l'impact potentiel que le climat pourrait avoir sur celle-ci. Tout ce projet a été réalisé à partir de données trouvées en libre accès sur internet, je mets à disposition mes sources dans la page _Sources_ avec liens de téléchargement et/ou site internet correspondants.")
    
    st.subheader('**Parties** : ')
    st.markdown('**Analyse agriculture** : ')
    st.markdown("J'ai récolté les données chiffrées des différentes grandes cultures françaises de 2018 à 2021 pour observer leurs évolutions.")
    st.markdown("**Analyse Phénologique** : ")
    st.markdown("L'analyse phénologique du blé est l'étude de son cycle de vie, c'est une denrée agriculture indispensable. Et c'est l'analyse de son comportement qui peut expliquer/prédire son rendement.")
    st.markdown('**Analyse météorologique** : ')
    st.markdown('Données météorologiques de 2018 à 2021')
    st.markdown("**Machine Learning** : ")
    st.markdown('Statistiques exploratoires et Prédictions')
    st.markdown('**Conclusion** : ')
    st.markdown('Résumé et mise en perspective')