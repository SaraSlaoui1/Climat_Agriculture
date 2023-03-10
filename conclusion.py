# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:54:22 2023

@author: Sara
"""
import streamlit as st
def conclusion_function():
 st.header("Conclusion")

 st.markdown("En conclusion, nous avons exploré les données de productions agricoles de 2018 à 2021 et avons constaté une **baisse de rendement en 2020**. Sachant que cette année a été fortement impactée par les confinements, cette baisse pourrait s'expliquer par le manque de main d'oeuvre pour travailler les terres agricoles. Le but étant d'établir un lien entre les données météorologiques et la production agricole, les données telles quelles étaient insuffisantes. Il fallait comprendre le comportement du végétal dans son développement étape par étape.") 
 st.markdown("J'ai choisi de cibler **le blé car il représente une denrée alimentaire essentielle pour l'humanité** et un bon rendement est donc crucial. L'étude phénologique du blé se base sur la **mise en relation des dates attendues des différents stades par rapport aux dates observées**. Pour les années concernées, on observe déjà un décalage, **souvent les stades sont précoces** (ils commencent plus tôt que prévus), ce qui est communément **signe de températures plus élevées que la moyenne de saison** et les stades 1 et 2 durent plus longtemps que la moyene. En cherchant plus loin la signification et l'importance de chaque stade, j'ai pu apprendre qu'en agriculture, **le stade de montaison est déterminant pour la récolte. Si ce stade est plus court, les épis seront moins denses donc produiront moins de grains**. C'est comme ça que j'ai pu établir le lien entre les rendements de 2018 à 2020 en constatant que **2019 était l'année la plus productive car le stade 3 était plus long pour la majorité des plantes observées.**") 
 st.markdown("Ceci étant fait, j'ai analysé les **données météo** et l'année **la plus froide fut 2021** et **la plus chaude 2020. La plus pluvieuse 2018 et la plus sèche 2019**. Ceci dit, ce sont des observations des moyennes des différentes données, qui ne prennent donc pas en compte les amplitudes.") 
 st.markdown("J'ai ensuite mergé les données phénologiques et météo pour ensuite effectué des tests statistiques afin de démontrer une corrélation entre ces deux types de variables. Selon le test d'ANOVA', **les températures et l'humidité sont bien corrélées aux stades de croissance du blé**. On a donc pu établir un modèle de ML avec en features les données météo et target les stades phénologiques. Premièrement, j'ai prédit les stades de l'année 2021 car incomplets dans le dataset d'origine, puis j'ai finalement fait les prédictions de l'année 2100 en utilisant les hausses de température estimées par le dernier rapport du GIEC.") 
 st.markdown("Les résultats montrent une **diminution de la période de montaison (stade 3) en 2100**, qui pourrait donc amener à une **récolte plus faible.**")
 st.markdown(" **Les variables déterminantes sont en réalité plus complexes et variées**, du recensement de la main d'oeuvre agricole à l'utilisation de pesticides, en passant par les irrigations artificielles, les facteurs sont très nombreux. C'est pourquoi, ces prédictions restent aproximatives et ne prennent pas en compte tout le champs possible des conditions de l'exploitation agricole.")


# In[ ]:




