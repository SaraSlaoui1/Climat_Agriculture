# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:37:36 2023

@author: Hassan
"""

import streamlit as st
def sources_function():
 st.header("**Sources**")
 st.markdown('Productions végétales :')
 link_url = "https://agreste.agriculture.gouv.fr/agreste-web/download/publication/publie/IraGcu20144/2020_144donneesgrandescultures.xls"

 st.markdown(f'<a href="{link_url}" target="_blank">Premier lien agreste.agriculture.gouv</a>', unsafe_allow_html=True)
 link_url2 = "https://agreste.agriculture.gouv.fr/agreste-web/download/publication/publie/IraGcu19147/2019_147donneesdeptgrandescultures.xls"

 st.markdown(f'<a href="{link_url2}" target="_blank">Deuxième lien agreste.agriculture.gouv</a>', unsafe_allow_html=True)
 link_url3 = 'https://agreste.agriculture.gouv.fr/agreste-web/download/publication/publie/IraGcu21149/2021_149donneesdeptgrandescultures.xlsx'

 st.markdown(f'<a href="{link_url3}" target="_blank">Troisième lien agreste.agriculture.gouv</a>', unsafe_allow_html=True)
 st.markdown('Météo')
 link_url4 = 'https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm/table/?flg=fr&sort=date'
 
 st.markdown(f'<a href="{link_url4}" target="_blank">Lien DataSet météo public.opendatasoft</a>', unsafe_allow_html=True)
 st.markdown('Phénologie blé')
 link_url5 = 'https://data.pheno.fr/'
 
 st.markdown(f'<a href="{link_url5}" target="_blank">Lien site référencement données phénologiques</a>', unsafe_allow_html=True)