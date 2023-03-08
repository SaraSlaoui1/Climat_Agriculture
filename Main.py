# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 22:41:10 2023

@author: Sara
"""

import streamlit as st
import sys
from skimage import io
import streamlit as st
import matplotlib.pyplot as plt
sys.path.append(r"C:\Users\Hassan\Documents\datascientest\projet datascientest PROD AGRICOLES\pages\Machine Learning.py")
import machine_learning
sys.path.append(r"C:\Users\Hassan\Documents\datascientest\projet datascientest PROD AGRICOLES\pages\analyse_agriculture.py")
import analyse_agriculture
sys.path.append(r"C:\Users\Hassan\Documents\datascientest\projet datascientest PROD AGRICOLES\pages\analyse_meteo.py")
import analyse_meteo
sys.path.append(r"C:\Users\Hassan\Documents\datascientest\projet datascientest PROD AGRICOLES\pages\phenologie_ble.py")
import phenologie_ble
sys.path.append(r"C:\Users\Hassan\Documents\datascientest\projet datascientest PROD AGRICOLES\pages\Conclusion.py")
import conclusion
sys.path.append(r"C:\Users\Hassan\Documents\datascientest\projet datascientest PROD AGRICOLES\pages\introduction.py")
import introduction
from analyse_agriculture import analyse_agriculture_function
from phenologie_ble import phenologie_ble_function
from analyse_meteo import analyse_meteo_function
from machine_learning import machine_learning_function
from conclusion import conclusion_function
from introduction import intro_function
from sources import sources_function
st.cache()


# Load image
img = io.imread("Resources/futuristic-heat-absorber.jpg")

st.sidebar.image(img, caption='Heat Absorber', width=300)
st.sidebar.write("Bâtiment imaginé avec image générée par une IA permettant d'absorber la chaleur sur les champs d'exploitations agricoles. Cette chaleur accumulée serait la source d'énergie alimentant les machines agricoles de l'exploitation.")


def main():
    st.title("Projet DataScientest Agriculture et Météo")
    menu = ["Introduction","Analyse Agriculture", "Phenologie Ble", "Analyse Meteo", "Machine Learning", "Conclusion", "Sources"]
    choice = st.sidebar.selectbox("Select a page", menu)
    if choice == 'Introduction':
        intro_function()
    elif choice == "Analyse Agriculture":
        analyse_agriculture_function()
    elif choice == "Phenologie Ble":
        phenologie_ble_function()
    elif choice == "Analyse Meteo":
        analyse_meteo_function()
    elif choice == "Machine Learning":
        machine_learning_function()
    elif choice == 'Conclusion':
        conclusion_function()
    else:
        sources_function()

if __name__ == "__main__":
    main()
st.sidebar.markdown('DataScientest - Promo DA Continu Juillet 2022')
st.sidebar.markdown('Sara Slaoui')
link_url = 'www.linkedin.com/in/sara-slaoui-11490717b'
link_url2 = 'https://github.com/SaraSlaoui1/Sara_Data_Analyst'

st.sidebar.markdown(f'<a href="{link_url}" target="_blank">LinkedIn</a>', unsafe_allow_html=True)
st.sidebar.markdown(f'<a href="{link_url2}" target="_blank">GitHub</a>', unsafe_allow_html=True)




