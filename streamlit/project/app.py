import streamlit as st
import pandas as pd
import glob
import os 
import sqlite3

import main_lib as lib

st.set_page_config(page_title="PicsQuizz")
st.title("PicQuizz")


# new game
new_game = st.button("New Game")
if new_game:
    lib.new_game()

def display_image():

    # find random file in folder
    file = lib.random_image("./data/sources/")
    # take a random subset of the file

    # tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Random", "Puzzle", "Saturated", "Portraits", "Original"])

    with tab1:
        # sketch = lib.make_sketch(file)
        # st.image(sketch, caption="random", use_column_width=True)

        hide_everything_except_faces = lib.hide_everything_except_faces(file)
        st.image(hide_everything_except_faces, caption="faces", use_column_width=True)
      
    with tab2:
        puzzle = lib.make_puzzle(file)
        st.image(puzzle, caption="puzzle", use_column_width=True)
        exif = lib.get_exif(file)
        st.markdown(f"> {exif}")

    with tab3:
        sepia = lib.make_sepia(file)
        st.image(sepia, caption="sepia", use_column_width=True)

    with tab4:
        faces = lib.crop_faces(file)
        for face in faces:
            st.image(face, caption="face", use_column_width=True)

    with tab5:
        portrait = lib.add_rectangle(file)
        st.image(portrait, caption="portrait", use_column_width=True)
        st.image(file, caption="original", use_column_width=True)

    
display_image()

# table with two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Équipe Atos")
    # score of Atos
    st.metric(label="", value=lib.get_score("atos"), delta="")

with col2:
    st.header("Équipe Caramel")
    st.metric(label="", value=lib.get_score("caramel"), delta="")


if col1.button("Add point to Atos"):
    lib.add_point("atos")
    st.experimental_rerun()

if col2.button("Add point to Caramel"):
    lib.add_point("caramel")
    st.experimental_rerun()

