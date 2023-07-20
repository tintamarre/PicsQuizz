import streamlit as st
import hashlib
import random

import main_lib as lib
import image_manipulation as img

st.set_page_config(page_title="PicsQuizz", layout="wide")

def clear():
    st.session_state.result = None

def generate_image(file):
    hide_everything_except_faces = img.hide_everything_except_faces(file)
    if random.randint(0, 1) == 0:
        tranformed = img.make_sepia(file)
    else:
        tranformed = img.make_sketch(file)
    
    puzzle = img.make_puzzle(file)

    portrait = img.add_rectangle(file)

    return hide_everything_except_faces, tranformed, puzzle, portrait

file = img.random_image('./data/sources/')

if file:
    hide_everything_except_faces, tranformed, puzzle, portrait = generate_image(file)    

    # md5 of the file name
    md5 = hashlib.md5(file.encode()).hexdigest()[:8]

    # tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portraits",
        "Transformée",
        "Puzzle",
        "Original",
        "SOLUTION"
        ])

    with tab1:
        st.image(hide_everything_except_faces, caption="faces", use_column_width=True)
        
        next_image = st.button("Suivant")

    with tab2:
        # random effect on image, either sepia or sketch      
        st.image(tranformed, caption="random", use_column_width=True)
    with tab3:
        st.image(puzzle, caption="puzzle", use_column_width=True)


    with tab4:
        st.image(portrait, caption="portrait", use_column_width=True)
        

    with tab5:
        exif = img.get_exif(file)
        for value in exif:
            st.markdown(f"- {value}")
    
    st.markdown(f"#### {md5}")
