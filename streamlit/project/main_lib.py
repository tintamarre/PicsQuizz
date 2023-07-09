import streamlit as st
import os, random
import pandas as pd
import glob
import cv2
import numpy as np
import piexif
import sqlite3

# Parameters
rows = 12
cols = 12

def random_image(path):
    images = glob.glob(path + "*.jpg")
    return np.random.choice(images)


def get_exif(file):
    year = file.split("/")[-1].split(".")[0][0:4]
    exif = piexif.load(file)

    if exif["0th"] and piexif.ImageIFD.DateTime in exif["0th"]:
        exif_date = exif["0th"][piexif.ImageIFD.DateTime]
        date = exif_date.decode("utf-8")
    else:
        date = year + ":01:01 00:00:00"

    gps = None
    if exif["GPS"]:
        gps = exif["GPS"]

    return date, gps

def split_path_name(path):
    name = path.split("/")[-1].split(".")[0]
    # path without the file
    root_path = "/".join(path.split("/")[:-1]) + "/"
    return name, root_path

def puzzle_image(file):
    img = cv2.imread(file)
    height, width = img.shape[:2]
    cell_height = height // rows
    cell_width = width // cols

    orginal_name, root_path = split_path_name(file)
    
    for i in range(rows):
        for j in range(cols):
            y = i * cell_height
            x = j * cell_width
            cell = img[y:y+cell_height, x:x+cell_width]
            cv2.imwrite(f"{root_path}tmp/cell_{i}_{j}.jpg", cell)

# recompose the image
def recompose_image(file):
    orginal_name, root_path = split_path_name(file)
    images = glob.glob(root_path + "tmp/cell_*.jpg")
    # in random order
    images = np.random.choice(images, size=rows*cols, replace=False)
    # images = sorted(images, key=lambda x: int(x.split("_")[1]))
    images = np.array(images).reshape(rows, cols)
    for i in range(rows):
        for j in range(cols):
            img = cv2.imread(images[i][j])
            if j == 0:
                row = img
            else:
                row = np.concatenate((row, img), axis=1)
        if i == 0:
            final = row
        else:
            final = np.concatenate((final, row), axis=0)
    
    new_file = root_path + 'tmp/' + orginal_name + "_puzzle.jpg"
    cv2.imwrite(new_file, final)
    return new_file

def make_puzzle(file):
    puzzle_image(file)
    return recompose_image(file)

def detect_face(file):
    # Read the input image
    image = cv2.imread(file)

    # convert to greyscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            ).detectMultiScale(
                gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
    return faces

def add_rectangle(file):
    faces = detect_face(file)
    image = cv2.imread(file)

    filename, root_path = split_path_name(file)

    for (x, y, w, h) in faces:
        # add rectangle to the image 
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 10)
     
    # save the image
    new_file = root_path + "tmp/" + filename + "_faces_detected.jpg"
    cv2.imwrite(new_file, image)
    
    return new_file


def crop_faces(file):
    faces = detect_face(file)
    img = cv2.imread(file)
    filename, root_path = split_path_name(file)

    new_files = []
    # crop each face
    for i, face in enumerate(faces):
        x, y, w, h = face
        face = img[y:y+h, x:x+w]
        new_file = root_path + "tmp/" + filename + "_face_" + str(i) + ".jpg"
        cv2.imwrite(new_file, face)
        new_files.append(new_file)

    return new_files

def make_sketch(file):
    img = cv2.imread(file)
    filename, root_path = split_path_name(file)

    # convert to greyscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # invert the image
    inverted_image = 255 - gray_image

    # blur the image by gaussian function
    blurred = cv2.GaussianBlur(inverted_image, (35, 35), 0)

    # invert the blurred image
    inverted_blurred = 255 - blurred

    # create pencil sketch image
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    new_file = root_path + "tmp/" + filename + "_sketch.jpg"
    cv2.imwrite(new_file, pencil_sketch)

    return new_file

def make_sepia(file):
    img = cv2.imread(file)
    filename, root_path = split_path_name(file)

    # convert to greyscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # invert the image
    inverted_image = 255 - gray_image

    # blur the image by gaussian function
    blurred = cv2.GaussianBlur(inverted_image, (35, 35), 0)

    # invert the blurred image
    inverted_blurred = 255 - blurred

    # create pencil sketch image
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    # convert to sepia
    sepia = cv2.cvtColor(pencil_sketch, cv2.COLOR_GRAY2BGR)
    sepia[:, :, 0] = (sepia[:, :, 0] * 0.272) + (sepia[:, :, 1] * 0.534) + (sepia[:, :, 2] * 0.131)
    sepia[:, :, 1] = (sepia[:, :, 0] * 0.349) + (sepia[:, :, 1] * 0.686) + (sepia[:, :, 2] * 0.168)

    new_file = root_path + "tmp/" + filename + "_sepia.jpg"
    cv2.imwrite(new_file, sepia)

    return new_file
    


def new_game():
    # delete jpg content of tmp folder
    files = glob.glob("./data/tmp/*.jpg")
    for f in files:
        os.remove(f)

    c, conn = get_db_cursor()

    c.execute("DROP TABLE IF EXISTS scores")
    c.execute("CREATE TABLE scores (team STRING, score INT)")
    c.execute("INSERT INTO scores VALUES ('atos', 0)")
    c.execute("INSERT INTO scores VALUES ('caramel', 0)")

    # create a dataframe with pics
    df = pd.DataFrame()
    df["pics"] = glob.glob("./data/sources/*.jpg")
    df["exif"] = ''
    df["done"] = False
    df["score_atos"] = 0
    df["score_caramel"] = 0
    df.to_sql("pics", conn, if_exists="replace")
    conn.commit()
    conn.close()

def get_db_cursor():
    conn = sqlite3.connect("data/scores.db")
    c = conn.cursor()
    return c, conn

def add_point(team, points=1):
    c, conn = get_db_cursor()
    c.execute(f"SELECT score FROM scores WHERE team='{team}'")
    score = c.fetchone()[0]
    score += points
    c.execute(f"UPDATE scores SET score={score} WHERE team='{team}'")
    conn.commit()
    conn.close()


def get_score(team):
    c, conn = get_db_cursor()
    c.execute(f"SELECT score FROM scores WHERE team='{team}'")
    score = c.fetchone()[0]
    conn.close()
    return score

   