import os, random
import pandas as pd
import glob
import cv2
import numpy as np
# import piexif
from exif import Image

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Parameters
rows = 10
cols = 10

def random_image(path):
    images = glob.glob(path + "*.jpg")
    return np.random.choice(images)


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
                gray_image, scaleFactor=1.15, minNeighbors=7, minSize=(100, 100)
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
    blurred = cv2.GaussianBlur(inverted_image, (5, 5), 0)

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
    
def hide_everything_except_faces(file):
    faces = detect_face(file)
    img = cv2.imread(file)
    filename, root_path = split_path_name(file)

    # create a new image with the same size as the original
    new_img = np.zeros(img.shape, np.uint8)
    new_img[:, :] = (0, 255, 0)

    # crop each face
    for i, face in enumerate(faces):
        x, y, w, h = face
        new_img[y:y+h, x:x+w] = img[y:y+h, x:x+w]

    new_file = root_path + "tmp/" + filename + "_faces_only.jpg"
    cv2.imwrite(new_file, new_img)

    return new_file



def image_coordinates(image_path):
    try:
        # Ouvrir l'image avec Pillow
        image = Image.open(image_path)

        # Vérifier si les données EXIF existent dans l'image
        if hasattr(image, '_getexif'):
            exif_data = image._getexif()

            if exif_data is not None:
                # Rechercher les balises GPS dans les métadonnées EXIF
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)

                    if tag_name == 'GPSInfo':
                        gps_info = {}
                        for key in value:
                            sub_decoded_tag = GPSTAGS.get(key, key)
                            gps_info[sub_decoded_tag] = value[key]

                        # Extraire les coordonnées GPS
                        latitude = gps_info.get('GPSLatitude')
                        longitude = gps_info.get('GPSLongitude')
                        latitude_ref = gps_info.get('GPSLatitudeRef', 'N')
                        longitude_ref = gps_info.get('GPSLongitudeRef', 'E')

                        if latitude and longitude:
                            # Convertir les coordonnées GPS degrés/minutes/secondes en décimales
                            # latitude_decimal = (latitude[0] + latitude[1] / 60 + latitude[2] / 3600) * (-1 if latitude_ref == 'S' else 1)
                            latitude_decimal = decimal_coords(latitude, latitude_ref)
                            # longitude_decimal = (longitude[0] + longitude[1] / 60 + longitude[2] / 3600) * (-1 if longitude_ref == 'W' else 1)
                            longitude_decimal = decimal_coords(longitude, longitude_ref)

                            return latitude_decimal, longitude_decimal
        else:
            print("L'image n'a pas de données EXIF.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

    return None, None


def decimal_coords(coords, ref):
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref =='W' :
        decimal_degrees = -decimal_degrees
    return float(decimal_degrees)


def get_exif(file):
    filename = file.split("/")[-1].split(".")[0]
    year = str(filename[0:4])
    month = str(filename[4:6])

    # month is a month if isdigit() is True
    if not month.isdigit():
        month = None

    # name is char after the first _
    name = filename.split("_")[1]
    # replace _ by space
    name = name.replace("_", " ")
    name = name.replace("-", " ")

    latitude, longitude = image_coordinates(file)
    gmap_url =  None

    if latitude is not None and longitude is not None:
        gmap_url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
   
    return year, month, name, latitude, longitude, gmap_url

def split_path_name(path):
    name = path.split("/")[-1].split(".")[0]
    # path without the file
    root_path = "/".join(path.split("/")[:-1]) + "/"
    return name, root_path