import os, random
import pandas as pd
import glob
import sqlite3
import image_manipulation as img


def new_game():
    # delete jpg content of tmp folder
    files = glob.glob("./data/sources/tmp/*.jpg")
    for f in files:
        os.remove(f)

    c, conn = get_db_cursor()

    conn.commit()

    # create a dataframe with pics
    df = pd.DataFrame()
    df["pics"] = glob.glob("./data/sources/*.jpg")
    df["exif"] = ''

    # only take 10 random pics
    df = df.sample(10)

    for idx, image in df.iterrows():
        # get exif
        exif = img.get_exif(image["pics"])
        df.loc[idx, "exif"] = exif[0]


    df.to_sql("pics", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

def get_db_cursor():
    conn = sqlite3.connect("data/scores.db")
    c = conn.cursor()
    return c, conn


def get_pics():
    c, conn = get_db_cursor()
    df = pd.read_sql("SELECT * FROM pics", conn)
    # randomize order
    df = df.sample(frac=1)
    
    conn.close()
    return df
   