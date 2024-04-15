import pandas as pd
import numpy as np
import os
from PIL import Image
import csv

def encoding(name):
    classes = {'nv':0,
           'mel' :1,
           'bkl' :2, 
           'bcc'  :3,
           'vasc' :4,
           'akiec' :5,
           'df': 6}
    return classes[name]
        

def image_data(size):
    pixel = []
    df = pd.read_csv('dataset\HAM10000_metadata.csv')
    count=0

    for i in df['image_id']:
        count+=1
        path = os.path.join("dataset", "HAM10000_images_part_1", f"{i}.jpg")

        img = np.asarray(Image.open(path).resize((size[0], size[1])))
        img = img.astype(int).reshape(-1)
        pixel.append(img)
        if count%1000==0:
            print(f"done with {count} samples")

    csv_file = "dataset\pixel_data_" + str(size[0])+"_" + str(size[1]) + ".csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(pixel)
    print("complete reading the file saving it to csv")
    data = pd.read_csv(csv_file, header=None)
    data.columns = [f'pix{i}' for i in range(1, data.shape[1]+1)]
    data['label'] = df['dx'].apply(encoding)
    data.to_csv(csv_file, index=False)
    print("data ingestion complete")
