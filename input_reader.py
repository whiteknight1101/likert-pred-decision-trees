import cv2
import numpy as np 
import os

def process_image(image_path):
    img = cv2.imread(image_path)

    features = img.flatten()
    
    return features

# Function to parse images 
def parse_train_images_bin(train_path, flag=True):
    
    X = []
    Y = []

    # Iterate through the subfolders in the train folder 
    for dirpath, dirnames, files in os.walk(train_path):
        
        dir =os.path.basename(dirpath)

        if (dir == "train"):
            pass

        # Person images 
        if (dir == "person"):
            for img_name in files:
                if img_name.endswith((".png")):
                    img_path = os.path.join(train_path, "person", img_name)
                    # print(img_path, train_path, dir, img_name)
                    x = process_image(img_path)
                    if (flag):
                        x = np.append(x,1)
                    X.append(x)
                    Y.append(1)

        elif ((dir == "airplane") or (dir == "car") or (dir == "dog")):
            for img_name in files:
                if img_name.endswith((".png")):
                    img_path = os.path.join(train_path, dir, img_name)
                    # print(img_path, train_path, dir, img_name)
                    x = process_image(img_path)
                    if (flag):
                        x = np.append(x,-1)
                    X.append(x)
                    Y.append(-1)

    X = np.array(X)
    Y = np.array(Y)
    # return X
    return (X,Y)

# Function to parse images 
def parse_train_images_multi(train_path, flag=True):
    
    X = []
    Y = []

    # Iterate through the subfolders in the train folder 
    for dirpath, dirnames, files in os.walk(train_path):
        
        # Person images 
        if (os.path.basename(dirpath) == "person"):
            for img_name in files:
                if img_name.endswith((".png")):
                    img_path = os.path.join(train_path, "person", img_name)
                    x = process_image(img_path)
                    if (flag):
                        x = np.append(x,1)
                    X.append(x)
                    Y.append(1)

        # Person images 
        elif (os.path.basename(dirpath) == "dog"):
            for img_name in files:
                if img_name.endswith((".png")):
                    img_path = os.path.join(train_path, "dog", img_name)
                    x = process_image(img_path)
                    if (flag):
                        x = np.append(x,3)
                    X.append(x)
                    Y.append(3)

                # Person images 
        elif (os.path.basename(dirpath) == "car"):
            for img_name in files:
                if img_name.endswith((".png")):
                    img_path = os.path.join(train_path, "car", img_name)
                    x = process_image(img_path)
                    if (flag):
                        x = np.append(x,0)
                    X.append(x)
                    Y.append(0)

                # Person images 
        elif (os.path.basename(dirpath) == "airplane"):
            for img_name in files:
                if img_name.endswith((".png")):
                    img_path = os.path.join(train_path, "airplane", img_name)
                    x = process_image(img_path)
                    if (flag):
                        x = np.append(x,2)
                    X.append(x)
                    Y.append(2)

    X = np.array(X)
    Y = np.array(Y)

    return (X,Y)

def parse_test_images(test_path, ids=[]):
    
    X = []

    # Iterate through the subfolders in the train folder 
    for dirpath, dirnames, files in os.walk(test_path):
        
        dir =os.path.basename(dirpath)
        
        for img_name in files:
            if img_name.endswith((".png")):
                img_path = os.path.join(test_path, img_name)
                x = process_image(img_path)
                X.append(x)
                id_name = img_name.replace('.png','')
                ids.append(id_name)

    X = np.array(X)

    return X

# parse_train_images_bin()

# Write the output to csv file 
def write_test_output(outfile):
    return

