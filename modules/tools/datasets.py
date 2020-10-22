from sklearn import datasets
import os
import random
import sys
import requests


def write_dataset(dataset, filename):
    classes = set()
    for row in dataset:
        classes.add(row[1])
    ref = dict((v, k) for (k, v) in enumerate(list(classes)))
    with open(filename, "w") as f:
        for row in dataset:
            f.write(f"{ref[row[1]]}")
            for v in row[0]:
                f.write(f"\t{v}")
            f.write("\n")

            
def get_mnist(random_seed=13):
    '''loads data and creates train/test splitting. Returns a tuple (train, test) where each item is a list of tuples (vector, class)'''
    random.seed(random_seed)
    digits = datasets.load_digits()
    X, Y = [x.flatten() for x in digits.images], digits.target
    XY = list(zip(X, Y))
    random.shuffle(XY)
    train_mnist, test_mnist = XY[:9 * len(XY) // 10], XY[9 * len(XY) // 10:]
    return train_mnist, test_mnist


def load_100leaves_from_drive(folder, random_seed=13):
    random.seed(13)

    texture_file = os.path.join(folder, "data_Tex_64.txt")
    shape_file = os.path.join(folder, "data_Sha_64.txt")
    margin_file = os.path.join(folder, "data_Mar_64.txt")

    dataset1, dataset2, dataset3 = [], [], []
    for line1, line2, line3 in zip(open(texture_file), open(shape_file), open(margin_file)):
        parts = line1.strip().split(',')
        class_, vect = parts[0], list(map(float, parts[1:]))
        dataset1.append((vect, class_))
        parts = line2.strip().split(',')
        class_, vect = parts[0], list(map(float, parts[1:]))
        dataset2.append((vect, class_))
        parts = line3.strip().split(',')
        class_, vect = parts[0], list(map(float, parts[1:]))
        dataset3.append((vect, class_))

    random.shuffle(dataset1)
    random.shuffle(dataset2)
    random.shuffle(dataset3)
       
    return {
        "Tex": (dataset1[:9 * len(dataset1) // 10], dataset1[9 * len(dataset1) // 10:]),
        "Sha": (dataset2[:9 * len(dataset2) // 10], dataset2[9 * len(dataset2) // 10:]),
        "Mar": (dataset3[:9 * len(dataset3) // 10], dataset3[9 * len(dataset3) // 10:])
    }


def cached_download(url, destination_folder, file):
    folder = os.path.abspath(destination_folder)
    thefile = os.path.join(folder, file)
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    if not os.path.exists(thefile):
        r = requests.get(url, allow_redirects=True)
        open(thefile, 'wb').write(r.content)
    return thefile

       
def download_covtype_binary():
    dir_ = "../data/covtype"
    thefile = cached_download(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2",
                dir_,
                "covtype.binary.scale.bz2")
    exe7z = "C:/Program Files/7-Zip/7z.exe"
    exe = f"\"{exe7z}\" e -o\"{dir_}\" \"{thefile}\""
    return exe

def download_susy():
    dir_ = "../data/susy"
    thefile = cached_download(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.bz2",
                dir_,
                "SUSY.bz2")
    exe7z = "C:/Program Files/7-Zip/7z.exe"
    exe = f"\"{exe7z}\" e -o\"{dir_}\" \"{thefile}\""
    return exe

def download_higgs():
    dir_ = "../data/higgs"
    thefile = cached_download(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2",
                dir_,
                "HIGGS.bz2")
    exe7z = "C:/Program Files/7-Zip/7z.exe"
    exe = f"\"{exe7z}\" e -o\"{dir_}\" \"{thefile}\""
    return exe