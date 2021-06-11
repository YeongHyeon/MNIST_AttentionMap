import os, glob, shutil

def make_dir(path, refresh=False):

    try: os.mkdir(path)
    except:
        if(refresh):
            shutil.rmtree(path)
            os.mkdir(path)

def sorted_list(path):

    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

def min_max_norm(x):

    min_x, max_x = x.min(), x.max()
    return (x - min_x + 1e-12) / (max_x - min_x + 1e-12)
