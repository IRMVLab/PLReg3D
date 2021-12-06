# Code taken from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random
from tqdm import tqdm


pos_range = 5
neg_range = 30

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "../benchmark_datasets/"
# dataset = "sjtu"
dataset = "kitti"
runs_folder = "{}/".format(dataset)
pickle_save_folder = "pickles".format(dataset)
if not os.path.exists(pickle_save_folder):
    os.mkdir(pickle_save_folder)
output_name = os.path.join(pickle_save_folder, dataset)

train_fname= "pointcloud_locations_20m_10overlap.csv"
pointcloud_fols = "/pointcloud_20m_10overlap/"

data_path = os.path.join(BASE_DIR, base_path, runs_folder)
all_folders = sorted(os.listdir(data_path))
folders = []
index_list = [0]
for index in index_list:
    folders.append(all_folders[index])
print(folders)

def construct_query_dict(df_centroids, output_name):
    tree = KDTree(df_centroids[['northing', 'easting', 'altitude']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting', 'altitude']], r=pos_range)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting', 'altitude']], r=neg_range)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        negatives = np.setdiff1d(df_centroids.index.values.tolist(), ind_r[i]).tolist()
        random.shuffle(negatives)
        # random.shuffle(positives)
        queries[i] = {"query": query, "positives": positives, "negatives": negatives}


    filename = '{}_training_queries_baseline.pickle'.format(output_name)
    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


####Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file', 'northing', 'easting', 'altitude'])

for folder in folders:
    # get for the case with one subfile
    df_locations = pd.read_csv(os.path.join(BASE_DIR, base_path, runs_folder, folder, train_fname), sep=',')
    df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})
    for index, row in df_locations.iterrows():
        df_train = df_train.append(row, ignore_index=True)
    print("Number of training submaps: " + str(len(df_train['file'])))


construct_query_dict(df_train, output_name)

