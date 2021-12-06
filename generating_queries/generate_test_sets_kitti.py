# Code taken from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random


###################
pos_range = 10
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

fname= "pointcloud_locations_20m.csv"
pointcloud_fols = "/pointcloud_20m/"

data_path = os.path.join(BASE_DIR, base_path, runs_folder)
all_folders = sorted(os.listdir(data_path))
folders = []
index_list = [1]
for index in index_list:
    folders.append(all_folders[index])
print(folders)


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if (point[0] - x_width < northing and northing < point[0] + x_width and point[
            1] - y_width < easting and easting < point[1] + y_width):
            in_test_set = True
            break
    return in_test_set


##########################################

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, output_name):
    database_trees = []
    test_trees = []
    for folder in folders:
        print(folder)
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting', 'altitude'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting', 'altitude'])

        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        # df_locations['timestamp']=runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str)+'.bin'
        # df_locations=df_locations.rename(columns={'timestamp':'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            # if (check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
            df_test = df_test.append(row, ignore_index=True)
            df_database = df_database.append(row, ignore_index=True)

        database_tree = KDTree(df_database[['northing', 'easting', 'altitude']])
        test_tree = KDTree(df_test[['northing', 'easting', 'altitude']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)


    test_sets = []
    database_sets = []
    for folder in folders:
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
        # file name of the cloud file
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            # if (check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
            test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting'], 'altitude': row['altitude']}
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                              'easting': row['easting'], 'altitude': row['altitude']}
        database_sets.append(database)
        test_sets.append(test)

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            # if (i == j):
            #     continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"], test_sets[j][key]["altitude"]]])
                index, distances = tree.query_radius(coor, r=pos_range, return_distance=True)
                # indices of the positive matches in database i of each query (key) in test set j
                positive_set = []
                for dist_idx, dist in enumerate(distances[0]):
                    if dist>0.1:
                        positive_set.append(index[0][dist_idx])
                test_sets[j][key][i] = positive_set

    output_to_file(database_sets, '{}_evaluation_database.pickle'.format(output_name))
    output_to_file(test_sets, '{}_evaluation_query.pickle'.format(output_name))

###Building database and query files for evaluation
construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, fname, output_name)
