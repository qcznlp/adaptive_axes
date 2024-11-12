import numpy as np
import pandas as pd
import os


def get_original_axes(file_name, data_base_path):
    """
    This function reads the semantic axes from the file and returns a dict of axes.
    """
    sem_axis = pd.read_csv(
        file_name,
        sep='\t',
        header=None,
        names=['seed_adj', 'left_poles', 'right_poles'],
    )
    sem_axis['left_poles'] = sem_axis['left_poles'].apply(lambda x: x.split(','))
    sem_axis['right_poles'] = sem_axis['right_poles'].apply(lambda x: x.split(','))

    bert_prob_list = [
        x for x in os.listdir(data_base_path + "bert-base-prob") if x.endswith("npy")
    ]
    bert_default_list = [
        x for x in os.listdir(data_base_path + "bert-default") if x.endswith("npy")
    ]

    sem_axis_dict = {}
    for i, j, k in zip(
        sem_axis['seed_adj'], sem_axis['left_poles'], sem_axis['right_poles']
    ):
        left_name = i + "_left.npy"
        right_name = i + "_right.npy"
        if left_name in bert_prob_list:
            axis = np.mean(
                np.load(data_base_path + "bert-base-prob/" + left_name), axis=0
            ) - np.mean(
                np.load(data_base_path + "bert-base-prob/" + right_name), axis=0
            )
        else:
            axis = np.mean(
                np.load(data_base_path + "bert-default/" + left_name), axis=0
            ) - np.mean(np.load(data_base_path + "bert-default/" + right_name), axis=0)
        sem_axis_dict[i] = {'left': j, 'right': k, 'axis': axis}
    return sem_axis_dict


def get_text_embedding_axes(file_name, data_base_path):
    """
    This function reads the semantic axes from the file and returns a dict of axes.
    """
    sem_axis = pd.read_csv(
        file_name,
        sep='\t',
        header=None,
        names=['seed_adj', 'left_poles', 'right_poles'],
    )
    sem_axis['left_poles'] = sem_axis['left_poles'].apply(lambda x: x.split(','))
    sem_axis['right_poles'] = sem_axis['right_poles'].apply(lambda x: x.split(','))

    sem_axis_dict = {}
    for i, j, k in zip(
        sem_axis['seed_adj'], sem_axis['left_poles'], sem_axis['right_poles']
    ):
        left_vectors, right_vectors = [], []
        for l in j:
            left_vectors.append(np.load(data_base_path + l + ".npy"))
        for r in k:
            right_vectors.append(np.load(data_base_path + r + ".npy"))
        left_vectors = np.mean(np.array(left_vectors), axis=0)
        right_vectors = np.mean(np.array(right_vectors), axis=0)
        axis = left_vectors - right_vectors
        sem_axis_dict[i] = {'left': j, 'right': k, 'axis': axis}

    return sem_axis_dict


# if __name__ == "__main__":
#     sem_axis_dict = get_text_embedding_axes(
#         "./dataset/wordnet_axes.txt", "./dataset/text_embeddings/"
#     )
#     print(sem_axis_dict)
