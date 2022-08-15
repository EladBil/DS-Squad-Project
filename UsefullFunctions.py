import pandas as pd


"""
A method to load all the parts of the dataset and combine them to the original one.
Returns a DataFrame of the original dataset.
"""
def load_dataset():
    # Load Dataset
    first_dest_file_path = "dataset/dataset_part1.csv"
    second_dest_file_path = "dataset/dataset_part2.csv"
    df1 = pd.read_csv(first_dest_file_path)
    df2 = pd.read_csv(second_dest_file_path)
    df = pd.concat([df1, df2])
    return df