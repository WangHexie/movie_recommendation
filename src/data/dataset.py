import os
import pickle
from pathlib import Path

import pandas as pd
from scipy.sparse import coo_matrix

from src.config.configs import DataPath, TmpVarPath


def root_dir():
    return Path(os.path.abspath(os.path.dirname(__file__))).parent.parent


def read_data(path: list):
    return pd.read_csv(os.path.join(root_dir(), "data", *path))


def save_tmp_data(data, path):
    os.makedirs(os.path.join(root_dir(), "models", *path[:-2]), exist_ok=True)
    with open(os.path.join(root_dir(), "models", *path), "wb") as f:
        pickle.dump(data, f)


def read_tmp_data(path):
    with open(os.path.join(root_dir(), "models", *path), "rb") as f:
        data = pickle.load(f)
    return data


class Rating:
    def __init__(self, tmp_var_path: TmpVarPath() = TmpVarPath(), data_path: DataPath() = DataPath()):
        self.tmp_var_path = tmp_var_path
        self.data_path = data_path

    def read_rating(self):
        return read_data(self.data_path.rating)

    def output_dataset(self):
        rating = self.read_rating()
        rating = self.delete_useless_rating(rating)
        rating = self.normalize_rating(rating)
        rating_matrix = self.transform_into_sparse_matrix(rating)
        return rating_matrix

    @staticmethod
    def delete_useless_rating(rating, threshold_for_dropping_users_according_to_whose_number_of_rating=2):
        """
        drop rating of users that have less than x ratings
        :param rating:
        :param threshold_for_dropping_users_according_to_whose_number_of_rating:
        :return:
        """
        user_rate_count = rating.groupby("USER_MD5").count()
        user_to_drop = user_rate_count[
            user_rate_count[
                "RATING_ID"] < threshold_for_dropping_users_according_to_whose_number_of_rating].index.values
        rating = rating[~rating["USER_MD5"].isin(user_to_drop)]
        return rating

    @staticmethod
    def normalize_rating(rating):
        """
        min max normalize now
        :param rating:
        :return:
        """
        grouper = rating.groupby('USER_MD5')['RATING']
        maxes = grouper.transform('max')
        mins = grouper.transform('min')
        result = rating.assign(RATING=(rating.RATING - mins) / (maxes - mins))
        result = result.fillna(0.5)
        return result

    @staticmethod
    def transform_into_sparse_matrix(rating):
        """
        RUN normalize rating first!
        :param rating:
        :return:
        """
        users = set(rating["USER_MD5"].values)
        user_map = dict(zip(users, range(len(users))))

        movie_id_dict = Movies().get_movie_id_dict()
        movie_map = movie_id_dict[0]

        rating["USER_MD5"] = rating["USER_MD5"].map(user_map)
        rating["MOVIE_ID"] = rating["MOVIE_ID"].map(movie_map)

        rating_matrix = coo_matrix((rating["RATING"], (rating["USER_MD5"], rating["MOVIE_ID"])),
                                   shape=(len(set(rating["USER_MD5"])), len(movie_map.values())))
        return rating_matrix


class Movies:
    def __init__(self, tmp_var_path: TmpVarPath() = TmpVarPath(), data_path: DataPath() = DataPath()):
        self.tmp_var_path = tmp_var_path
        self.data_path = data_path

    def read_movies(self):
        return read_data(self.data_path.movies)

    def get_movie_id_dict(self):
        # hard-code rewrite
        if not os.path.exists(os.path.join(root_dir(), "models", *self.tmp_var_path.movie_id_dict)):
            # lack some movie data, so use rating data in temporary
            movie_id_dict = Movies().transform_movie_id_to_num(Rating().read_rating())
            save_tmp_data(movie_id_dict, self.tmp_var_path.movie_id_dict)
        else:
            movie_id_dict = read_tmp_data(self.tmp_var_path.movie_id_dict)

        return movie_id_dict

    @staticmethod
    def transform_movie_id_to_num(movies: pd.DataFrame):
        """

        :param movies:
        :return: (original_id:new_id, new_id:original_id)
        """
        movies = set(movies["MOVIE_ID"].values)
        org_to_new = dict(zip(movies, range(len(movies))))
        new_to_org = dict(zip(range(len(movies)), movies))
        return org_to_new, new_to_org


if __name__ == '__main__':
    print(read_data(DataPath().users))
