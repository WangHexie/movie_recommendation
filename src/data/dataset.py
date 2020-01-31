import os
import pickle
from pathlib import Path

import pandas as pd
from scipy.sparse import coo_matrix

from src.config.configs import DataPath, TmpVarPath


def root_dir():
    return Path(os.path.abspath(os.path.dirname(__file__))).parent.parent


def read_data(path: tuple):
    return pd.read_csv(os.path.join(root_dir(), "data", *path))


def save_tmp_data(data, path):
    os.makedirs(os.path.join(root_dir(), "models", *path[:-2]), exist_ok=True)
    with open(os.path.join(root_dir(), "models", *path), "wb") as f:
        pickle.dump(data, f)


def read_tmp_data(path):
    with open(os.path.join(root_dir(), "models", *path), "rb") as f:
        data = pickle.load(f)
    return data


def transform_id_to_num(data: pd.DataFrame, column_name):
    """

    :param column_name:
    :param data:
    :return: (original_id:new_id, new_id:original_id)
    """
    rate_count = data.groupby([column_name]).count()
    data = rate_count.sort_values(by=column_name, ascending=False).index.tolist()
    # data = set(data[column_name].values)
    org_to_new = dict(zip(data, range(len(data))))
    new_to_org = dict(zip(range(len(data)), data))
    return org_to_new, new_to_org


class Rating:
    def __init__(self, tmp_var_path: TmpVarPath() = TmpVarPath(), data_path: DataPath() = DataPath()):
        self.tmp_var_path = tmp_var_path
        self.data_path = data_path

    def read_rating(self):
        rating = read_data(self.data_path.rating)
        rating["RATING_TIME"] = pd.to_datetime(rating["RATING_TIME"])
        return rating

    def output_dataset(self, drop_threshold=2):
        rating = self.read_rating()
        rating = self.delete_useless_rating(rating, drop_threshold)
        rating = self.normalize_rating(rating)
        rating_matrix = self.transform_into_sparse_matrix(rating)
        print("matrix shape", rating_matrix.shape)
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
    def rename_id(rating):
        """
        :param rating:
        :param predictio
        n_mode:
        :return:
        """
        # if not prediction_mode:
        #     user_map = User().get_id_dict()[0]
        # else:
        users = set(rating["USER_MD5"].values)
        user_map = dict(zip(users, range(len(users))))
        movie_map = Movies().get_id_dict()[0]

        rating["USER_MD5"] = rating["USER_MD5"].map(user_map).fillna(0)  # in recommendation.not real id.have to fillna
        rating["MOVIE_ID"] = rating["MOVIE_ID"].map(movie_map)

        # drop rating that not included in the dictionary
        rating = rating.dropna()
        return rating

    @staticmethod
    def transform_into_sparse_matrix(rating):
        """
        RUN normalize rating first!
        :param rating:
        :return:
        """
        rating = Rating.rename_id(rating)
        return Rating.dataframe_to_sparse_matrix(rating)

    @staticmethod
    def dataframe_to_sparse_matrix(rating):
        movie_map = Movies().get_id_dict()[0]  # load for the shape

        # TODO: is there any need to use shape? May cause bug or performance problem
        rating_matrix = coo_matrix((rating["RATING"], (rating["USER_MD5"], rating["MOVIE_ID"])),
                                   shape=(int(rating["USER_MD5"].max() + 1), len(movie_map.values())))
        return rating_matrix


class BaseObject:
    def __init__(self):
        self.id_name = None
        self.id_dict_path = None

    def transform_id_to_num(self, movies: pd.DataFrame):
        """

        :param movies:
        :return: (original_id:new_id, new_id:original_id)
        """
        return transform_id_to_num(movies, self.id_name)

    def get_id_dict(self):
        # hard-code rewrite
        if not os.path.exists(os.path.join(root_dir(), "models", *self.id_dict_path)):
            # lack some movie data, so use rating data in temporary
            id_dict = self.transform_id_to_num(Rating().read_rating())
            save_tmp_data(id_dict, self.id_dict_path)
        else:
            id_dict = read_tmp_data(self.id_dict_path)

        return id_dict

    def output_low_num_of_rating_iu(self, data: pd.DataFrame, keep_threshold=2):
        """

        :param keep_threshold:
        :param data:
        :return: (original_id:new_id, new_id:original_id)
        """
        rate_count = data.groupby([self.id_name]).count()
        # data = rate_count.sort_values(by=self.id_name, ascending=False)
        user_to_keep = rate_count[rate_count["RATING_ID"] < keep_threshold].index.values
        return user_to_keep


class Movies(BaseObject):
    def __init__(self, tmp_var_path: TmpVarPath() = TmpVarPath(), data_path: DataPath() = DataPath()):
        super().__init__()
        self.tmp_var_path = tmp_var_path
        self.data_path = data_path
        self.id_name = 'MOVIE_ID'
        self.id_dict_path = tmp_var_path.movie_id_dict

    def read_movies(self):
        return read_data(self.data_path.movies)

    @staticmethod
    def index_to_movies(indexes):
        movies = Movies().read_movies()
        movies = movies.set_index('MOVIE_ID')

        reverse_dict = Movies().get_id_dict()[1]
        indexes = [reverse_dict[i] for i in indexes]

        return movies.loc[indexes]


class User(BaseObject):
    def __init__(self, tmp_var_path: TmpVarPath() = TmpVarPath(), data_path: DataPath() = DataPath()):
        super().__init__()
        self.tmp_var_path = tmp_var_path
        self.data_path = data_path
        self.id_name = 'USER_MD5'
        self.id_dict_path = tmp_var_path.user_id_dict


if __name__ == '__main__':
    print(Rating().output_dataset())
