import os
import pickle
from pathlib import Path

import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

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


class BaseObject:
    def __init__(self):
        self.id_name = None
        self.id_dict_path = None
        self.file_path = None

    def read(self):
        data = read_data(self.file_path)
        return data

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


class Rating(BaseObject):
    def __init__(self, tmp_var_path: TmpVarPath() = TmpVarPath(), data_path: DataPath() = DataPath()):
        super().__init__()
        self.tmp_var_path = tmp_var_path
        self.data_path = data_path
        self.file_path = data_path.rating

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
        user_to_drop = User().output_low_num_of_rating_iu(rating, threshold_for_dropping_users_according_to_whose_number_of_rating)
        rating = rating[~rating["USER_MD5"].isin(user_to_drop)]
        return rating

    @staticmethod
    def normalize_rating(rating):
        """
        min max normalize now  TODO: hard code scalar, or add base number
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


class Movies(BaseObject):
    def __init__(self, tmp_var_path: TmpVarPath() = TmpVarPath(), data_path: DataPath() = DataPath()):
        super().__init__()
        self.tmp_var_path = tmp_var_path
        self.data_path = data_path
        self.id_name = 'MOVIE_ID'
        self.id_dict_path = tmp_var_path.movie_id_dict
        self.file_path = data_path.movies

    @staticmethod
    def index_to_movies(indexes):
        movies = Movies().get_movies_with_genres()
        movies = movies.set_index('MOVIE_ID')

        reverse_dict = Movies().get_id_dict()[1]
        indexes = [reverse_dict[i] for i in indexes]

        return movies.loc[indexes]

    @staticmethod
    def transform_movie_genres():
        pass

    @staticmethod
    def get_movies_with_genres():
        movies = Movies().read()
        movies["GENRES"] = movies["GENRES"].map(lambda x: str(x))  # fix nan problem

        cv = CountVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None,
                             lowercase=True, preprocessor=None, tokenizer=lambda x: x.split("/"), stop_words=None,
                             token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1,
                             max_features=None, vocabulary=None)
        genres = cv.fit_transform(movies["GENRES"])
        genres = genres.todense()
        genres_df = pd.DataFrame(genres, columns=cv.get_feature_names())

        movies_long = pd.concat([movies, genres_df], axis=1)

        # fix name problem
        genres = [('紀錄片 documentary', '纪录片'), ('音樂 music', '音乐'), ('劇情 drama', '剧情'), ('動作 action', '动作'),
                  ('兒童 kids', '儿童'), ('喜劇 comedy', '喜剧'), ('愛情 romance', '爱情'), ('懸疑 mystery', '悬疑'),
                  ('驚悚 thriller', '惊悚'), ('comedy', '喜剧'), ('talk-show', '脱口秀'), ('reality-tv', '真人秀'),
                  ('傳記 biography', '传记'), ('動畫 animation', '动画')]

        for i in genres:
            movies_long[i[1]] = movies_long[i[1]] + movies_long[i[0]]
            movies_long.drop(columns=i[0], inplace=True)

        return movies_long


class User(BaseObject):
    def __init__(self, tmp_var_path: TmpVarPath() = TmpVarPath(), data_path: DataPath() = DataPath()):
        super().__init__()
        self.tmp_var_path = tmp_var_path
        self.data_path = data_path
        self.id_name = 'USER_MD5'
        self.id_dict_path = tmp_var_path.user_id_dict
        self.file_path = data_path.users


if __name__ == '__main__':
    print(Rating().output_dataset())
