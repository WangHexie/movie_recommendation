import re

import numpy as np
import pandas as pd

from src.crawler.user_info import UserSpider
from src.data.dataset import Rating, Movies
from src.train.matrix_factorization import SKLNMF


class Recommendation:
    @staticmethod
    def nmf_recommend(user_id, num_of_recommend=20, fake_load=False):
        """

        :param user_id:
        :param num_of_recommend:
        :param fake_load: server use only
        :return:
        """
        rating_matrix = Recommendation.user_id_to_matrix(user_id)
        model = SKLNMF()
        if fake_load:
            print("fake loading")
            model = model.fake_load_model()
            print("fake loaded")
        else:
            model = model.load_model()

        user_embedding = model.transform(rating_matrix)
        movies_embedding = model.components_

        print(user_embedding.shape, movies_embedding.shape)

        result = np.matmul(user_embedding, movies_embedding)[0]

        indexes = result.argsort()[::-1]
        indexes = Recommendation.remove_watched_movie(rating_matrix, indexes)

        if fake_load:
            reverse_dict = Movies().get_id_dict()[1]
            indexes = [reverse_dict[i] for i in indexes]
            movies = indexes[:num_of_recommend]
        else:

            movies = Movies.index_to_movies(indexes[:num_of_recommend])
        return movies

    @staticmethod
    def remove_watched_movie(rating_matrix, indexes):
        watched = rating_matrix.nonzero()[1]
        print([i for i in range(len(indexes)) if indexes[i] in watched])
        indexes = [i for i in indexes if i not in watched]
        return indexes

    @staticmethod
    def user_id_to_matrix(user_id):
        movies = UserSpider().get_all_movies(user_id)
        map_dict = dict(zip(["rating{}-t".format(i) for i in range(1, 6)], range(5)))

        movies["RATING"] = movies["RATING"].map(map_dict)
        movies["USER_MD5"] = np.zeros(len(movies))

        movies = movies.dropna()

        r = Rating()
        rating = r.normalize_rating(movies)
        rating_matrix = r.transform_into_sparse_matrix(rating)
        return rating_matrix


if __name__ == '__main__':
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(Recommendation.nmf_recommend("test", 20))
