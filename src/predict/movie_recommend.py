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
            model = model.fake_load_model()
        else:
            model = model.load_model()

        usre_embedding = model.transform(rating_matrix)
        movies_embedding = model.components_

        result = np.matmul(usre_embedding, movies_embedding)[0]

        indexes = result.argsort()[::-1]
        indexes = Recommendation.remove_watched_movie(rating_matrix, indexes)

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
        movies = movies.dropna()
        movies["RATING"] = movies["RATING"].map(lambda x: int(x[6]))
        movies["USER_MD5"] = np.zeros(len(movies))

        r = Rating()
        rating = r.normalize_rating(movies)
        rating_matrix = r.transform_into_sparse_matrix(rating)
        return rating_matrix


if __name__ == '__main__':
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(Recommendation.nmf_recommend("test", 20))
