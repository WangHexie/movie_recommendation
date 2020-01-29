from sklearn.neighbors import NearestNeighbors
import numpy as np

from src.data.dataset import Movies, Rating
from src.train.matrix_factorization import SKLNMF


class Search:
    @staticmethod
    def nearest_movies(movies_embedding, movie_id, movie_num=10):
        neigh = NearestNeighbors(5, 0.4)
        neigh.fit(movies_embedding)

        m = Movies()
        movies = np.array(list(m.get_movie_id_dict()[0].keys()))
        index_of_movie = list(movies).index(movie_id)

        indexes = neigh.kneighbors(movies_embedding[index_of_movie].reshape((1, -1)), movie_num)[1][0]

        result = Movies.index_to_movies(indexes)
        print(result)
        return result

    def search(self, movie_id, movie_num=10):
        model = SKLNMF().load_model()
        x = model.components_.T
        return self.nearest_movies(x, movie_id, movie_num)


if __name__ == '__main__':
    Search().search(25820603, 20)
