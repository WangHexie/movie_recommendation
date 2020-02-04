import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import shuffle
from sklearn.manifold import TSNE

from src.data.dataset import Movies, Rating
from src.train.matrix_factorization import SKLNMF


def shuffle_sparse_matrix(matrix):
    index = np.arange(np.shape(matrix)[0])
    np.random.shuffle(index)
    return matrix[index, :]


class TSNEVisual:

    def sklnmf_visualization(self, visualize_mode=0, drop_threshold=2):
        """

        :param drop_threshold:
        :param visualize_mode: 0:visualize movie vectors. 1: visualize user vectors
        :return:
        """
        visualization_num = 10000

        model = SKLNMF().load_model()

        if visualize_mode == 0:
            x = model.components_.T[:visualization_num]
            print(x.shape)

        if visualize_mode == 1:
            full_data = shuffle_sparse_matrix(Rating().output_dataset(drop_threshold).tocsr())[:visualization_num]
            x = model.transform(full_data)

        genres = self.get_movie_genres()[:visualization_num]

        x_embedded = self.tsne_lower_dimension(x)
        self.visualization(x_embedded, genres)

    @staticmethod
    def tsne_lower_dimension(data):
        x_embedded = TSNE(n_components=2, n_jobs=7, verbose=1).fit_transform(data)
        return x_embedded

    @staticmethod
    def visualization(x_embedded, genres):
        ts_data = pd.DataFrame(data=x_embedded, columns=["x", "y"])
        ts_data["label"] = genres
        sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Arial']})
        sns.relplot(x="x", y="y", hue="label",
                    sizes=(700, 700), data=ts_data)
        plt.show()

    @staticmethod
    def get_movie_genres():
        """
        SHOULD （but not done）be able to return genres according to input
        :return: all movie genres
        """
        movie_profile = Movies().read()
        movie_profile = movie_profile.set_index('MOVIE_ID')

        movies = list(Movies().get_id_dict()[0].keys())
        movies_types = movie_profile.loc[list(movies)]["GENRES"].map(
            lambda x: x.split("/")[0] if not isinstance(x, float) else "nan").values
        return movies_types


if __name__ == '__main__':
    TSNEVisual().sklnmf_visualization(visualize_mode=0, drop_threshold=2)
