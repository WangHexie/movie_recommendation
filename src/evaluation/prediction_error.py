import numpy as np
from sklearn.metrics import mean_absolute_error

from src.config.configs import NMFParam
from src.data.dataset import Rating
from src.train.matrix_factorization import SKLNMF


class Evaluation:

    @staticmethod
    def split_train_test(rating):
        rating = Rating().delete_useless_rating(rating, 3)
        rating = Rating().normalize_rating(rating)
        rating = Rating().rename_id(rating)
        rating = rating.sort_values(by="RATING_TIME")

        test_data = rating.groupby(by="USER_MD5").tail(1)
        train_data = rating.drop(index=test_data.index)
        return train_data, test_data

    @staticmethod
    def model_train(train_data, model):
        """
        UNDONE
        :param train_data:
        :param model:
        :return:
        """
        # model = SKLNMF(NMFParam(max_iter=100, model_name=("NMF_evalutate.model",)))
        w = model.train(model)

    @staticmethod
    def model_evaluate():
        train_data, test_data = Evaluation.split_train_test(Rating().read_rating())

        model = SKLNMF(NMFParam(max_iter=2000, model_name=("NMF_evaluate.model",)))

        matrix = Rating.dataframe_to_sparse_matrix(train_data)

        w = model.train(matrix)
        h = model.get_movie_embedding()

        left = w[test_data["USER_MD5"].values]
        right = h[test_data["MOVIE_ID"].values]

        prediction = np.multiply(left, right)
        prediction = prediction.sum(axis=1)

        return mean_absolute_error(test_data["RATING"].values, prediction)


if __name__ == '__main__':
    print(Evaluation.model_evaluate())