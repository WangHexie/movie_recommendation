import numpy as np
from sklearn.metrics import mean_absolute_error

from src.config.configs import NMFParam, evaluate_config
from src.data.dataset import Rating
from src.train.matrix_factorization import SKLNMF


class Evaluation:

    @staticmethod
    def split_train_test(rating):
        r = Rating()
        rating = r.delete_useless_rating(rating, 3)
        rating = r.normalize_rating(rating)
        rating = r.rename_id(rating)
        rating = rating.sort_values(by="RATING_TIME")

        test_data = rating.groupby(by="USER_MD5").tail(1)
        train_data = rating.drop(index=test_data.index)
        return train_data, test_data

    @staticmethod
    def model_train(param: NMFParam, train_data):
        """
        UNDONE
        :param param:
        :param train_data:
        :return:
        """
        model = SKLNMF(param)
        w = model.train(train_data)
        model.save_model()
        h = model.get_movie_embedding()
        return w, h

    @staticmethod
    def load_exist_model(train_data):
        """
        UNDONE
        :param param:
        :param train_data:
        :return:
        """
        model = SKLNMF()
        model = model.load_model()
        w = model.transform(train_data)
        h = model.components_.T
        return w, h

    @staticmethod
    def model_evaluate(param: NMFParam, mode=0):
        train_data, test_data = Evaluation.split_train_test(Rating().read_rating())

        matrix = Rating.dataframe_to_sparse_matrix(train_data)

        if mode == 0:
            w, h = Evaluation.model_train(param, matrix)
        else:
            w, h = Evaluation.load_exist_model(matrix)

        left = w[test_data["USER_MD5"].values]
        right = h[test_data["MOVIE_ID"].values]

        prediction = np.multiply(left, right)
        prediction = prediction.sum(axis=1)

        return mean_absolute_error(test_data["RATING"].values, prediction)


if __name__ == '__main__':
    print(Evaluation.model_evaluate(evaluate_config, mode=1))