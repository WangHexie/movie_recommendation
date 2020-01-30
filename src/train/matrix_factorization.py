from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
import numpy as np
from src.config.configs import TrainParam, NMFParam
from src.data.dataset import Rating, save_tmp_data, read_tmp_data


class BayesianSVD:
    def __init__(self, train_param: TrainParam = TrainParam()):
        self.train_param = train_param

    @staticmethod
    def train():
        rating_matrix = Rating().output_dataset()
        size = int(rating_matrix.shape[0] / 2)
        train, test = train_test_split(rating_matrix, test_size=size, train_size=size)


class SKLNMF:
    def __init__(self, train_param: NMFParam = NMFParam()):
        self.train_param = train_param
        self.model = None

    def train(self, rating_matrix=None):
        if rating_matrix is None:
            Rating().output_dataset()
        # train, test = train_test_split(rating_matrix, test_size=size, train_size=size)
        model = NMF(n_components=self.train_param.n_components,
                    init=self.train_param.init,
                    random_state=self.train_param.random_state,
                    verbose=self.train_param.verbose,
                    solver=self.train_param.solver,
                    alpha=self.train_param.alpha,
                    l1_ratio=self.train_param.l1_ratio,
                    max_iter=self.train_param.max_iter)
        w = model.fit_transform(rating_matrix)
        self.model = model
        return w

    def get_movie_embedding(self):
        return self.model.components_.T

    def save_model(self):
        save_tmp_data(self.model, self.train_param.model_name)

    def load_model(self) -> NMF:
        self.model = read_tmp_data(self.train_param.model_name)
        return self.model

    # ONLY for the SERVER!!!!!
    def save_components(self):
        np.savez("components", self.model.components_)

    def load_components(self):
        self.model.components_ = np.load("components.npz")

    def fake_load_model(self):
        model = NMF(n_components=self.train_param.n_components,
                    init=self.train_param.init,
                    random_state=self.train_param.random_state,
                    verbose=self.train_param.verbose,
                    solver=self.train_param.solver,
                    alpha=self.train_param.alpha,
                    l1_ratio=self.train_param.l1_ratio,
                    max_iter=self.train_param.max_iter)
        self.model = model
        self.load_components()
        return self.model


if __name__ == '__main__':
    m = SKLNMF(train_param=NMFParam(max_iter=3000))
    # m.train()
    # m.save_model()
    # m.load_model()
    m.fake_load_model()
