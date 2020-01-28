from sklearn.model_selection import train_test_split
import smurff
from src.data.dataset import Movies, Rating, TmpVarPath, DataPath
from src.config.configs import TrainParam


class BayesianSVD:
    def __init__(self, train_param: TrainParam = TrainParam()):
        self.train_param = train_param

    @staticmethod
    def train():
        rating_matrix = Rating().output_dataset()
        size = int(rating_matrix.shape[0] / 2)
        train, test = train_test_split(rating_matrix, test_size=size, train_size=size)
