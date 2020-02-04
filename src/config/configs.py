from dataclasses import dataclass


@dataclass()
class TrainParam:
    pass


@dataclass()
class NMFParam(TrainParam):
    n_components: int = 100
    init: str = 'nndsvd'
    random_state: int = 0
    verbose: int = 10
    solver: str = "cd"
    alpha: float = 0.001
    l1_ratio: float = 0.01
    max_iter: int = 200
    model_name: tuple = ("NMF.model",)
    beta_loss:int = 2


@dataclass()
class DataPath:
    rating: tuple = ("moviedata", "ratings.csv")
    movies: tuple = ("moviedata", "movies.csv")
    users: tuple = ("moviedata", "users.csv")


@dataclass()
class TmpVarPath:
    movie_id_dict: tuple = ("movie_id_dict.pk",)
    user_id_dict: tuple = ("user_id.pk",)


evaluate_config = NMFParam(max_iter=100, model_name=("NMF_evaluate.model",), beta_loss=2, solver='mu', init='nndsvda')
