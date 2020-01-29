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
    model_name: list = ("NMF.model",)


@dataclass()
class DataPath:
    rating: list = ("moviedata", "ratings.csv")
    movies: list = ("moviedata", "movies.csv")
    users: list = ("moviedata", "users.csv")


@dataclass()
class TmpVarPath:
    movie_id_dict: list = ("movie_id_dict.pk",)
