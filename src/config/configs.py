from dataclasses import dataclass


@dataclass()
class TrainParam:
    pass


@dataclass()
class DataPath:
    rating: tuple = ("moviedata", "ratings.csv")
    movies: tuple = ("moviedata", "movies.csv")
    users: tuple = ("moviedata", "users.csv")


@dataclass()
class TmpVarPath:
    movie_id_dict: tuple = ("movie_id_dict.pk")
