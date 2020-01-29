import os
import re
import warnings
from time import sleep
from typing import List

import math
import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd

from src.data.dataset import root_dir


class UserSpider:
    base_url = "https://movie.douban.com/people/"
    url_after = "/collect?start={}&sort=time&rating=all&filter=all&mode=grid"
    num_of_movie_in_a_page = 15

    def __init__(self):
        pass

    @staticmethod
    def get_page(url):
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"}
        r = requests.get(url, verify=False, proxies=None,
                         headers=headers,
                         timeout=10)
        return r.content

    @staticmethod
    def get_rating(movie_item: List[BeautifulSoup]):
        rating = []
        for item in movie_item:
            result = item.select_one("li:not([class='title']) span:not([class='date'])")
            if result is not None:
                rating.append(result["class"][0])
            else:
                rating.append(None)
        return rating

    @staticmethod
    def get_date(movie_item: List[BeautifulSoup]):
        return [item.select_one("span.date").text for item in movie_item]

    @staticmethod
    def get_movie_id(movie_item: List[BeautifulSoup]):
        return [re.findall("\\d+", item.select("div.info a[href]")[0]["href"])[0] for item in movie_item]

    @staticmethod
    def number_of_movie(content: BeautifulSoup):
        return int(re.findall("\\d+", content.title.text)[0])

    @staticmethod
    def get_movie_item(content: BeautifulSoup):
        return content.select("div.item div.info")

    @staticmethod
    def save_data(data: pd.DataFrame, name):
        data.to_csv(os.path.join(root_dir(), "data", name))

    @staticmethod
    def exist(user_id: str):
        return os.path.exists(os.path.join(root_dir(), "data", str(user_id)+".csv"))

    @staticmethod
    def read_data(user_id: str):
        return pd.read_csv(os.path.join(root_dir(), "data", str(user_id)+".csv"), index_col=0)

    def get_all_movies(self, user_id: str):
        """
        add save and load function (not finished)
        :param user_id:
        :return:
        """
        # if file exist
        if self.exist(user_id):
            return self.read_data(user_id)

        data = self._get_all_movies(user_id)
        self.save_data(data, str(user_id)+".csv")
        return data

    def _get_all_movies(self, user_id: str):
        item_start_id = 0
        start_url = self.base_url + user_id + self.url_after.format(item_start_id)

        first_page = self.get_page(start_url)
        soup = BeautifulSoup(first_page)
        number_of_page = math.ceil(self.number_of_movie(soup) / self.num_of_movie_in_a_page)

        full_data = []
        for i in range(number_of_page):
            sleep(1)
            item_start_id = i * self.num_of_movie_in_a_page
            url = self.base_url + user_id + self.url_after.format(item_start_id)
            print(url)
            page = self.get_page(url)
            movie_item = self.get_movie_item(BeautifulSoup(page))

            full_data.append(np.array([self.get_movie_id(movie_item),
                                       self.get_rating(movie_item),
                                       self.get_date(movie_item)]
                                      ).T)
        return pd.DataFrame(np.vstack(full_data), columns=["MOVIE_ID", "RATING", "RATING_TIME"])


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        UserSpider().get_all_movies("test")



