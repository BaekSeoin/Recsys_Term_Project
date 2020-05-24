# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from Item_based_CF import Item_based_CF
from Matrix_Factorization import Matrix_Factorization


def make_item_user_matrix():
    train = './data/ratings_train.csv'
    test = './data/ratings_test.csv'

    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)

    user_real = dict()
    movie_real = dict()

    train_set = dict()

    nmovies = train_data.movieId.nunique()
    nusers = train_data.userId.nunique()

    train_matrix = np.zeros((nmovies,nusers))

    user_num = 0
    movie_num = 0
    for userId, movieId, rating in zip(train_data.userId, train_data.movieId, train_data.rating):
        try:
            x = user_real[userId]
        except:
            user_real[userId] = user_num
            user_num += 1
        try:
            x = movie_real[movieId]
        except:
            movie_real[movieId] = movie_num
            movie_num += 1

        useridx = user_real[userId]
        movieidx = movie_real[movieId]
        train_matrix[movieidx,useridx] = rating

        try:
            train_set[useridx]['movieId'].append(movieidx)
            train_set[useridx]['rating'].append(rating)
        except:
            train_set[useridx] = dict()
            train_set[useridx]['movieId'] = []
            train_set[useridx]['rating'] = []
            train_set[useridx]['movieId'].append(movieidx)
            train_set[useridx]['rating'].append(rating)


    test_set = dict()

    for userId, movieId, rating in zip(test_data.userId, test_data.movieId, test_data.rating):
        try:
            movieidx = movie_real[movieId]
        except:
            continue
        try:
            useridx = user_real[userId]
        except:
            continue

        try:
            test_set[useridx]['movieId'].append(movieidx)
            test_set[useridx]['rating'].append(rating)
        except:
            test_set[useridx] = dict()
            test_set[useridx]['movieId'] = []
            test_set[useridx]['rating'] = []
            test_set[useridx]['movieId'].append(movieidx)
            test_set[useridx]['rating'].append(rating)

    return train_matrix, train_set,test_set, user_real, movie_real


def read_user_id():
    with open('input.txt', 'r') as f:
        return [tuple(map(int, l.strip().split(','))) for l in f.readlines()]


def write_output(IPredict, MPredict,OPredict):
    with open('output.txt', 'w') as f:
        for i,m,o in zip(IPredict,MPredict,OPredict):
            f.write(i + "\n")
            f.write(m + "\n")
            f.write(o + "\n")


if __name__ == "__main__":
    train_matrix, train_set,test_set,user_real, movie_real = make_item_user_matrix()
    test_users = read_user_id()

    ItemBasedCF = Item_based_CF(train_matrix,train_set,test_set,user_real, movie_real,test_users)
    ItemBasedCF.calculate_movie_movie_sim()
    IPredict = ItemBasedCF.calculate_recommendation_score()
    #ItemBasedCF.total_RMSE()

    MF = Matrix_Factorization(train_matrix,train_set,test_set,user_real, movie_real, test_users)
    MF.SVD()
    MPredict,OPredict = MF.calculate_recommendation_score()
    #MF.total_RMSE()

    write_output(IPredict,MPredict,OPredict)

