import numpy as np
import pandas as pd

class Item_based_CF:

    def __init__(self,train_matrix, train_set,test_set,user_real, movie_real,test_users):
        self.train_matrix = train_matrix
        self.train_set = train_set
        self.test_set = test_set
        self.user_real = user_real
        self.movie_real = movie_real
        self.test_users = test_users

    def cosine_sim(self, A):
        similarity = np.dot(A, A.T)
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        return cosine.T * inv_mag


    def calculate_movie_movie_sim(self):
        self.movie_movie_sim_matrix = pd.DataFrame(self.cosine_sim(self.train_matrix))


    def RMSE(self,A,B):
        s = 0
        for i,j in zip(A,B):
            rd = i - j
            s += np.sum(rd ** 2)
        return np.sqrt(float(s) / len(A))

    def calculate_recommendation_score(self):

        predict = []

        store = dict()

        for userId,movieId in self.test_users:
            useridx = self.user_real[userId]
            movieidx = self.movie_real[movieId]

            try:
                scores = store[useridx]
            except:
                movies = self.train_set[useridx]['movieId']

                user_sim = np.array(self.movie_movie_sim_matrix.loc[movies,:])
                user_rating = np.array(self.train_set[useridx]['rating']).T
                sim_sum = np.array([i.sum() for i in user_sim.T])

                scores = np.matmul(user_sim.T, user_rating) / (sim_sum + 1)
                store[useridx] = scores
            """
            try:
                test_movies = self.test_set[useridx]['movieId']
                test_ratings = self.test_set[useridx]['rating']
                pred_ratings = [round(scores[i],4) for i in test_movies]

                print(round(self.RMSE(test_ratings, pred_ratings),4))
            except:
                pass"""
            ans = str(userId) + ',' + str(movieId) + ',' + str(round(scores[movieidx],4))
            predict.append(ans)

        return predict

    def total_RMSE(self):

        item_based_rmse = 0

        A = []
        B = []

        for useridx in self.test_set.keys():
            movies = self.train_set[useridx]['movieId']

            user_sim = np.array(self.movie_movie_sim_matrix.loc[movies, :])
            user_rating = np.array(self.train_set[useridx]['rating']).T
            sim_sum = np.array([i.sum() for i in user_sim.T])

            scores = np.matmul(user_sim.T, user_rating) / (sim_sum + 1)

            movies = self.test_set[useridx]['movieId']
            ratings = self.test_set[useridx]['rating']
            pred_ratings = [round(scores[i],4) for i in movies]
            A += ratings
            B += pred_ratings

        item_based_rmse += self.RMSE(A,B)

        print("Item_based_RMSE:",round(item_based_rmse,4))