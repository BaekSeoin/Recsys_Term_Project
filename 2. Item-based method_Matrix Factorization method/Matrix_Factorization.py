import numpy as np
import pandas as pd

class Matrix_Factorization:
    def __init__(self, train_matrix,train_set,test_set,user_real, movie_real,test_users):
        fill_mean_func = lambda g: g.fillna(g.mean())

        self.train_matrix = train_matrix
        self.optimized_matrix = train_matrix
        self.train_matrix = np.array(self.train_matrix).T
        self.train_matrix = pd.DataFrame(self.train_matrix)
        self.optimized_matrix = pd.DataFrame(self.optimized_matrix)
        self.train_matrix = self.train_matrix.replace(to_replace=0,value = np.nan)
        self.optimized_matrix = self.optimized_matrix.replace(to_replace=0,value = np.nan)
        self.train_matrix = self.train_matrix.apply(fill_mean_func)
        self.optimized_matrix = self.optimized_matrix.apply(fill_mean_func)
        self.train_matrix = self.train_matrix.to_numpy()
        self.optimized_matrix = self.optimized_matrix.to_numpy()
        self.optimized_matrix = self.optimized_matrix.T


        self.train_set = train_set
        self.test_set = test_set
        self.user_real = user_real
        self.movie_real = movie_real
        self.test_users = test_users

    def SVD(self):
        user_factors, singular_values, item_factors_T = np.linalg.svd(self.train_matrix)

        user_factors_1 = user_factors[:,:400]
        singular_values_1 = singular_values[:400]

        sv_1 = np.zeros((400,400))
        sv_1 = np.diag(singular_values_1)

        item_factors_T_1 = item_factors_T[:400,:]

        self.reconstruction_1 = np.matmul(np.matmul(user_factors_1,sv_1),item_factors_T_1)
        #print(self.reconstruction1.shape)

        user_factors, singular_values, item_factors_T = np.linalg.svd(self.optimized_matrix)
        #optimized # 30, 50, 100
        k = 100
        user_factors_2 = user_factors[:, :k]
        singular_values_2 = singular_values[:k]

        sv_2 = np.zeros((k, k))
        sv_2 = np.diag(singular_values_2)

        item_factors_T_2 = item_factors_T[:k, :]

        self.reconstruction_2 = np.matmul(np.matmul(user_factors_2, sv_2), item_factors_T_2)
        #print(self.reconstruction2.shape)


    def RMSE(self,A,B):
        s = 0
        for i,j in zip(A,B):
            rd = i - j
            s += np.sum(rd ** 2)
        return np.sqrt(float(s) / len(A))

    def calculate_recommendation_score(self):

        predict_svd = []
        predict_optimized = []

        for userId,movieId in self.test_users:
            useridx = self.user_real[userId]
            movieidx = self.movie_real[movieId]
            scores = self.reconstruction_1[useridx, :]
            optimized_scores = self.reconstruction_2[useridx, :]

            """
            try:
                test_movies = self.test_set[useridx]['movieId']
                test_ratings = self.test_set[useridx]['rating']
                pred_ratings = [round(scores[i],4) for i in test_movies]
                optimized_pred_ratings = [round(optimized_scores[i],4) for i in test_movies]

                print(round(self.RMSE(test_ratings,pred_ratings),4))
                print(round(self.RMSE(test_ratings,optimized_pred_ratings),4))
                #print(test_ratings)
                #print(pred_ratings)
            except:
                pass"""

            ans = str(userId) + ',' + str(movieId) + ',' + str(round(scores[movieidx],4))
            predict_svd.append(ans)

            ans = str(userId) + ',' + str(movieId) + ',' + str(round(optimized_scores[movieidx], 4))
            predict_optimized.append(ans)

        return predict_svd, predict_optimized

    def total_RMSE(self):

        svd_rmse = 0
        optimized_rmse = 0

        A = []
        B = []
        C = []

        for useridx in self.test_set.keys():
            scores = self.reconstruction_1[useridx, :]
            optimized_scores = self.reconstruction_2[useridx, :]
            movies = self.test_set[useridx]['movieId']
            ratings = self.test_set[useridx]['rating']
            pred_ratings = [round(scores[i],4) for i in movies]
            optimized_pred_ratings = [round(optimized_scores[i], 4) for i in movies]
            A += ratings
            B += pred_ratings
            C += optimized_pred_ratings

        svd_rmse += self.RMSE(A,B)
        optimized_rmse += self.RMSE(A,C)

        print("SVD_RMSE:",round(svd_rmse,4))
        print("Optimized_RMSE:",round(optimized_rmse,4))

