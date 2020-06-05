import torch
from torch import nn, optim
from torch.utils.data import (Dataset,DataLoader,TensorDataset)
import tqdm
import numpy as np
import pandas as pd
import copy
from statistics import mean


class MovieLensDataset(Dataset):
    def __init__(self, x, y, user_rating_mean, movie_rating_mean, reconstruction, user_dict, movie_dict,
                 user_most_freq_rating, rating_lst):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.user_rating_mean = user_rating_mean
        self.movie_rating_mean = movie_rating_mean
        self.reconstruction = reconstruction
        self.user_dict = user_dict
        self.movie_dict = movie_dict
        self.user_most_freq_rating = user_most_freq_rating
        self.rating_lst = rating_lst


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        # x = (userId, movieId)
        userId = int(x[0])
        movieId = int(x[1])
        user_mean = self.user_rating_mean[userId]['sum'] / self.user_rating_mean[userId]['total']
        movie_mean = self.movie_rating_mean[movieId]['sum'] / self.movie_rating_mean[movieId]['total']
        user_num = self.user_dict[userId]
        movie_num = self.movie_dict[movieId]
        svd_value = self.reconstruction[user_num, movie_num]

        user_lst = copy.deepcopy(self.user_most_freq_rating[userId])
        user_max = max(user_lst)
        first_user_idx = user_lst.index(user_max)
        user_lst[first_user_idx] = 0
        first_user_idx = self.rating_lst[first_user_idx]
        user_max = max(user_lst)
        second_user_idx = user_lst.index(user_max)
        user_lst[second_user_idx] = 0
        second_user_idx = self.rating_lst[second_user_idx]

        a = [user_mean, movie_mean,svd_value, first_user_idx, second_user_idx]

        a = torch.tensor(a, dtype=torch.float32)
        return x, y, a


class NeuralMatrixFactorization(nn.Module):
    def __init__(self, user_k=3, item_k=3, add_feature = 5, hidden_dim=10):
        super().__init__()
        self.data_preprocessing()
        torch.manual_seed(12345)
        self.user_emb = nn.Embedding(self.max_user, user_k, 0)
        self.item_emb = nn.Embedding(self.max_item, item_k, 0)
        self.mlp = nn.Sequential(
            nn.Linear(user_k + item_k + add_feature, hidden_dim),
            #nn.ReLU(),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.LeakyReLU(0.1),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x,a):
        user_idx = x[:, 0]
        item_idx = x[:, 1]
        user_feature = self.user_emb(user_idx)
        item_feature = self.item_emb(item_idx)
        # 사용자 특이량과 상품 특이량을 모아서 하나의 벡터로 만듦
        out = torch.cat([user_feature, item_feature, a], 1)

        # 모은 벡터를 MLP에 넣는다
        out = self.mlp(out)
        out = torch.sigmoid(out) * 5
        return out.squeeze()


    def delete_users_items(self, train_users,train_movies, test_df):
        train_users_ = dict()
        train_movies_ = dict()
        test_df_ = []
        cnt1 = dict()
        cnt2 = dict()
        for i in train_users:
            train_users_[i] = True
        for i in train_movies:
            train_movies_[i] = True

        for userId, movieId, rating in zip(test_df.userId, test_df.movieId, test_df.rating):
            try:
                x = train_users_[userId]
            except:
                cnt1[userId] = True
                continue
            try:
                y = train_movies_[movieId]
            except:
                cnt2[movieId] = True
                continue
            test_df_.append([userId, movieId,rating])
        test_df_ = pd.DataFrame(test_df_,columns=["userId","movieId","rating"])
        return test_df_


    def data_preprocessing(self):
        self.train_df = pd.read_csv("./data/ratings_train.csv")
        self.test_df = pd.read_csv("./data/ratings_vali.csv")

        train_users = list(set(self.train_df.userId))
        train_movies = list(set(self.train_df.movieId))

        self.test_df = self.delete_users_items(train_users,train_movies, self.test_df)
        self.df = pd.concat([self.train_df, self.test_df])

        self.X = self.df[["userId", "movieId"]].values
        self.Y = self.df[["rating"]].values

        self.max_user, self.max_item = self.X.max(0)
        # np.int64형을 파이썬의 표준 int로 캐스트
        self.max_user = int(self.max_user) + 1
        self.max_item = int(self.max_item) + 1

        # 각 유저의 rating 평균, 각 movie의 rating 평균 계산
        self.user_rating_mean = dict()
        self.movie_rating_mean = dict()

        n_users = self.train_df.userId.nunique()
        n_movies = self.train_df.movieId.nunique()
        self.matrix = np.zeros((n_users,n_movies))

        self.user_dict = dict()
        self.movie_dict = dict()

        user_count = 0
        movie_count = 0

        self.rating_lst = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

        self.user_most_freq_rating = dict()

        for userId, movieId, rating in zip(self.train_df.userId, self.train_df.movieId, self.train_df.rating):
            try:
                self.user_rating_mean[userId]['sum'] += rating
                self.user_rating_mean[userId]['total'] += 1
                idx = self.rating_lst.index(rating)
                self.user_most_freq_rating[userId][idx] += 1
            except:
                self.user_rating_mean[userId] = dict()
                self.user_rating_mean[userId]['sum'] = rating
                self.user_rating_mean[userId]['total'] = 1
                self.user_most_freq_rating[userId] = [0 for i in range(10)]
                idx = self.rating_lst.index(rating)
                self.user_most_freq_rating[userId][idx] += 1

            try:
                self.movie_rating_mean[movieId]['sum'] += rating
                self.movie_rating_mean[movieId]['total'] += 1

            except:
                self.movie_rating_mean[movieId] = dict()
                self.movie_rating_mean[movieId]['sum'] = rating
                self.movie_rating_mean[movieId]['total'] = 1


            try:
                user_num = self.user_dict[userId]
            except:
                self.user_dict[userId] = user_count
                user_count += 1
            try:
                movie_num = self.movie_dict[movieId]
            except:
                self.movie_dict[movieId] = movie_count
                movie_count += 1

            user_num = self.user_dict[userId]
            movie_num = self.movie_dict[movieId]
            self.matrix[user_num,movie_num] = rating

        fill_mean_func = lambda g: g.fillna(g.mean())

        self.matrix = self.matrix.T
        self.item_based_matrix = copy.deepcopy(self.matrix)
        self.matrix = pd.DataFrame(self.matrix)
        self.matrix = self.matrix.replace(to_replace=0,value = np.nan)
        self.matrix = self.matrix.apply(fill_mean_func)
        self.matrix = self.matrix.to_numpy()
        self.matrix = self.matrix.T

        ##SVD
        user_factors, singular_values, item_factors_T = np.linalg.svd(self.matrix)
        k = 11
        user_factors = user_factors[:, :k]
        singular_values = singular_values[:k]

        sv = np.zeros((k, k))
        sv = np.diag(singular_values)

        item_factors_T = item_factors_T[:k, :]

        self.reconstruction = np.matmul(np.matmul(user_factors, sv), item_factors_T)


        # X는 (userId, movieId) 쌍
        self.train_X = self.train_df[["userId", "movieId"]].values
        self.train_Y = self.train_df[["rating"]].values

        self.test_X = self.test_df[["userId", "movieId"]].values
        self.test_Y = self.test_df[["rating"]].values

        # X는 ID이고 정수이므로 int64, Y는 실수이므로 float32의 Tensor로 변환
        self.train_dataset = MovieLensDataset(
            torch.tensor(self.train_X, dtype=torch.int64), torch.tensor(self.train_Y, dtype=torch.float32),
            self.user_rating_mean, self.movie_rating_mean, self.reconstruction, self.user_dict, self.movie_dict,
            self.user_most_freq_rating, self.rating_lst)
        self.test_dataset = MovieLensDataset(
            torch.tensor(self.test_X, dtype=torch.int64), torch.tensor(self.test_Y, dtype=torch.float32),
            self.user_rating_mean, self.movie_rating_mean, self.reconstruction,self.user_dict, self.movie_dict,
            self.user_most_freq_rating, self.rating_lst)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=1024, num_workers=4, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1024, num_workers=4)


    def get_feature(self, userId, movieId):
        user_mean = self.user_rating_mean[userId]['sum'] / self.user_rating_mean[userId]['total']
        movie_mean = self.movie_rating_mean[movieId]['sum'] / self.movie_rating_mean[movieId]['total']

        user_num = self.user_dict[userId]
        movie_num = self.movie_dict[movieId]
        svd_value = self.reconstruction[user_num,movie_num]

        user_lst = copy.deepcopy(self.user_most_freq_rating[userId])
        user_max = max(user_lst)
        first_user_idx = user_lst.index(user_max)
        user_lst[first_user_idx] = 0
        first_user_idx = self.rating_lst[first_user_idx]
        user_max = max(user_lst)
        second_user_idx = user_lst.index(user_max)
        user_lst[second_user_idx] = 0
        second_user_idx = self.rating_lst[second_user_idx]

        a = [[user_mean, movie_mean, svd_value, first_user_idx, second_user_idx]]

        a = torch.tensor(a, dtype=torch.float32)
        return a


def eval_net(net, loader, score_fn=nn.functional.mse_loss, device="cpu"):
    ys = []
    ypreds = []

    for x, y, a in loader:
        x = x.to(device)
        a = a.to(device)
        ys.append(y)
        with torch.no_grad():
            ypred = net(x,a).to("cpu").view(-1)
        ypreds.append(ypred)

    ans = RMSE(torch.cat(ys).squeeze(),torch.cat(ypreds))
    score = score_fn(torch.cat(ys).squeeze(),torch.cat(ypreds))
    return score.item(), ans


def read_user_id():
    with open('input.txt', 'r') as f:
        return [tuple(map(int, l.strip().split(','))) for l in f.readlines()]


def predict(input_users,net):

    ypreds = []
    for query in input_users:
        userId = query[0]
        movieId = query[1]
        query = torch.tensor(query, dtype=torch.int64).view(1,-1)
        a = net.get_feature(userId, movieId)
        pred_rating = net(query,a)
        ypreds.append(str(userId) + "," + str(movieId) + "," + str(round(pred_rating.item(),8)))

    return ypreds


def RMSE(A,B):
    s = 0
    for i,j in zip(A,B):
        rd = i.item() - round(j.item(),2)
        s += np.sum(rd ** 2)
    return np.sqrt(float(s) / len(A))

def model_train():

    net = NeuralMatrixFactorization()

    # net.to("cuda:0")
    opt = optim.Adam(net.parameters(), lr=0.001)
    loss_f = nn.MSELoss()

    NeuralMF_loss = []
    epochs = 4
    for epoch in range(1,epochs+1):
        loss_log = []
        for x, y, a in tqdm.tqdm(net.train_loader):
            # x = x.to("cuda:0")
            # y = y.to("cuda:0")
            o = net(x, a)
            loss = loss_f(o, y.view(-1))
            net.zero_grad()
            loss.backward()
            opt.step()
            loss_log.append(loss.item())
        test_score, ans = eval_net(net, net.test_loader, device="cpu")
        NeuralMF_loss.append(test_score)
        #if epoch % 10 == 0:
        print(epoch, np.sqrt(mean(loss_log)), np.sqrt(test_score),ans, flush=True)

    input_users = read_user_id()
    ypreds = predict(input_users,net)

    #모델 저장
    net.cpu()
    params = net.state_dict()
    torch.save(params, "param.data", pickle_protocol=4)

    return ypreds


def run_pre_trained_model():

    net = NeuralMatrixFactorization()
    #저장된 모델 파라미터 불러오기
    params = torch.load("param.data", map_location="cpu")
    net.load_state_dict(params)

    input_users = read_user_id()
    ypreds = predict(input_users,net)

    return ypreds
