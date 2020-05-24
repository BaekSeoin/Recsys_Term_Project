import numpy as np
import pandas as pd

class ContentBased():

    def __init__(self):
        path = './data/movies_w_imgurl.csv'
        self.genre_info = pd.read_csv(path)
        path = './data/tags.csv'
        self.tag_info = pd.read_csv(path)
        path = './data/ratings.csv'
        self.rating_info = pd.read_csv(path)
        self.genre_total_count = self.genre_info.movieId.nunique()
        self.tag_total_count = self.tag_info.movieId.nunique()

    def genre_TF_IDF(self):

        genre_dict = dict()
        movie_genre = dict()
        genre_list = []

        for movieId, genres in zip(self.genre_info.movieId, self.genre_info.genres):
            genres = genres.split('|')
            for genre in genres:

                #몇 개의 장르가 존재하는지 check
                if genre not in genre_list:
                    genre_list.append(genre)

                #각 장르별 movie 개수 count
                try:
                    genre_dict[genre] +=1
                except:
                    genre_dict[genre] = 1

                #각 영화가 속해있는 장르 check
                try:
                    movie_genre[movieId][genre] = True
                except:
                    movie_genre[movieId] = dict()
                    movie_genre[movieId][genre] = True

        genre_list = sorted(genre_list)

        self.movie_representation = []

        for movieId in self.genre_info.movieId:
            new = []
            new.append(movieId)
            for genre in genre_list:
                try:
                    movie_check = movie_genre[movieId][genre] #각 영화가 해당 장르에 포함되는지 check
                    genre_count = genre_dict[genre]
                    IDF = np.log10(self.genre_total_count / genre_count)
                    TF = 1
                    TF_IDF = TF * IDF
                    new.append(TF_IDF)
                except:
                    new.append(0)
            self.movie_representation.append(new)

        self.movie_representation = pd.DataFrame(self.movie_representation)
        genre_list = ['movieId'] + genre_list
        self.movie_representation.columns = genre_list

        print("1. Genre movie representation")
        print("---------------------------------------")


    def tag_TF_IDF(self):

        tag_dict = dict()
        movie_dict = dict()
        for userId, movieId, tags in zip(self.tag_info.userId, self.tag_info.movieId, self.tag_info.tag):
            tags = tags.strip().split(',')

            for tag in tags:

                #각 tag에 있는 movie 수 count
                try:
                    tag_dict[tag][movieId] = True
                except:
                    tag_dict[tag] = dict()
                    tag_dict[tag][movieId] = True

                #한 영화에 대해 tag가 몇 번 출현했는지 count
                try:
                    movie_dict[movieId][tag] += 1
                    movie_dict[movieId]['total'] += 1
                except:
                    try:
                        movie_dict[movieId][tag] = 1
                        movie_dict[movieId]['total'] += 1
                    except:
                        movie_dict[movieId] = dict()
                        movie_dict[movieId][tag] = 1
                        movie_dict[movieId]['total'] = 1

        tag_list = sorted(list(tag_dict.keys()))

        dat = []

        for movieId in self.genre_info.movieId:
            new = []
            new.append(movieId)
            for tag in tag_list:
                try:
                    tag_count = len(list(tag_dict[tag].keys()))
                    TF = movie_dict[movieId][tag] / movie_dict[movieId]['total']
                    IDF = np.log10(self.tag_total_count / tag_count)
                    TF_IDF = TF * IDF
                    new.append(TF_IDF)
                except:
                    new.append(0)
            dat.append(new)

        dat = pd.DataFrame(dat)
        tag_list = ['movieId'] + tag_list
        dat.columns = tag_list
        self.movie_representation  = pd.merge(self.movie_representation, dat, on = 'movieId',how="outer")

        print("2. Tag movie representation")
        print("---------------------------------------")


    def cosine_sim(self,A):
        # https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
        # base similarity matrix (all dot products)
        # replace this with A.dot(A.T).toarray() for sparse representation
        similarity = np.dot(A, A.T)

        # squared magnitude of preference vectors (number of occurrences)
        square_mag = np.diag(similarity)

        # inverse squared magnitude
        inv_square_mag = 1 / square_mag

        # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
        inv_square_mag[np.isinf(inv_square_mag)] = 0

        # inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)

        # cosine similarity (elementwise multiply by inverse magnitudes)
        cosine = similarity * inv_mag
        return cosine.T * inv_mag

    def cosine_similarity(self):
        self.movie_representation = self.movie_representation.set_index("movieId")
        self.movie_movie_sim_matrix = self.cosine_sim(self.movie_representation)

        self.movie_movie_sim_matrix = pd.DataFrame(self.movie_movie_sim_matrix)
        self.movie_movie_sim_matrix.columns = self.movie_representation.index
        self.movie_movie_sim_matrix.index = self.movie_representation.index

        #print(self.movie_movie_sim_matrix.iloc[:6,:6])
        print("3. Calculate movie-movie similarity")
        print("---------------------------------------")

    def recommend(self,user_ids):
        userMovieRating = dict()

        for userId, movieId, rating in zip(self.rating_info.userId, self.rating_info.movieId,self.rating_info.rating):
            try:
                userMovieRating[userId]['movieId'].append(movieId)
                userMovieRating[userId]['rating'].append(rating)
            except:
                userMovieRating[userId] = dict()
                userMovieRating[userId]['movieId'] = []
                userMovieRating[userId]['rating'] = []
                userMovieRating[userId]['movieId'].append(movieId)
                userMovieRating[userId]['rating'].append(rating)

        prediction = []
        for userId in user_ids:
            movies = userMovieRating[userId]['movieId']
            user_sim = np.array(self.movie_movie_sim_matrix.loc[movies,:])
            user_rating = np.array(userMovieRating[userId]['rating']).T
            sim_sum = np.array([i.sum() for i in user_sim.T])

            scores = np.matmul(user_sim.T,user_rating) / (sim_sum + 1)
            scores = pd.DataFrame(scores)
            scores.index = self.genre_info.movieId
            scores = scores.reset_index()
            scores.columns = ["movieId","score"]
            scores = scores.sort_values(by =['score','movieId'], ascending=[False,True])

            index = 0
            for movieId, score in zip(scores.movieId, scores.score):
                if index == 30:
                    break

                prediction.append([str(userId), str(movieId), str(round(score,4))])
                index += 1

        print("4. Recommend")
        print("---------------------------------------")

        return prediction


