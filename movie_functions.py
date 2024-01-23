# import gzip
# import pickle
# import pandas as pd
# from fastapi import HTTPException
# from fuzzywuzzy import process, fuzz

# class MovieFunctions:
#     def __init__(self, df_path='data/df_v4.csv'):
#         self.df = pd.read_csv(df_path)

#     def load_model(self, model_path):
#         with gzip.open(model_path, 'rb') as f:
#             self.model = pickle.load(f)
#         return self.model

#     def get_all_movies(self):
#         """
#         Return a list of all Movies in database.

#         Returns
#         -------
#         dict
#             A dictionary containing all the movies in the database.

#         Example
#         -------
#         >>> movie_functions = MovieFunctions()
#         >>> movies = movie_functions.get_all_movies()
#         {
#             'movies': [
#                 {
#                     'id': 27205,
#                     'original_title': 'Inception',
#                     'overview': 'Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets is offered a chance to regain his old life as payment for a task considered to be impossible: "inception", the implantation of another person\'s idea into a target\'s subconscious.',
#                     'popularity': 29.108,
#                     'release_date': '2010-07-16',
#                     'vote_average': 8.3,
#                     'vote_count': 22186
#                 },
#                 {
#                     'id': 157336,
#                     'original_title': 'Interstellar',
#                     'overview': 'Interstellar chronicles the adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.',
#                     'popularity': 29.108,
#                     'release_date': '2014-11-05',
#                     'vote_average': 8.3,
#                     'vote_count': 22186
#                 },
#                 ...
#             ]
#         }
#         """
#         return {"movies": self.df[['id', 'original_title']].to_dict(orient='records')}

#     def search_movie_title(self, query):
#         """
#         Search for a movie title that matches the given query.

#         Parameters
#         ----------
#         query : str
#             The query to search for.

#         Returns
#         -------
#         str
#             The title of the movie that best matches the query, or None if no match is found.

#         Example
#         -------
#         >>> movie_functions = MovieFunctions()
#         >>> title = movie_functions.search_movie_title('Inception')
#         'Inception'
#         """
#         movie_titles = self.df['original_title']
#         matches = process.extract(query, movie_titles, scorer=fuzz.ratio)
#         matches = [match for match in matches if match[1] >= 60]
#         if matches:
#             return matches[0][0]
#         else:
#             return None

#     def get_movie_by_title(self, movie_title):
#         """
#         Get a movie by its title.

#         Parameters
#         ----------
#         movie_title : str
#             The title of the movie to get.

#         Returns
#         -------
#         dict
#             A dictionary containing the movie's data, or raises an HTTPException if the movie is not found.

#         Example
#         -------
#         >>> movie_functions = MovieFunctions()
#         >>> movie = movie_functions.get_movie_by_title('Inception')
#         {
#             'movie': [
#                 {
#                     'id': 27205,
#                     'original_title': 'Inception',
#                     'overview': 'Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets is offered a chance to regain his old life as payment for a task considered to be impossible: "inception", the implantation of another person\'s idea into a target\'s subconscious.',
#                     'popularity': 29.108,
#                     'release_date': '2010-07-16',
#                     'vote_average': 8.3,
#                     'vote_count': 22186
#                 }
#             ]
#         }
#         """
#         movie_title = movie_title.lower()
#         movie_title = self.search_movie_title(movie_title)

#         movie = self.df[self.df['original_title'] == movie_title]
#         if movie.empty:
#             raise HTTPException(status_code=404, detail="Movie not found")
#         return {"movie": movie.to_dict(orient='records')}

#     def get_content_based_recommendations(self, movie_title, limit=6,path = 'Cine-API\model\content_based_model_v2.pkl.gz'):
#         """
#         Get content-based recommendations for a movie.

#         Parameters
#         ----------
#         movie_title : str
#             The title of the movie to get recommendations for.
#         limit : int, optional
#             The number of recommendations to get. Defaults to 6.

#         Returns
#         -------
#         dict
#             A dictionary containing the recommended movies.

#         Example
#         -------
#         >>> movie_functions = MovieFunctions()
#         >>> recommendations = movie_functions.get_content_based_recommendations('Inception')
#         {
#             'recommendations': [
#                 {
#                     'id': 27205,
#                     'original_title': 'Inception'
#                 },
#                 {
#                     'id': 157336,
#                     'original_title': 'Interstellar'
#                 },
#                 ...
#             ]
#         }
#         """
#         movie_title = movie_title.lower()
#         cosine_similarities = self.load_model(path)
#         matched_title = self.search_movie_title(movie_title)
#         if matched_title is None:
#             raise HTTPException(status_code=404, detail="Movie not found")
#         movie_index = self.df[self.df['original_title'] == matched_title].index[0]
#         similar_movies = list(enumerate(cosine_similarities[movie_index]))
#         similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
#         similar_movies = similar_movies[1:limit]
#         recommended_movie_titles = [self.df['original_title'].iloc[i[0]] for i in similar_movies]
#         # recommended_movie_titles = [self.df[['id', 'original_title']].iloc[i[0]].to_dict() for i in similar_movies]
#         response = {"recommendations": recommended_movie_titles}
#         return response

#     def get_popular_movies(self, sortby: str, limit: int):
#         """
#         Get popular movies.

#         Parameters
#         ----------
#         sortby : str
#             The field to sort the movies by.
#         limit : int
#             The number of movies to get.

#         Returns
#         -------
#         dict
#             A dictionary containing the popular movies.

#         Example
#         -------
#         >>> movie_functions = MovieFunctions()
#         >>> popular_movies = movie_functions.get_popular_movies('score', 5)
#         {
#             'movies': [
#                 {
#                     'id': 27205,
#                     'original_title': 'Inception'
#                 },
#                 {
#                     'id': 157336,
#                     'original_title': 'Interstellar'
#                 },
#                 ...
#             ]
#         }
#         """
#         if sortby == 'score':
#             sortby = 'vote_average'
#         if sortby == 'title':
#             sortby = 'original_title'

#         sorted_movies = self.df.sort_values(by=sortby, ascending=False)[['id', 'original_title']].head(limit+1)
#         movies_list = sorted_movies.values.tolist()[1:]

#         res = {
#             "movies": [{"id": movie[0], "title": movie[1]} for movie in movies_list]
#         }

#         return res


