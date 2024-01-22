from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from movie_functions import get_all_movies, get_movie_by_title, search_movie_title, get_content_based_recommendations,load_model
from movie_functions import get_popular_movies as popular_movies
from fastapi import Query


# Reading data
df = pd.read_csv('data/df_v4.csv')


cosine_similarity_matrix = load_model('model/content_based_model_v2.pkl.gz')

app = FastAPI(
    title="CineAPI",
    description="A simple API for movie recommendations",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Recommendations",
            "description": "Endpoints for getting movie recommendations"
        },
        {
            "name": "Movies",
            "description": "Endpoints for working with movie data"
        },
    ]
)

origins = [
    "http://localhost:3000",  # React app address
    # add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root() -> dict:
    """
    **Description:**
    Root endpoint of the API.

    **Returns:**
    A welcome message.

    **Example:**

    `curl http://localhost:8000/`
    """
    return {
        "message": "Welcome to CineAPI!"
        }

@app.get("/movies", tags=["Movies"])
async def get_movies() -> dict:
    """
    **Description:**
    Retrieves a list of all available movies.

    **Returns:**
    A list of movie titles.

    **Example:**

    `curl -X 'GET' \
  'http://127.0.0.1:8000/movies' \
  -H 'accept: application/json'`
    
    **Example Response:**
    ```json
    {
    "movies": [
        {
        "id": 135397,
        "original_title": "jurassic world"
        },
        {
        "id": 76341,
        "original_title": "mad max fury road"
        },
        {
        "id": 262500,
        "original_title": "insurgent"
        }
    ]
    }
    ```
    """
    return get_all_movies(df)

@app.get("/movies/{movie_title}", tags=["Movies"])

async def get_movie_details(movie_title: str) -> dict:
    """
    **Description:**
    Retrieves details of a specific movie.

    **Parameters:**
    - movie_title (str): The title of the movie to retrieve details for.

    Returns:
    A dictionary containing the movie's details, or an HTTP 404 error if not found.

    **Example:**

    `curl http://localhost:8000/movies/The%20Matrix`

    **Example Response:**
    ```json
    {
    "movie": [
        {
        "id": 603,
        "popularity": 7.753899,
        "original_title": "the matrix",
        "cast": "Keanu Reeves Laurence Fishburne Carrie-Anne Moss Hugo Weaving Gloria Foster",
        "director": "Lilly Wachowski|Lana Wachowski",
        "keywords": "saving the world artificial intelligence man vs machine philosophy prophecy",
        "runtime": 136,
        "genres": "Action Science Fiction",
        "release_date": "3/30/1999",
        "vote_count": 6351,
        "vote_average": 7.8,
        "release_year": 1999,
        "budget_adj": 82470329.34,
        "revenue_adj": 606768749.7,
        "profit": 524298420.36,
        "release_month": 3
        }
    ]
    }
    ```
    """
    return get_movie_by_title(movie_title, df)

@app.get("/recommendations/{movie}", tags=["Recommendations"])
async def get_movie_recommendations(
    movie: str ,
    limit: int = Query(5, ge=1, le=10, description="The number of recommendations to return")
) -> dict:
    """
    **Description:**
    Retrieves top N recommendations for a specific movie.

    **Parameters:**
    - movie (str): The title of the movie to get recommendations for.
    - limit (int, optional): The number of recommendations to return. Defaults to 5.

    **Returns:**
    A list of the top N recommended movie titles.

    **Example:**

    `curl -X 'GET' \
  'http://127.0.0.1:8000/recommendations/spd%20man?limit=5' \
  -H 'accept: application/json'`

    **Example Response:**
    ```json
    {
    "recommendations": [
        {
        "id": 557,
        "original_title": "spider man"
        },
        {
        "id": 559,
        "original_title": "spider man 3"
        },
        {
        "id": 102382,
        "original_title": "the amazing spider man 2"
        },
        {
        "id": 1930,
        "original_title": "the amazing spider man"
        }
    ]
    }
    ```
    """
    return get_content_based_recommendations(movie, cosine_similarity_matrix, df, limit)


@app.get("/popular", tags=["Recommendations"])
async def get_popular_movies(
    sortby: str = Query('score', description="The metric to sort by", enum=['score', 'title', 'release_year']),
    limit: int = Query(10, ge=1, description="The number of movies to return")
) -> dict:
    """
    **Description:**
    Retrieves a list of popular movies sorted by a given metric.

    **Parameters:**
    - sortby (str, optional): The metric to sort by. Valid values are `score`, `title`, and `release_year`. Defaults to `score`.
    - limit (int, optional): The number of movies to return. Defaults to 10.

    **Returns:**
    A list of movie titles sorted by the given metric.

    **Example:**

    `curl -X 'GET' \
    'http://127.0.0.1:8000/popular?sortby=score&limit=3' \
    -H 'accept: application/json'`

    **Example Response:**
    ```json
    {
    "movies": [
        {
        "id": 278,
        "title": "the shawshank redemption"
        },
        {
        "id": 238,
        "title": "the godfather"
        },
        {
        "id": 244786,
        "title": "whiplash"
        }
    ]
    }
    ```
    """
    return popular_movies(df,sortby=sortby, limit=limit)