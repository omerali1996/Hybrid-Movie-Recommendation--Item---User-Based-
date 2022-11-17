#############################################
# PROJECT: Hybrid Recommender System
#############################################

# Estimation by using user-based & item based recommendation for user giving userId
# Recommend 5 suggestion from each model
#############################################
# Task 1: Preparetion of Data
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


# Read movie and rating .csv files,then merge them
def prep_data():
    movie = pd.read_csv("Cases/Hybrid_Recommender/datasets/movie.csv")
    rating = pd.read_csv("Cases/Hybrid_Recommender/datasets/rating.csv")
    print(5 * "*" + "Movies Analysis" + "*" * 5)
    print("\nGENERAL")
    print(f"\n{movie.info()}")
    print(f"\n\nDESCRİBE \n{movie.describe().T}")
    print(f"\n\nCOLUMNS \n{[col for col in movie.columns]}")
    print(f"\n\nSHAPE\n{movie.shape}")
    print(f"\n\nNULL VALUES\n{movie.isnull().sum()}\n\n")

    print(5 * "*" + "Ratings Analysis" + "*" * 5)
    print("\nGENERAL")
    print(f"\n{rating.info()}")
    print(f"\n\nDESCRİBE \n{rating.describe().T}")
    print(f"\n\nCOLUMNS \n{[col for col in rating.columns]}")
    print(f"\n\nSHAPE\n{rating.shape}")
    print(f"\n\nNULL VALUES\n{rating.isnull().sum()}\n\n")

    df= rating.merge(movie,how="left",on="movieId")
    print(5 * "*" + "Df Analysis" + "*" * 5)
    print("\nGENERAL")
    print(f"\n{df.info()}")
    print(f"\n\nFirst 5 Value \n{df.head().T}")
    print(f"\n\nSHAPE\n{df.shape}")
    return rating,movie,df

def User_Based(perc=0.6,corr_perc=0.65,wars_perc=3.5,howmany_movie=5,plot=False):
    rating,movie,df = prep_data()

    # Analyze how many people rate for each movie
    # Discard from dataframe if total rate below 1000
    # Create a pivot table; index=userId, column=movie title, values =ratings

    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(values="rating",index=["userId"],columns=["title"]).fillna(0)

    import numpy as np
    users = rating["userId"].unique()

    user = np.random.choice(users)

    print(f"\n\nRandom_User ID => {user} Selected  ")

    random_user_df = user_movie_df[user_movie_df.index == user]

    print(f"\n\nRandom_User_Df Shape \n\n{random_user_df.shape}")

    random_user_movies = random_user_df.columns[random_user_df.notna().any()].tolist()

    print(f"\n\nRandom User Movies List\n\n {random_user_movies} and\n {len(random_user_movies)} movies watched. ")
    #############################################
    # Task 2: To be suggested users movies
    #############################################
    movies_watched_df = user_movie_df[random_user_movies]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    percentage = len(random_user_movies) * perc
    users_same_movies = (user_movie_count[user_movie_count["movie_count"] > percentage]["userId"])



    import seaborn as sns
    import matplotlib.pyplot as plt
    #############################################
    # Task 3: Getting most close to be recommended UserIDs
    #############################################

    final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]

    corr_df1 = final_df.T.corr().unstack().sort_values()
    corr_df = pd.DataFrame(corr_df1, columns=["corr"])
    corr_df.index_name = ["userId1", "userId2"]
    corr_df = corr_df.reset_index()


    top_users=corr_df[(corr_df["userId1"] ==user) & (corr_df["corr"]>corr_perc)]["userId2","title"].reset_index(drop=True)
    top_users=top_users.sort_values(ascending=False,by="corr")
    top_users.columns =["userId1","userId","corr"]


    top_users_rating=top_users.merge(rating[["userId", "movieId", "rating"]],how="inner")
    top_users_rating=top_users_rating[top_users_rating["userId"]!=user]
    #############################################
    # Task 4: Calculate Weighted Average Recommendation Score (WARS) and First 5 Movie Recommended
    #############################################

    top_users["weighted_rating"]=top_users["corr"]*top_users["rating"]
    print(f"\n\nTop_Users_Df Info \n\n{top_users.info()}")


    recommendation_df = top_users.groupby(["movieId"]).agg({"weighted_rating":"mean"})
    recommendation_df=recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] >wars_perc].sort_values(by="weighted_rating",ascending=False)

    print(movies_to_be_recommend.merge(movie[["movieId","title"]])["title"][:,howmany_movie])
    movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)] \
                   .sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

    movie[movie["movieId"] == movie_id]["title"].values[0]
    movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]

    user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

    movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0],
                                                    user_movie_df)
    movies_from_item_based[1:howmany_movie+1].index

    if plot:
        sns.heatmap(corr_df, annot=True, fmt="d", linewidth=0.05)
        plt.xlabel("Selected_User")
        plt.ylabel("Recommended User")
        plt.title("Corr Graph")



User_Based()


#############################################
# Adım 6: Item-Based Recommendation
#############################################

def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)




