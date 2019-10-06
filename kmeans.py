# to run do: exec(open("kmeans.py").read())

import pandas as pd
import sys
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random

epsilon = 0.01

#path = sys.argv[1]
path = ""
#n = sys.argv[2]
#k = sys.argv[3]

features = ["popularity","revenue","vote_average","vote_count","runtime"]

def transform(df,n,k): # one hot encode and select features and data points
    features_to_drop = [f for f in df.columns if f not in features]
    for i in range(n):
        df.at[i,"genres"] = [g["name"] for g in eval(df.loc[i,"genres"])]
        df.at[i,"production_companies"] = [g["name"] for g in eval(df.loc[i,"production_companies"])]

    existing_languages = []
    existing_genres = []
    for i in range(len(df["genres"])):
        if df["original_language"][i] not in existing_languages:
            existing_languages.append(df["original_language"][i])
        gs = df["genres"][i]
        for g in gs:
            if g not in existing_genres:
                existing_genres.append(g)

    for lang in existing_languages:
        df[lang] = np.zeros(n)
        for i in range(n):
            df.at[i,lang] = int(lang == df.at[i,"original_language"])
    for genre in existing_genres:
        df[genre] = np.zeros(n)
        for i in range(n):
            df.at[i,genre] = int(genre in df.at[i,"genres"])

    df = df.drop(columns = features_to_drop)

    for feature in features:
        df[feature] = (df[feature]-df[feature].mean())/(df[feature].std())
    return df

def distance(a,b):
    return (np.linalg.norm(a-b))**2

def kmeans(n,k,init):
    data = pd.read_csv(path+"movies.csv")
    dataframe = pd.DataFrame(data).iloc[:n]

    df = transform(dataframe,n,k)
    if init == "random":
        indices = []
        for i in range(k):
            r = -1
            while True:
                r = random.randint(0,n-1)
                if r not in indices:
                    break
            indices.append(r)
        centers = [df.loc[i] for i in indices]
    elif init == "k-means++":
        r = random.randint(0,n-1)
        indices = [r]*k
        centers = [df.loc[r]]*k
        for j in range(k):
            r = random.random()
            distances = [0]*n
            total_distance = 0
            for i in range(n):
                distances[i] = min([distance(df.loc[i],center) for center in centers])
                total_distance += distances[i]
            r = r*total_distance
            running_sum = 0
            index = 0.5
            for i in range(n):
                running_sum += distances[i]
                if running_sum > r:
                    index = i
                    break
            indices[j] = index
            centers[j] = df.loc[index]
    else:
        print("Invalid init argument. Aborting.")
        return df
    loss = np.inf
    prev_loss = loss
    counter = 1

    while True:
        clusters = [[] for _ in range (k)]
        cluster_indices = [[] for _ in range (k)]
        for i in range(n):
            index = np.argmin([distance(df.loc[i],center) for center in centers])
            clusters[index].append(df.loc[i])
            cluster_indices[index].append(i)

        prev_loss = loss
        loss = 0
        for j in range(k):
            loss = loss + sum([distance(vec,centers[j]) for vec in clusters[j]])

        #if counter%5 == 1:
            #print("Step {}: cost = {}\n".format(counter,loss))

        if np.abs(prev_loss - loss) < epsilon:
            print("Took {} steps to converge for n = {}, k = {}.\nFinal cost = {}".format(counter,n,k,loss))
            break

        for j in range(k):
            centers[j] = sum(clusters[j])/len(clusters[j])
        counter += 1
    return round(loss,3)
