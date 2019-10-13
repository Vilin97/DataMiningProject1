# to run do: exec(open("kmeans.py").read())

import pandas as pd
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import numpy as np
import random

epsilon = 0.01

#path = sys.argv[1]
path = ""
#n = sys.argv[2]
#k = sys.argv[3]

def transform(dataframe,features,normalize): # one hot encode and select features and data points
    df = dataframe.copy()
    n = len(df)
    features_to_drop = [f for f in df.columns if f not in features]

    df["genres"] = df["genres"].map(lambda ls: [g["name"] for g in eval(ls)])

    existing_languages = list(set(df["original_language"]))
    existing_genres = list(set((lambda l: [item for sublist in l for item in sublist])(df["genres"])))

    for lang in existing_languages:
        df[lang] = np.zeros(n)
        df[lang] = df["original_language"].map(lambda l: int(l == lang))
    for genre in existing_genres:
        df[genre] = np.zeros(n)
        df[genre] = df["genres"].map(lambda gs: int(genre in gs))

    df = df.drop(columns = features_to_drop)
    imp_mean = SimpleImputer(missing_values=np.nan)
    imp_mean.fit(df)
    df = pd.DataFrame(imp_mean.transform(df),columns = df.columns)
    if normalize:
        for feature in features:
            df[feature] = (df[feature]-df[feature].mean())/(df[feature].std())
    return df

def distance(a,b):
    return (np.linalg.norm(a-b))**2

def kmeans(dataframe,k,init):
    n = len(dataframe)
    df = dataframe.copy()
    if init == "random":
        indices = []
        for i in range(k):
            r = 0.5
            while True:
                r = random.randint(0,n-1)
                if r not in indices:
                    break
            indices.append(r)
        centers = [df.iloc[i] for i in indices]
    elif init == "k-means++":
        r = random.randint(0,n-1)
        indices = [r]*k
        centers = [df.iloc[r]]*k
        for j in range(k):
            r = random.random()
            distances = [0]*n
            total_distance = 0
            for i in range(n):
                distances[i] = min([distance(df.iloc[i],center) for center in centers])
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
            centers[j] = df.iloc[index]
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
            index = np.argmin([distance(df.iloc[i],center) for center in centers])
            clusters[index].append(df.iloc[i])
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
    df["cluster"] = np.zeros
    for j in range(k):
        for index in cluster_indices[j]:
            df.at[index,"cluster"] = j
    return df,round(loss,2)

def do_kmeans(path,n,k,init):
    data = pd.read_csv(path+"movies.csv")
    dataframe = pd.DataFrame(data).iloc[:n]
    features = ["popularity","revenue","vote_average","vote_count","runtime"]
    df = transform(dataframe,features,True)
    if init == "1d":
        return one_d_kmeans(df,k)
    else:
        return kmeans(df,k,init)[0]

# Problem 3:
def pca_2():
    n = 250
    data = pd.read_csv(path+"movies.csv")
    df = pd.DataFrame(data)
    df["total_votes"] = df["vote_average"]*df["vote_count"]
    df = df.sort_values(by=['total_votes'],ascending = False).iloc[:n]
    features = ["popularity","revenue","vote_average","vote_count","runtime","total_votes"]
    df = transform(df,features,normalize = True)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    principal_df = pd.DataFrame(data=principal_components,columns=['pc1','pc2'])
    principal_df["cluster"] = clustered_df["cluster"]
    return principal_df


# Problem 4:
def unit_cost(i,j,partial_square_sums,partial_sums):
    if i != 0:
        return (partial_square_sums[j] - partial_square_sums[i-1]) - (partial_sums[j] - partial_sums[i-1])**2
    else:
        return partial_square_sums[j] - partial_sums[j]**2

def one_d_kmeans(df,k):
    n = len(df)

    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(df)
    series = pd.Series(pd.DataFrame(principal_component)[0])

    sorted = series.sort_values()

    partial_sums = [0]*n
    partial_sums[0] = sorted.iloc[0]
    partial_square_sums = [0]*n
    partial_square_sums[0] = sorted.iloc[0]**2
    for i in range(1,n):
        partial_sums[i] = partial_sums[i-1]+sorted.iloc[i]
        partial_square_sums[i] = partial_square_sums[i-1]+sorted.iloc[i]**2

    unit_costs = np.array([[unit_cost(i,j,partial_square_sums,partial_sums) for j in range(n)] for i in range(n)]) # unit_costs[i,j] = cost of putting ith thru jth points in one cluster
    boundaries = [0.5]*(k-1)

    costs = np.zeros((n,k)) # costs[i,j] = min cost for j+1 clusters of i+1 first points
    cutoffs = np.zeros((n,k))
    for i in range(n):
        costs[i,0] = unit_costs[0,i]
    for j in range(1,k):
        for i in range(n):
            if i <= j:
                costs[i,j] = 0
            else:
                a = [costs[q,j-1]+unit_costs[q+1,i] for q in range(i)]
                q = np.argmin(a) # elt q is the last in the cluster
                costs[i,j] = a[q]
                cutoffs[i,j] = q

    boundaries = [0]*(k-1)
    i = n-1
    j = k-1
    while j >= 1:
        boundaries[j-1] = int(cutoffs[i,j])
        i = int(cutoffs[i,j])
        j = j - 1

    clusters = [[] for i in range(k)]
    j = 0
    for i in range(n):
        clusters[j].append(sorted.index[i])
        if j < k-1 and boundaries[j] == i:
            j += 1
    df = pd.DataFrame(series)
    df["cluster"] = np.zeros
    for j in range(k):
        for index in clusters[j]:
            df.at[index,"cluster"] = j
    return df
