from pyspark import SparkContext
import re
from random import shuffle

sc = SparkContext(appName="Assignment1")

k = 3

data = sc.textFile('dataset2/plot_summaries.txt')                                                                                # reading the dataset
data = data.map(lambda plot: tuple(plot.split("\t")))                                                                            # (movie id, plot)
data = data.map(lambda movie_plot: (movie_plot[0], re.sub(r'[^\w\s]', '', movie_plot[1].lower())))                               # removing punctuation and case-sensitiveness from plot
data = data.map(lambda movie_plot: (movie_plot[0], movie_plot[1].split()))                                                       # (movie id, [list of words in plot]) 
data = data.map(lambda movie_plot: (movie_plot[0], {" ".join(movie_plot[1][i:i+k]) for i in range(len(movie_plot[1])-k + 1)}))   # (movie id, {k-shingles})

vocab = data.flatMap(lambda movie_shingles: movie_shingles[1]).distinct().collect()

one_hot = data.map(lambda movie_shingles: (movie_shingles[0], [1 if shingle in movie_shingles[1] else 0 for shingle in vocab]))

def create_n_hashes(n):
    hashes = []
    for _ in range(n):
        hash = list(range(1,len(vocab)+1))
        shuffle(hash)
        hashes.append(hash)

    return hashes

hashes = create_n_hashes(20)

signatures = one_hot.map(lambda movie_onehot: (movie_onehot[0], [next(func.index(i) for i in range(1, len(vocab) + 1) if movie_onehot[1][func.index(i)] == 1) for func in hashes]))

print(signatures.take(1))