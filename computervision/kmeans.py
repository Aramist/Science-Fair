'''
Created on Thursday, January 5, 2017 at 00:35
Author: Aramis Tanelus
'''

import random

def kmeans_1d(array, num_centroids, iterations=100):
    '''
    K-Means algorithm in one dimention
    Arguments:
    array: an ndarray containing the data points
    num_centroids: an int representing the number of centroids
    iterations: an int representing the number of times the algorithm will loop
    '''

    #Create two centroids points at random locations
    centroids = [random.randrange(0, 1000) for _ in range(num_centroids)]

    for _ in range(iterations):
        homes = []
        for __ in range(num_centroids):
            homes.append(list())

        #Loop through each point and determine which centroid it belongs to
        for point in array:
            distances = {abs(k-point):k for k in centroids}
            homes[centroids.index(distances[min(distances.keys())])].append(point)

        for i in range(num_centroids):
            centroids[i] = sum(homes[i])/max(len(homes[i]), 1)

    return {centroids[L]:homes[L] for L in range(num_centroids)}
