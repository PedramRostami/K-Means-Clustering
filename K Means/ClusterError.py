import numpy as np
import imp1.KMeans as imp1_A

def calculate_error(k, num_of_iterations):
    data, centroids = imp1_A.clustering(k, num_of_iterations, imp1_A.read_data(imp1_A.data_address))
    errors = []
    for i in range(k):
        cluster_data = data[np.where(data[:, 2] == i), :]
        x_dist = np.subtract(cluster_data[0, :, 0], centroids[i, 0])
        y_dist = np.subtract(cluster_data[0, :, 1], centroids[i, 1])
        dist = np.power(np.add(np.power(x_dist, 2), np.power(y_dist, 2)), 0.5)
        if len(dist) == 0:
            errors.append(0)
        else:
            errors.append(np.sum(dist) / len(dist))
    return errors


if __name__ == '__main__':
    print(calculate_error(imp1_A.k, imp1_A.num_of_iterations))