import numpy as np
import csv
import matplotlib.pyplot as plt

data_address = '../data/Dataset1.csv'
k = 4
num_of_iterations = 20

def denormalize(array, bound):
    return np.add(np.multiply(array, bound[1] - bound[0]), bound[0])

def labeling(array, label):
    result = np.zeros((np.shape(array)[0], np.shape(array)[1] + 1))
    for i in range(np.shape(array)[1]):
        result[:, i] = array[:, i]
    result[:, np.shape(array)[1]] = label
    return result

def clustering(k, num_of_iterations, data):

    centroids = np.random.rand(k, np.shape(data)[1])
    for i in range(num_of_iterations):
        d_dist = np.zeros((np.shape(data)[0], k))
        dist = np.zeros((np.shape(data)[0], k))
        for j in range(k):
            for l in range(np.shape(data)[1]):
                d_dist = np.subtract(data[:, l], centroids[j, l])
                dist[:, j] = np.add(dist[:, j], np.power(d_dist, 2))
            dist[:, j] = np.power(dist[:, j], 0.5)
        data_clusters = np.argmin(dist, axis=1)
        for j in range(k):
            cluster_indices = np.where(data_clusters == j)
            if np.size(cluster_indices) > 0:
                for l in range(np.shape(data)[1]):
                    centroids[j, l] = data[cluster_indices, l].sum() / np.size(cluster_indices)
        print('iteration ', i + 1, ' finished')
    d_dist = np.zeros((np.shape(data)[0], k))
    dist = np.zeros((np.shape(data)[0], k))
    for j in range(k):
        for l in range(np.shape(data)[1]):
            d_dist = np.subtract(data[:, l], centroids[j, l])
            dist[:, j] = np.add(dist[:, j], np.power(d_dist, 2))
        dist[:, j] = np.power(dist[:, j], 0.5)
    data_clusters = np.argmin(dist, axis=1)
    data = labeling(data, data_clusters)
    return data, centroids

def plot(data, centroids, k):
    colors = ['red', 'blue', 'yellow', 'green', 'black', 'orange', 'cyan', 'magenta', 'purple', 'brown',
                         'pink', 'gray', 'olive', 'aqua', 'azure', 'beige', 'coral', 'darkblue', 'gold', 'lavender',
                         'lightgreen']
    centroids = labeling(centroids, k)
    plot_data = []
    for i in range(k):
        plot_data.append(data[np.where(data[:, np.shape(data)[1] - 1] == i), :])
    plot_data.append(centroids)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for cluster_data, color in zip(plot_data, colors):
        if len(np.shape(cluster_data)) == 3:
            x = cluster_data[0, :, 0]
            y = cluster_data[0, :, 1]
            ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)
        elif len(np.shape(cluster_data)) == 2:
            x = cluster_data[:, 0]
            y = cluster_data[:, 1]
            ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)


    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()

def read_data(address):
    with open(data_address, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = np.array(list(reader)).astype(float)
    return data

if __name__ == '__main__':
    data, centroids = clustering(k, num_of_iterations, read_data(data_address))
    plot(data, centroids, k)
