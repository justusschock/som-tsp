from sys import argv

import numpy as np
from tqdm import tqdm
from io_helper import read_tsp, normalize
from neuron import generate_network, get_neighborhood, get_route
from distance import select_closest, euclidean_distance, route_distance
from plot import plot_network, plot_route
from kaggle_converter import kaggle_csv_to_tps

import pickle

def main():
    if len(argv) != 2:
        print("Correct use: python src/main.py <filename>.tsp")
        return -1

    if argv[1].endswith(".tsp"):
        tsp_file = argv[1]
    else:
        kaggle_csv_to_tps(argv[1], argv[1].rsplit(".", 1)[0] + ".tsp", "Kaggle Cities")
        tsp_file = argv[1].rsplit(".", 1)[0] + ".tsp"

    problem = read_tsp(tsp_file)

    route = som(problem, 100000)

    problem = problem.reindex(route)

    distance = route_distance(problem)

    print('Route found of length {}'.format(distance))
    print(route)
    with(open(argv[1].rsplit(".", 1)[0] + "_route.pkl")) as f:
        pickle.dump(route, f)

def som(problem, iterations, learning_rate=0.8):
    """Solve the TSP using a Self-Organizing Map."""

    # Obtain the normalized set of cities (w/ coord in [0,1])
    cities = problem.copy()

    cities[['x', 'y']] = normalize(cities[['x', 'y']])

    # The population size is 8 times the number of cities
    n = cities.shape[0] * 8

    # Generate an adequate network of neurons:
    network = generate_network(n)
    print('Network of {} neurons created. Starting the iterations:'.format(n))

    for i in tqdm(range(iterations)):

        # Choose a random city
        city = cities.sample(1)[['x', 'y']].values
        winner_idx = select_closest(network, city)
        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(winner_idx, n//10, network.shape[0])
        # Update the network's weights (closer to the city)
        network += gaussian[:,np.newaxis] * learning_rate * (city - network)
        # Decay the variables
        learning_rate = learning_rate * 0.99997
        n = n * 0.9997

        # Check for plotting interval
        # if not i % 1000:
        #     plot_network(cities, network, name='diagrams/{:05d}.png'.format(i))

        # Check if any parameter has completely decayed.
        if n < 1:
            print('Radius has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
    else:
        print('Completed {} iterations.'.format(iterations))

    plot_network(cities, network, name='diagrams/final.png')

    route = get_route(cities, network)
    plot_route(cities, route, 'diagrams/route.png')
    return route

if __name__ == '__main__':
    main()
