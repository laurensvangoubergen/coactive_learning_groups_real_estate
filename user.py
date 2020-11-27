import scipy.interpolate
import random
import utils
import numpy as np
import math
from real_estate_object import Real_estate_object


class User:
    def __init__(self, problem, weight_range, nb_optimal_preferences):
        self.problem = problem
        self.preference_weights = self.generate_weights(nb_optimal_preferences, weight_range)

        self.x_star = utils.model(self.preference_weights, problem)
        self.utility_star_x_star = self.get_utility_star(self.x_star.phi)

        self.utility_star_group_x_star_min = None
        self.utility_star_group_x_star_avg = None
        self.current_weights = [np.zeros(problem.types), np.zeros(problem.areas),
                                np.zeros((problem.roomtypes, problem.max_rooms)),
                                np.zeros((problem.features, problem.options))]


    def __getattr__(self, item):
        return item

    def __set__(self, instance, value):
        self.instance = value

    def reset_weights(self):
        self.current_weights = [np.zeros(self.problem.types), np.zeros(self.problem.areas),
                                np.zeros((self.problem.roomtypes, self.problem.max_rooms)),
                                np.zeros((self.problem.features, self.problem.options))]

    def get_regret(self, phi):
        return (self.utility_star_x_star - self.get_utility_star(phi))

    def generate_weights(self, nb_optimal_preferences, weight_range):
        type = np.random.randint(0, weight_range, self.problem.types)
        area = np.random.randint(0, weight_range, self.problem.areas)
        rooms = np.random.randint(0, weight_range, (self.problem.roomtypes, self.problem.max_rooms))
        layout = self.generate_weights_human(nb_optimal_preferences, weight_range)
        return [type, area, rooms, layout]

    def generate_weights_human(self, n, weight_range):
        options = self.problem.options
        features = self.problem.features
        weights = np.zeros((features, options))

        for i in range(features):
            #range for random points
            r = options/n
            x = np.arange(0, options+2, r)

            #choose n points to interpolate between
            rand = np.random.randint(0, weight_range, n)
            y = np.append([0], rand)
            #create interpolation function
            func = scipy.interpolate.interp1d(x, y, kind='cubic')

            #range to apply func to
            xnew = np.linspace(0, options, num=options, endpoint=True)

            weights[i] = func(xnew)

            # import matplotlib.pyplot as plt
            # x = np.arange(0, 40, 1)
            # fig, ax = plt.subplots()
            # ax.bar(x, weights[i], label='User preferences')
            # ax.set(xlabel='Units', ylabel='Preference Values', title='User Preferences')

            # plt.show()
        return weights




    def generate_weights_layout_random(self, weight_range):
        features = self.problem.features
        options = self.problem.options
        values = np.array([])
        for i in range(0, features * options):
            values = np.append(values, random.randint(-weight_range, weight_range))
        values = values.reshape(features, options)
        return values

    def get_utility(self, phi):
        utility = 0
        for i in range(len(phi)):
            utility += np.sum(self.current_weights[i] * phi[i])
        return utility

    def get_utility_star(self, phi):
        utility = 0
        for i in range(len(phi)):
            utility += np.sum(self.preference_weights[i] * phi[i])
        return utility

    def step(self, phi_y, y, alpha):
        return utils.step(phi_y, y, self.x_star.phi, self.preference_weights, alpha, self.problem)

    def update_weights(self, phi_y_bar, phi_y):
        for i in range(len(self.current_weights)):
            self.current_weights[i] = self.current_weights[i] + (
                        np.array(phi_y_bar[i]).astype(float) - np.array(phi_y[i]).astype(float))

