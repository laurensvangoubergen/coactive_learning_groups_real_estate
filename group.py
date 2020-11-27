import numpy as np
import user
import utils
import statistics
from real_estate_object import Real_estate_object

class Group:
    def __init__(self, users, problem):
        self.users = users
        self.problem = problem
        self.size = len(users)

        ####the optimal object according to min and avg aggregation function###
        self.x_star_min = self.get_x_star(False)
        self.x_star_avg = self.get_x_star(True)
        for u in self.users:
            u.utility_star_group_x_star_min = self.x_star_min.get_utility(u.preference_weights)
            u.utility_star_group_x_star_avg = self.x_star_avg.get_utility(u.preference_weights)

        ###ONLY FOR PERCEPTRON_AVERAGE ###
        self.weights = [np.zeros(problem.types), np.zeros(problem.areas),
                        np.zeros((problem.roomtypes, problem.max_rooms)),
                        np.zeros((problem.features, problem.options))]

        self.avg_weights_star = self.get_avg_weights()
        self.x_star_average_perceptron = utils.model(self.avg_weights_star, problem)
        self.utility_star_x_star = self.x_star_average_perceptron.get_utility(self.avg_weights_star)


    def get_x_star(self, agg):
        weights = []
        for u in self.users:
            weights.append(u.preference_weights)
        return utils.model_aggregation(weights, self.problem, agg)

    def reset_weights(self):
        for u in self.users:
            u.reset_weights()


    def get_index(self, user):
        return self.users.index(user)


##################ONLY FOR AVERGE PERCEPTRON##############
    def get_avg_weights(self):
        all_weights = []
        for u in self.users:
            all_weights.append(u.preference_weights)

        arr = utils.reshape(all_weights)

        avg = np.average(arr, axis=0)
        list = utils.reshape_to_list(avg, self.problem)

        return list

    def update_weights(self, phi_y_bar, phi_y):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (
                        np.array(phi_y_bar[i]).astype(float) - np.array(phi_y[i]).astype(float))