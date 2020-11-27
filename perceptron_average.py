import numpy as np
from random import randint
import utils
import time

from utils import get_aggregation_object


def run(group, problem, alpha, file_prefix, total_time, simulation_time):
    regret_min = []
    regret_avg = []
    r = []
    individual_regret = np.zeros([group.size, problem.iterations])

    start_time = time.time()
    ##########RUN PERCEPTRON AVERAGE###############
    for counter in range(problem.iterations-1):

        weights = group.weights

        x = utils.model(weights, problem)


        print('System recommendation using the average perceptron: ', x.object)

        regret = group.utility_star_x_star - x.get_utility(group.avg_weights_star)
        r.append(regret)

        # Calculate the regret for every individual user
        #
        for i in range(0, group.size):
            usr = group.users[i]
            reg = usr.get_regret(x.phi)
            individual_regret[i, counter] = reg


        if (regret == 0.0):
            final_time = (time.time() - start_time) + total_time
            file = utils.save_to_pickle(x.object, r, regret_min, regret_avg, regret_min, 'average', file_prefix, problem, group, simulation_time, final_time, counter, individual_regret)
            return x.object, simulation_time, file


        # phi_y_bar, s_t = average(group, phi_y_list, alpha)

        improvements = []

        for u in group.users:
            sim_start_time = time.time()
            x_bar = u.step(x.phi, x.object, alpha)
            bar = [np.array(x_bar.phi[0]), np.array(x_bar.phi[1]), np.array(x_bar.phi[2]), np.array(x_bar.phi[3])]
            improvements.append(bar)

            # y, improvements[group.get_index(u)] = u.step(phi_y, alpha)
            temp_simulation_time = time.time() - sim_start_time
            simulation_time += temp_simulation_time
            print('Improvement user ', group.get_index(u), ': ', x_bar.object)

        phi_x_bar = utils.avg_phi_list(improvements, group.problem)



        group.update_weights(phi_x_bar, x.phi)

    x = utils.model(weights, problem)
    regret = group.utility_star_x_star - x.get_utility(group.avg_weights_star)
    r.append(regret)

    # Calculate the regret for every individual user
    #
    for i in range(0, group.size):
        usr = group.users[i]
        reg = usr.get_regret(x.phi)
        individual_regret[i, counter+1] = reg

    final_time = (time.time() - start_time) + total_time
    #y, regret_avg, regret_min, regret_su_avg, regret_su_min, strategy, file_prefix, problem, group, sim_time, runtime, counter
    file = utils.save_to_pickle(x.object, r, regret_min, regret_avg, regret_min, 'average', file_prefix, problem, group, simulation_time, final_time, problem.iterations, individual_regret)
    return x.object, simulation_time, file



