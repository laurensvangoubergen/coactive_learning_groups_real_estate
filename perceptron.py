import numpy as np
from random import randint
import utils
import time


def run(strategy, group, problem, alpha, file_prefix, total_time, simulation_time, avg=False):
    regret_min = []
    regret_avg = []
    regret_su_min = []
    regret_su_avg = []
    individual_regret = np.zeros([group.size, problem.iterations])
    start_time = time.time()
    ##########RUN PERCEPTRON AVERAGE###############
    extra_time = 0
    for counter in range(problem.iterations - 1):
        print('Iteration: (', counter, '/', problem.iterations, ')')
        ######USER SELECTION#######
        if (strategy == strategy.LEAST_MISERY):
            user, x_group = utils.get_least_misery_user(group, problem, avg)

        elif (strategy == strategy.RANDOM):
            user, x_group, extra_t = utils.get_random_user(group, problem, avg)
            extra_time += extra_t
        ############################


        #Find object to present to this user
        x = utils.model(user.current_weights, problem)
        print('System recommendation using the ', strategy.value, ' strategy: ', x.object)

        #Calculate regret for the group recommendation
        #note: I use x_group because we want to know group wide regret, x is generated using individual weights
        rgrt_avg = utils.get_regret(x_group, group, aggregation_function='avg')
        regret_avg.append(rgrt_avg)
        rgrt_min = utils.get_regret(x_group, group, aggregation_function='min')
        regret_min.append(rgrt_min)

        #Calculate the regret for the picked user
        rgrt_avg = utils.get_regret(x, group, aggregation_function='avg')
        regret_su_avg.append(rgrt_avg)
        rgrt_min = utils.get_regret(x, group, aggregation_function='min')
        regret_su_min.append(rgrt_min)

        #Calculate the regret for every individual user
        #
        for i in range(0,group.size):
            usr = group.users[i]
            reg = usr.get_regret(x_group.phi)
            individual_regret[i,counter] = reg

        #Algorithm ends when the ideal object is found
        if (rgrt_min == 0.0 and not avg):
            final_time = ((time.time() - start_time) + total_time) - extra_time
            file = utils.save_to_pickle(x_group.object, regret_avg, regret_min, regret_su_avg, regret_su_min, strategy.value, file_prefix, problem, group,
                                 simulation_time, final_time, counter, individual_regret)
            return x_group.object, simulation_time, file
        if (rgrt_avg == 0.0 and avg):
            final_time = ((time.time() - start_time) + total_time) - extra_time
            file = utils.save_to_pickle(x_group.object, regret_avg, regret_min, regret_su_avg, regret_su_min, strategy.value, file_prefix, problem, group,
                                 simulation_time, final_time, counter, individual_regret)
            return x_group.object, simulation_time, file
        # start_time = time.time()

        start_sim_time = time.time()
        #Simulate a step from the chosen user
        x_bar = user.step(x.phi, x.object, alpha)

        temp_simulation_time = time.time() - start_sim_time
        print('Improvement user: ', group.get_index(user), ': ', x_bar.object)
        simulation_time += temp_simulation_time

        #update weights for the chosen user
        user.update_weights(x_bar.phi, x.phi)


    x_final = utils.get_aggregation_object(group, problem, avg)

    regret_avg.append(utils.get_regret(x_final, group, aggregation_function='avg'))
    regret_min.append(utils.get_regret(x_final, group, aggregation_function='min'))
    regret_su_avg.append(utils.get_regret(x_final, group, aggregation_function='avg'))
    regret_su_min.append(utils.get_regret(x_final, group, aggregation_function='min'))

    for i in range(0, group.size):
        usr = group.users[i]
        reg = usr.get_regret(x_final.phi)
        individual_regret[i, counter+1] = reg

    final_time = ((time.time() - start_time) + total_time) - extra_time
    file = utils.save_to_pickle(x_final, regret_avg, regret_min, regret_su_avg, regret_su_min, strategy.value, file_prefix, problem, group, simulation_time,
                         final_time, problem.iterations, individual_regret)
    return x_final, simulation_time, file

