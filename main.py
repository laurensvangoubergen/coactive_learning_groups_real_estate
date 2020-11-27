from problem import Problem
import user
import utils
import group
from strategy import Strategy
import time
import numpy as np
import perceptron
import perceptron_average
import os
import random



if __name__ == '__main__':
    #TODO: make console application
    alpha = 0.1

    #lists for setting as follows: [options, features, total, users, iterations, samples, price, nb_optimal_preferences, file_prefix]
    settings = np.array([

                            {'alpha': 0.4, 'options': 20, 'features': 6, 'total': 20, 'users': 4, 'iterations': 150, 'samples': 12,'price': 500000, 'opt_pref': 5, 'file_prefix': 'dell_real_estate_base', 'avg': True, 'avg_per':False},
                            {'alpha': 0.4, 'options': 40, 'features': 4, 'total': 40, 'users': 4, 'iterations': 150, 'samples': 12, 'price': 500000, 'opt_pref': 5, 'file_prefix': 'dell_real_estate_high_price', 'avg': False, 'avg_per':False},
                            {'alpha': 0.4, 'options': 40, 'features': 4, 'total': 40, 'users': 4, 'iterations': 250, 'samples': 12, 'price': 500000, 'opt_pref': 5, 'file_prefix': 'dell_real_estate_high_price', 'avg': False, 'avg_per': False},
                            {'alpha': 0.2, 'options' :40, 'features': 4, 'total': 40, 'users': 4, 'iterations': 150, 'samples': 12, 'price': 300000, 'opt_pref':5, 'file_prefix': 'dell_real_estate_low alpha', 'avg': True, 'avg_per':False},
                            {'alpha': 0.4, 'options': 40, 'features': 4, 'total': 40, 'users': 10, 'iterations': 150, 'samples': 12,'price': 300000, 'opt_pref': 5, 'file_prefix': 'dell_real_estate_large_group', 'avg':True, 'avg_per':True},
                            {'alpha': 0.4, 'options': 40, 'features': 4, 'total': 40, 'users': 10, 'iterations': 250, 'samples': 12, 'price': 300000, 'opt_pref': 5, 'file_prefix': 'dell_real_estate_large_group', 'avg': False, 'avg_per': False}
                         ])


    for s in settings:
        for i in range(s['samples']):


            try:
                file_list = []
                file_prefix = s['file_prefix'] + '_alpha_' + str(s['alpha']) + '_pr_' + str(s['price']) + '_' + str(s['options']) + '_' + str(
                    s['features']) + '_' + str(s['users']) + '_' + str(s['iterations'])
                ###########CREATE SETTING#################
                #we assume that the layout only defines the area of the kitchen and living room, the other area of the other rooms is not defined to reduce complexity
                #possible types of real estates e.g. [house, flat]
                types = 2
                #possible types of areas e.g [city, rural]
                areas = 2
                #possible different types of rooms e.g. [bathrooms, bedrooms] and their amounts so [2,4] is 2 bathrooms and 4 bedrooms
                #the kitchen and living room are fixed in the beginning of the layout, we assume not more than 10 of each roomstype
                roomtypes = 3
                #maximum number of rooms per type
                max_rooms = 3

                #create the problem, for the object with nb of users and iterations
                prblm = Problem(types, areas, s['features'], s['options'], s['price'], roomtypes, max_rooms, s['users'], s['iterations'])

                #store the time right before the entier algorithm starts
                start_time = time.time()

                #group is the collection of users which represent the group
                users = []
                for u in range(prblm.users):
                    wr = random.randint(100,1000)
                    usr = user.User(prblm, weight_range=wr, nb_optimal_preferences=s['opt_pref'])
                    t = time.time() - start_time
                    print(str(u) + '/' + str(prblm.users) + 'in ' + str(t))
                    users.append(usr)

                #create group
                grp = group.Group(users, prblm)


                print('Group target average strategy: ', grp.x_star_avg.object)
                print('Group target least misery strategy: ', grp.x_star_min.object)

                #Everything between this and when start time was stored is part of the simulation
                simulation_time = time.time() - start_time
                total_time = time.time() - start_time

                print('Time to generate objects: ', str(simulation_time))
                file_list = []

                grp.reset_weights()
                y_lm, simulation_time_lm, file = perceptron.run(Strategy.LEAST_MISERY, grp, prblm, s['alpha'], file_prefix,
                                                                     total_time, simulation_time)
                file_list.append(file)
                #simulation_time += simulation_time_lm
                #
                grp.reset_weights()
                y_rand, simulation_time_rand, file = perceptron.run(Strategy.RANDOM, grp, prblm, s['alpha'], file_prefix, total_time, simulation_time)
                # simulation_time += simulation_time_rand
                #
                file_list.append(file)

                if s['avg_per']:
                    grp.reset_weights()
                    y_avg, simulation_time_avg, file = perceptron_average.run(grp, prblm, s['alpha'], file_prefix, total_time, simulation_time)
                    #simulation_time += simulation_time_avg

                    file_list.append(file)

                if s['avg']:
                    grp.reset_weights()
                    y_lm_a, simulation_time_lm_a, file = perceptron.run(Strategy.LEAST_MISERY, grp, prblm, s['alpha'], str(file_prefix) + '_avg_agg',
                                                                         total_time, simulation_time, True)
                    simulation_time += simulation_time_lm_a
                    file_list.append(file)

                    grp.reset_weights()
                    y_rand_a, simulation_time_rand_a, file = perceptron.run(Strategy.RANDOM, grp, prblm, s['alpha'], str(file_prefix) + '_avg_agg',
                                                                             total_time, simulation_time, True)
                    simulation_time += simulation_time_rand_a
                    file_list.append(file)

            except:
                for f in file_list:
                    os.remove(f)
                continue



