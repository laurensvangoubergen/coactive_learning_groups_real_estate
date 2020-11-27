import datetime
import pickle
import numpy as np
import time

import pymzn
import random
from numpy.random import seed
from numpy.random import randint
import re
import fnmatch
from strategy import Strategy
import os
from real_estate_object import Real_estate_object

def get_least_misery_user(group, problem, avg):
    x_group = get_aggregation_object(group, problem, avg)

    utilities = []
    for u in group.users:
        utilities.append(u.get_utility(x_group.phi))
    minimum = min(utilities)
    print('User utilities: ')
    print(utilities)
    ls_user_indices = [i for i, v in enumerate(utilities) if v == minimum]
    if len(ls_user_indices) > 1:
        ls_user_index = ls_user_indices[randint(0, len(ls_user_indices))]
    else:
        ls_user_index = ls_user_indices[0]
    user = group.users[ls_user_index]
    return user, x_group

def get_random_user(group, problem, avg):
    start = time.time()
    x_group = get_aggregation_object(group, problem, avg)
    t = time.time() - start
    user = group.users[randint(0, group.size)]
    return user, x_group, t


def get_aggregation_object(group, problem, agg):
    weights = []
    for u in group.users:
        weights.append(u.current_weights)
    solns = model_aggregation(weights, problem, agg)
    return solns

#Calculate the regret for an object
def get_regret(x, group, aggregation_function):
    #first find the utilities for the users' ideal objects: u^*(x^*)
    #also collect the utilities for the given object x: u^*(x)
    utilities_star_x_group_star_min = []
    utilities_star_x_group_star_avg = []
    utilities_star_x = []
    testlist = []
    for u in group.users:
        utilities_star_x_group_star_min.append(u.utility_star_group_x_star_min)
        utilities_star_x_group_star_avg.append(u.utility_star_group_x_star_avg)

        utilities_star_x.append(u.get_utility_star(x.phi))
        testlist.append(x.get_utility(u.preference_weights))
    #now we aggregate utilities for both listst according to the aggregation function
    if aggregation_function.lower() == 'min':
        agg_x_star = min(utilities_star_x_group_star_min)
        agg_x = min(utilities_star_x)
    elif aggregation_function.lower() == 'avg':
        agg_x_star = np.average(np.array(utilities_star_x_group_star_avg))
        agg_x = np.average(np.array(utilities_star_x))
    return agg_x_star - agg_x

def get_regret_alt(x, group, aggregation_function):
    # Find the regret for every user for this object: r = u*(x*) - u*(x)

    regret = []
    for u in group.users:
        regret.append(u.utility_star_x_star - u.get_utility_star(x.phi))

    # now we aggregate regret, accrding to the aggregation function
    if aggregation_function.lower() == 'min':
        result = min(regret)
    elif aggregation_function.lower() == 'avg':
        result = sum(regret)/group.size
    return result


####CONSTRAINT SOLVERS#######

def model_aggregation(weights, problem, agg, all_solutions=False):
    nb_users = problem.users
    nb_types = problem.types
    nb_areas = problem.areas
    nb_room_types = problem.roomtypes
    nb_of_rooms = problem.max_rooms
    nb_layout_features = problem.features
    nb_layout_options = problem.options

    type_weights = np.zeros((nb_users, nb_types))
    for u in range(nb_users):
        type_weights[u] = weights[u][0]
    type_weights_mzn = convert_for_minizinc_2d(type_weights, nb_users, nb_types)

    area_weights = np.zeros((nb_users, nb_areas))
    for u in range(nb_users):
        area_weights[u] = weights[u][1]
    area_weights_mzn = convert_for_minizinc_2d(area_weights, nb_users, nb_areas)

    room_weights = np.zeros((nb_users, nb_room_types, nb_of_rooms))
    for u in range(nb_users):
        room_weights[u] = weights[u][2]
    room_weights_mzn = convert_for_minizinc_3d(room_weights, nb_users, nb_room_types, nb_of_rooms)

    layout_weights = np.zeros((nb_users, nb_layout_features, nb_layout_options))
    for u in range(nb_users):
        layout_weights[u] = weights[u][3]
    layout_weights_mzn = convert_for_minizinc_3d(layout_weights, nb_users, nb_layout_features, nb_layout_options)

    # data = {'weights': weights , 'nb_users': nb_users, 'nb_features': nb_features, 'nb_options': nb_options, 'price': price}
    data = {'nb_users': nb_users, 'nb_types': nb_types, 'nb_areas': nb_areas, 'nb_room_types': nb_room_types, 'nb_of_rooms': nb_of_rooms,
            'nb_layout_features': nb_layout_features, 'nb_layout_options': nb_layout_options, 'type_weights': type_weights_mzn,
            'area_weights': area_weights_mzn, 'room_weights': room_weights_mzn, 'layout_weights': layout_weights_mzn, 'price': problem.upper_price, 'total': problem.options}
    # print('Minizinc input: ', str(data).replace(' , ', ',').replace(', ', '; ').replace(': ', '= ').replace('{', '"').replace('}','"').replace("'", ''))
    if agg:
        solns = pymzn.minizinc('minizinc_files/aggregation_avg.mzn', data=data, all_solutions=all_solutions)
    else:
        solns = pymzn.minizinc('minizinc_files/aggregation.mzn', data=data, all_solutions=all_solutions)
    return Real_estate_object(solns)

def model(weights, problem, all_solutions=False):
    nb_types = problem.types
    nb_areas = problem.areas
    nb_room_types = problem.roomtypes
    nb_of_rooms = problem.max_rooms
    nb_layout_features = problem.features
    nb_layout_options = problem.options
    type_weights_mzn = np.array2string(weights[0], separator=',', formatter={'float_kind':lambda x: "%.2f" % x})
    area_weights_mzn = np.array2string(weights[1], separator=',', formatter={'float_kind':lambda x: "%.2f" % x})
    room_weights_mzn = convert_for_minizinc_2d(weights[2].astype(float), nb_room_types, nb_of_rooms)
    layout_weights_mzn = convert_for_minizinc_2d(weights[3].astype(float), nb_layout_features, nb_layout_options)

    data = {'nb_types': nb_types, 'nb_areas': nb_areas, 'nb_room_types': nb_room_types, 'nb_of_rooms': nb_of_rooms, 'nb_layout_features': nb_layout_features,
            'nb_layout_options': nb_layout_options, 'type_weights': type_weights_mzn, 'area_weights': area_weights_mzn, 'room_weights': room_weights_mzn,
            'layout_weights': layout_weights_mzn, 'price': problem.upper_price, 'total': problem.options}
    # print('Minizinc input: ', str(data).replace(' , ', ',').replace(', ', '; ').replace(': ', '= ').replace('{', '"').replace('}','"').replace("'", ''))

    solns = pymzn.minizinc('minizinc_files/features_real_estate.mzn', data=data, all_solutions=all_solutions)
    # object = np.array(features[0]['feature_rep'])
    # phi = np.array(features[0]['terms'])
    return Real_estate_object(solns)
def step(y_phi, y, target_phi, target_weights, alpha, problem, all_solutions= False):
    utility_star_y_star = sum([np.sum(j) for j in [target_phi[i] * target_weights [i] for i in range(len(target_phi))]])
    utility_star_y = sum([np.sum(j) for j in [y_phi[i] * target_weights [i] for i in range(len(y_phi))]])
    desired_utility = alpha*(utility_star_y_star - utility_star_y) + utility_star_y
    # desired_utility = 40

    nb_types = problem.types
    nb_areas = problem.areas
    nb_room_types = problem.roomtypes
    nb_of_rooms = problem.max_rooms
    nb_layout_features = problem.features
    nb_layout_options = problem.options

    type_weights_mzn = np.array2string(target_weights[0], separator=',', formatter={'float_kind': lambda x: "%.2f" % x})
    area_weights_mzn = np.array2string(target_weights[1], separator=',', formatter={'float_kind': lambda x: "%.2f" % x})
    room_weights_mzn = convert_for_minizinc_2d(target_weights[2].astype(float), nb_room_types, nb_of_rooms)
    layout_weights_mzn = convert_for_minizinc_2d(target_weights[3].astype(float), nb_layout_features, nb_layout_options)

    data ={'nb_types': nb_types, 'nb_areas': nb_areas, 'nb_room_types': nb_room_types, 'nb_of_rooms': nb_of_rooms, 'nb_layout_features': nb_layout_features,
            'nb_layout_options': nb_layout_options, 'type_weights': type_weights_mzn, 'area_weights': area_weights_mzn, 'room_weights': room_weights_mzn,
            'layout_weights': layout_weights_mzn, 'type_feature_rep': y[0], 'area_feature_rep': y[1], 'rooms_feature_rep':y[2],
            'layout_feature_rep': y[3], 'desired_utility': desired_utility, 'price': problem.upper_price, 'total': problem.options}
    print('Minizinc input: ',str(data).replace(' , ', ',').replace("', ", '; ').replace(': ', '= ').replace('{', '"').replace('}','"').replace(
              "'", ''))
    solns = pymzn.minizinc('minizinc_files/features_real_estate_step_simple.mzn', data=data, all_solutions=all_solutions)
    s = [0]
    s[0] = {'type_terms': solns[0]['type_terms_bar'], 'area_terms': solns[0]['area_terms_bar'], 'room_terms':solns[0]['room_terms_bar'], 'layout_terms': solns[0]['layout_terms_bar'],
         'type_feature_rep': solns[0]['type_feature_rep_bar'], 'area_feature_rep': solns[0]['area_feature_rep_bar'], 'rooms_feature_rep':solns[0]['rooms_feature_rep_bar'], 'layout_feature_rep': solns[0]['layout_feature_rep_bar']}
    return Real_estate_object(s)


#####OTHER HELPERS######

def convert_for_minizinc_2d(array, nb_features, nb_options):
    string = 'array2d(1..' + str(nb_features) + ', 1..' + str(nb_options) + ','
    flat_array = array.flatten()
    string += '['
    for i in range(len(flat_array)):
        string += str(flat_array[i]) + ','
    string += '])'
    return string

def convert_for_minizinc_3d(array, nb_users, nb_features, nb_options):
    string = 'array3d(1..' + str(nb_users) + ', 1..' + str(nb_features) + ', 1..' + str(nb_options) + ', '

    # string += np.array2string(array.flatten(), separator=', ', formatter={'float_kind':lambda x: "%.1f" % x})
    flat_array = array.flatten()
    string += '['
    for i in range(len(flat_array)):
        string += str(flat_array[i]) + ', '
    string += '])'
    return string

def save_to_pickle(y, regret_avg, regret_min, regret_su_avg, regret_su_min, strategy, file_prefix, problem, group, sim_time, runtime, counter, individual_regret):
    nb_features = problem.features
    nb_options = problem.options
    total = problem.total
    users = problem.users
    nb_iterations = problem.iterations
    if len(regret_avg) < nb_iterations:
        filling_avg = np.repeat(regret_avg[len(regret_avg) - 1], problem.iterations - counter - 1)
        regret_avg = np.append(regret_avg, filling_avg)

        filling_min = np.repeat(regret_min[len(regret_min) - 1], problem.iterations - counter - 1)
        regret_min = np.append(regret_min, filling_min)

    target_min =  group.x_star_min.object
    target_avg = group.x_star_avg.object


    data = {
        'setting': {'options': nb_options, 'features': nb_features, 'users': users, 'total': total,
                    'iterations': nb_iterations},
        'target_min': target_min, 'target_avg': target_avg,
        'y': y,
        'regret_avg': regret_avg, 'regret_min': regret_min, 'regret_su_avg': regret_su_avg,
        'regret_su_min': regret_su_min, 'individual_regret': individual_regret,
        'total_runtime': runtime,
        'simulation_time': sim_time,
        # 'distance': dist
    }
    str_dt = str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    path = 'pickle/rand_weight_range/' + strategy + '/' + file_prefix + '/'
    if not os.path.isdir(path):
        os.mkdir(path)

    with open(path + str_dt + '.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return path + str_dt + '.pickle'


#ONLY FOR AVERAGE PERCEPTRONd#

def avg_phi_list(list, problem):
    all_weights = []
    for u in range(len(list)):
        all_weights.append(list[u])

    arr = reshape(all_weights)

    avg = np.average(arr, axis=0)
    list_avg = reshape_to_list(avg, problem)

    return list_avg

def reshape(list):
    sizes = sum([i.size for i in list[0]])
    array = np.zeros((len(list) , sizes))
    for i in range(len(list)):
        counter = 0
        for a in range(len(list[i])):
            for n in range(len(list[i][a].flatten())):
                array[i][counter] = list[i][a].flatten()[n]
                counter += 1
    return array

def reshape_to_list(array, problem):
    counter = 0
    list = []
    type_weights = np.zeros(problem.types)
    for t in range(problem.types):
        type_weights[t] = array[counter]
        counter += 1
    list.append(type_weights)

    area_weights = np.zeros(problem.areas)
    for a in range(problem.areas):
        area_weights[a] = array[counter]
        counter += 1
    list.append(area_weights)

    room_weights = np.zeros((problem.roomtypes, problem.max_rooms))
    for t in range(problem.roomtypes):
        for r in range(problem.max_rooms):
            room_weights[t][r] = array[counter]
            counter += 1
    list.append(room_weights)

    layout_weights = np.zeros((problem.features, problem.options))
    for f in range(problem.features):
        for o in range(problem.options):
            layout_weights[f][o] = array[counter]
            counter += 1
    list.append(layout_weights)
    return list