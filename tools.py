import numpy as np
from os import listdir, mkdir
from os.path import *
import pickle
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

def iterations_needed(file):
    regret = get_from_pickle('regret_min', file)
    trimmed = np.trim_zeros(regret, 'b')
    return len(trimmed)

def avg_iterations_needed(path):
    files = [str(path + '/' + f) for f in listdir(path)]
    nb_files = len(files)
    iterations = np.array([])
    for f in range(nb_files):
        i = iterations_needed(files[f])
        iterations = np.append(iterations, i)
    return np.average(iterations)


def get_from_pickle(element, path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data[element]

def avg_total_runtime(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    nb_files = len(files)
    times = np.array([])
    for f in range(nb_files):
        times = np.append(times, get_from_pickle('total_runtime', str(path + '/' + files[f])))
    avg = np.average(times)
    return avg

def avg_simulation_time(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    nb_files = len(files)
    times = np.array([])
    for f in range(nb_files):
        times = np.append(times, get_from_pickle('simulation_time', str(path + '/' + files[f])))
    avg = np.average(times)
    return avg

def avg_runtime(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    nb_files = len(files)
    times = np.array([])
    for f in range(nb_files):
        total = get_from_pickle('total_runtime', str(path + '/' + files[f]))
        sim = get_from_pickle('simulation_time', str(path + '/' + files[f]))
        times = np.append(times, total - sim)
    avg = np.average(times)
    return avg

def plot_runtimes():
    path = 'pickle/average/dell_real_estate_base_alpha_0.4_pr_300000_40_4_2_150'

    runtime_avg_med_group = avg_runtime(path)

    sim_avg_med_group = avg_simulation_time(path)
    path = 'pickle/average/dell_real_estate_large_group_alpha_0.4_pr_300000_40_4_10_150'

    runtime_avg_lrg_group = avg_runtime(path)

    sim_avg_lrg_group = avg_simulation_time(path)
    path = 'pickle/least_misery/dell_real_estate_base_alpha_0.4_pr_300000_40_4_4_150'

    runtime_lm_med_group = avg_runtime(path)

    sim_lm_med_group = avg_simulation_time(path)
    path = 'pickle/least_misery/dell_real_estate_high_price_alpha_0.4_pr_500000_40_4_4_150'

    runtime_lm_med_group_high_price = avg_runtime(path)

    sim_lm_med_group_high_price = avg_simulation_time(path)
    path = 'pickle/least_misery/dell_real_estate_large_group_alpha_0.4_pr_300000_40_4_10_150'

    runtime_lm_lrg_group = avg_runtime(path)

    sim_lm_lrg_group = avg_simulation_time(path)
    path = 'pickle/random/dell_real_estate_base_alpha_0.4_pr_300000_40_4_4_150'

    runtime_rnd_med_group = avg_runtime(path)

    sim_rnd_med_group = avg_simulation_time(path)
    path = 'pickle/random/dell_real_estate_high_price_alpha_0.4_pr_500000_40_4_4_150'

    runtime_rnd_med_group_high_price = avg_runtime(path)

    sim_rnd_med_group_high_price = avg_simulation_time(path)
    path = 'pickle/random/dell_real_estate_large_group_alpha_0.4_pr_300000_40_4_10_150'

    runtime_rnd_lrg_group = avg_runtime(path)

    sim_rnd_lrg_group = avg_simulation_time(path)
    labels = ['Medium group', 'High price', 'Large group']
    lm_times = [runtime_lm_med_group, runtime_lm_med_group_high_price, runtime_lm_lrg_group]
    rnd_times = [runtime_rnd_med_group, runtime_rnd_med_group_high_price, runtime_rnd_lrg_group]

    sim_lm_times = [sim_lm_med_group, sim_lm_med_group_high_price, sim_lm_lrg_group]
    sim_rnd_times = [sim_rnd_med_group, sim_rnd_med_group_high_price, sim_rnd_lrg_group]


    x = np.arange(len(labels))


    ax = plt.subplot(111)
    width = 0.35
    ax.bar(x - width / 2, lm_times, width, color='b', label='Most unhappy')
    ax.bar(x + width / 2, rnd_times, width, color='orange', label='Random')
    ax.set_ylabel('Runtime')
    ax.legend()

    ax.set_title('Runtime for 150 iterations (without simulation)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)


    plt.show()

    ax_sim = plt.subplot(111)

    ax_sim.bar(x - width / 2, lm_times, width, color='b', label='Runtime')
    ax_sim.bar(x + width / 2, sim_lm_times, width, color='orange', label='Simulation time')
    ax_sim.set_ylabel('Time')

    ax_sim.set_title('Simulation vs runtime most unhappy')
    ax_sim.set_xticks(x)
    ax_sim.set_xticklabels(labels)

    ax_sim.legend()


    plt.show()

    ax_sim_rnd = plt.subplot(111)

    ax_sim_rnd.bar(x - width / 2, rnd_times, width, color='b', label='Runtime')
    ax_sim_rnd.bar(x + width / 2, sim_rnd_times, width, color='orange', label='Simulation time')
    ax_sim_rnd.set_ylabel('Time')

    ax_sim_rnd.set_title('Simulation vs runtime random')
    ax_sim_rnd.set_xticks(x)
    ax_sim_rnd.set_xticklabels(labels)

    ax_sim_rnd.legend()

    plt.show()



def avg_from_dir(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    nb_files = len(files)
    regret_arr = np.array([])
    setting = get_from_pickle('setting', str(path + '/' + files[0]))
    iterations = setting['iterations']
    for f in range(nb_files):
        regret_arr = np.append(regret_arr, get_from_pickle('regret', str(path + '/' + files[f])))
    regret_arr = regret_arr.reshape((nb_files, iterations))
    avg = np.average(regret_arr, axis=0)
    return avg

def avg_and_error_from_dir(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    nb_files = len(files)
    regret_arr = np.array([])
    setting = get_from_pickle('setting', str(path + '/' + files[0]))
    iterations = setting['iterations']
    for f in range(nb_files):
        regret_arr = np.append(regret_arr, get_from_pickle('regret', str(path + '/' + files[f])))
    regret_arr = regret_arr.reshape((nb_files, iterations))
    err = stats.sem(regret_arr)
    avg = np.average(regret_arr, axis=0)
    return avg, err


#get the average from the average regret for the path
def get_avg_from_avg_dir(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    nb_files = len(files)
    regret_arr = np.array([])
    setting = get_from_pickle('setting', str(path + '/' + files[0]))
    iterations = setting['iterations']
    for f in range(nb_files):
        regret_arr = np.append(regret_arr, get_from_pickle('regret_avg', str(path + '/' + files[f])))
    regret_arr = regret_arr.reshape((nb_files, iterations))
    avg = np.average(regret_arr, axis=0)
    return avg

def get_avg_and_error_from_avg_dir(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    nb_files = len(files)
    regret_arr = np.array([])
    setting = get_from_pickle('setting', str(path + '/' + files[0]))
    iterations = setting['iterations']
    for f in range(nb_files):
        regret_arr = np.append(regret_arr, get_from_pickle('regret_avg', str(path + '/' + files[f])))
    regret_arr = regret_arr.reshape((nb_files, iterations))
    err = stats.sem(regret_arr)
    avg = np.average(regret_arr, axis=0)
    return avg, err

#get the average from the minimum regret for the path
def get_avg_from_min_dir(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    nb_files = len(files)
    regret_arr = np.array([])
    setting = get_from_pickle('setting', str(path + '/' + files[0]))
    iterations = setting['iterations']
    for f in range(nb_files):
        regret_arr = np.append(regret_arr, get_from_pickle('regret_min', str(path + '/' + files[f])))
    regret_arr = regret_arr.reshape((nb_files, iterations))
    avg = np.average(regret_arr, axis=0)
    return avg

#get the average from the minimum regret for the path
def get_avg_and_error_from_min_dir(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    nb_files = len(files)
    regret_arr = np.array([])
    setting = get_from_pickle('setting', str(path + '/' + files[0]))
    iterations = setting['iterations']
    for f in range(nb_files):
        regret_arr = np.append(regret_arr, get_from_pickle('regret_min', str(path + '/' + files[f])))
    regret_arr = regret_arr.reshape((nb_files, iterations))
    err = stats.sem(regret_arr)
    avg = np.average(regret_arr, axis=0)
    return avg, err

def get_avg_and_error_from_min_dir_individual(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    nb_files = len(files)
    setting = get_from_pickle('setting', str(path + '/' + files[0]))
    iterations = setting['iterations']
    users = setting['users']
    out_avg = np.zeros([users,iterations])
    out_err = np.zeros_like(out_avg)
    for u in range(users):
        regret_arr = np.array([])

        for f in range(nb_files):
            regrets = get_from_pickle('individual_regret', str(path + '/' + files[f]))
            regret_arr = np.append(regret_arr, regrets[u])
        regret_arr = regret_arr.reshape((nb_files, iterations))
        out_err[u] = stats.sem(regret_arr)
        out_avg[u] = np.average(regret_arr, axis=0)
    return out_avg, out_err

def plot_save(line1, line2, iterations, title, path, line3 = 0):
    x = np.arange(0, iterations, 1)
    fig, ax = plt.subplots()
    ax.plot(x, line1, color='tab:blue', label='Most unhappy strategy')
    ax.plot(x, line2, color='tab:orange', label='Average strategy')
    if not(line3 == 0):
        ax.plot(x, line3, color='tab:green', label='Random strategy')
    else:
        pass
    ax.legend()
    ax.set(xlabel='nb of Iterations', ylabel='Regret', title=title)
    #  plt.ylim(top=200)
    plt.ylim(bottom=0)

    plt.savefig(path)
    plt.show()

def plot_with_err(line1, error1, label1, line2, error2, label2, title, ylabel):
    x = np.arange(0, len(line1), 1)
    fig, ax = plt.subplots()
    ax.plot(x, line1, color='tab:blue', label=label1, linewidth=0.75)
    ax.plot(x, line2, color='tab:orange', label=label2, linewidth=0.75)
    ax.legend()
    ax.set(xlabel='Iterations', ylabel=ylabel, title = title)
    plt.fill_between(x, line1-error1, line1+error1, color = '0.85')
    plt.fill_between(x, line2-error2, line2+error2, color = '0.85')
    plt.show()
#Need to only take every fourth datapoint from the least onnoying GWPP
def plot_with_err_three(line1, error1, label1, line2, error2, label2, line3, error3, label3, title, iterations):
    x = np.arange(0, iterations, 1)
    fig, ax = plt.subplots()
    ax.plot(x, line1, color='tab:blue', label=label1, linewidth=0.75)
    ax.plot(x, line2, color='tab:orange', label=label2, linewidth=0.75)
    ax.plot(x, line3, color='tab:green', label=label3, linewidth=0.75)
    ax.legend()
    ax.set(xlabel='Queries', ylabel='Regret', title = title)
    plt.fill_between(x, line1-error1, line1+error1, color = '0.85')
    plt.fill_between(x, line2-error2, line2+error2, color = '0.85')
    plt.fill_between(x, line3-error3, line3+error3, color = '0.85')
    plt.show()

def plot_avg_lm_and_ind_users(path, title):
    avg_ind, err_ind = get_avg_and_error_from_min_dir_individual(path)
    lm, lm_err = get_avg_and_error_from_min_dir(path)
    x=np.arange(0, len(lm), 1)
    fig, ax=plt.subplots()
    ax.plot(x, lm, color='tab:red', label='Least misery', linewidth=0.75)
    for i in range(len(avg_ind)):
        ax.plot(x, avg_ind[i], color=np.random.rand(3,), label='User '+str(i+1), linewidth=0.75)
    ax.legend()
    ax.set(xlabel='Iterations',ylabel='Regret', title=title)
    plt.show()

def plot_lm_and_ind_users(path, title):
    ind = get_from_pickle('individual_regret', path)
    lm = get_from_pickle('regret_min', path)
    x=np.arange(0, len(lm), 1)
    fig, ax=plt.subplots()
    ax.plot(x, lm, color='tab:red', label='Least misery', linewidth=0.75)
    for i in range(len(ind)):
        ax.plot(x, ind[i], color=np.random.rand(3,), label='User '+str(i+1), linewidth=0.75)
    ax.legend()
    ax.set(xlabel='Iterations',ylabel='Regret', title=title)
    plt.show()



def save_plot_with_err(line1, error1, label1, line2, error2, label2, title):
    iterations = len(line1)
    x = np.arange(0, iterations, 1)
    fig, ax = plt.subplots()
    ax.plot(x, line1, color='tab:blue', label=label1, linewidth=0.75)
    ax.plot(x, line2, color='tab:orange', label=label2, linewidth=0.75)
    ax.legend()
    ax.set(xlabel='Iterations', ylabel='Regret', title = title)
    plt.fill_between(x, line1-error1, line1+error1, color = '0.85')
    plt.fill_between(x, line2-error2, line2+error2, color = '0.85')
    plt.savefig('plots/' + str(title) + '.png')
    plt.show()


def save_plot_lm_rand(lm_path, title):
    rand_path = lm_path.replace('least_misery','random')
    lm, lm_err = get_avg_and_error_from_min_dir(lm_path)
    rand, rand_err = get_avg_and_error_from_min_dir(rand_path)
    save_plot_with_err(lm, lm_err, 'Least Misery', rand, rand_err, 'Random', title)
    return


def save_plot_all_avg(lm_path, title):
    rand_path = lm_path.replace('least_misery','random')
    avg_path = lm_path.replace('least_misery','average')
    lm, lm_err = get_avg_and_error_from_avg_dir(lm_path)
    rand, rand_err = get_avg_and_error_from_avg_dir(rand_path)
    avg, avg_err = get_avg_and_error_from_avg_dir(avg_path)
    save_plot_with_err(lm, lm_err, 'Least Misery Strategy', rand, rand_err, 'Random Strategy', avg, avg_err, 'Average Strategy', title)
    return



def plot(line1, line2, iterations, title):
    x = np.arange(0, iterations, 1)
    fig, ax = plt.subplots()
    ax.plot(x, line1, color='tab:blue', label='Least misery strategy')
    ax.plot(x, line2, color='tab:orange', label='Random strategy')
    ax.legend()
    ax.set(xlabel='nb of Iterations', ylabel='Regret', title=title)
    #  plt.ylim(top=200)
    plt.ylim(bottom=0)

    plt.show()

def plot_all():
    path_avg = 'pickle/average/'
    path_rand = 'pickle/random/'
    path_lm = 'pickle/least_misery/'
    all_dirs = [f for f in listdir(path_avg) if (isdir(join(path_avg, f)) and 'min_avg' not in f)]
    for dir in all_dirs:
        avg_avg = avg_from_dir(path_avg + dir)
        avg_lm = avg_from_dir(path_lm + dir.replace('average', 'least_misery'))
        avg_rand = avg_from_dir(path_rand + dir.replace('average','random'))
        plot_save(avg_avg, avg_rand, len(avg_avg), str(dir), 'plots/'+ str(dir.replace('average_', '')) + '.png', avg_lm)
    return

#plot and save all avg regrets with the given prefix
def plot_all_avg(name):
    path_avg = 'pickle/average/'
    path_rand = 'pickle/random/'
    path_lm = 'pickle/least_misery/'
    all_dirs = [f for f in listdir(path_avg) if (isdir(join(path_avg, f)) and 'min_avg' in f)]
    for dir in all_dirs:
        avg_avg = get_avg_from_avg_dir(path_avg + dir)
        avg_lm = get_avg_from_avg_dir(path_lm + dir.replace('average', 'least_misery'))
        avg_rand = get_avg_from_avg_dir(path_rand + dir.replace('average', 'random'))
        plot_save(avg_avg, avg_rand, len(avg_avg), str(dir), 'plots/' + str(dir.replace('average_', name)) + '.png',
                  avg_lm)
    return

#plot and save all minimum regrets with the given prefix
def plot_all_min(name):
    path_rand = 'pickle/random/'
    path_lm = 'pickle/least_misery/'
    all_dirs = [f for f in listdir(path_rand)]
    for dir in all_dirs:
        lm, lm_err = get_avg_and_error_from_min_dir(path_lm + dir.replace('random', 'least_misery'))
        rnd, rnd_err = get_avg_and_error_from_min_dir(path_rand + dir)
        save_plot_with_err(lm, lm_err, 'Least Misery', rnd, rnd_err, 'Random', str(dir.replace('random', '').replace('dell_real_estate_', name)))
    return