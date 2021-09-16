from src.BCD_DR import ALS_DR
import numpy as np
import matplotlib.pyplot as plt
import pickle
from itertools import cycle
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def initialize_loading(data_shape=[10,10,10], n_components=5):
    ### loading = python dict of [U1, U2, \cdots, Un], each Ui is I_i x R array
    loading = {}
    for i in np.arange(len(data_shape)):  # n_modes = X.ndim -1 where -1 for the last `batch mode'
        loading.update({'U' + str(i): np.random.rand(data_shape[i], n_components)})
    return loading

def Out_tensor(loading):
    ### given loading, take outer product of respected columns to get CPdict
    CPdict = {}
    n_modes = len(loading.keys())
    n_components = loading.get('U0').shape[1]
    print('!!! n_modes', n_modes)
    print('!!! n_components', n_components)

    for i in np.arange(n_components):
        A = np.array([1])
        for j in np.arange(n_modes):
            loading_factor = loading.get('U' + str(j))  ### I_i X n_components matrix
            # print('loading_factor', loading_factor)
            A = np.multiply.outer(A, loading_factor[:, i])
        A = A[0]
        CPdict.update({'A' + str(i): A})
    print('!!! CPdict.keys()', CPdict.keys())

    X = np.zeros(shape=CPdict.get('A0').shape)
    for j in np.arange(len(loading.keys())):
        X += CPdict.get('A' + str(j))

    return X


def ALS_run(X,
            n_components=10,
            iter=100,
            regularizer=None,  # L1 regularizer for each factor matrix
            ini_loading=None,
            beta=None,
            search_radius_const=1000,
            if_compute_recons_error=True,
            save_folder='Output_files',
            subsample_ratio = None,
            output_results=True):
    ALSDR = ALS_DR(X=X,
                        n_components=n_components,
                        ini_loading=None,
                        ini_A=None,
                        ini_B=None,
                        alpha=regularizer)

    result_dict = ALSDR.ALS(iter=iter,
                            ini_loading=ini_loading,
                            beta=beta,
                            search_radius_const=search_radius_const,
                            if_compute_recons_error=if_compute_recons_error,
                            save_folder=save_folder,
                            subsample_ratio=subsample_ratio,
                            output_results=output_results)

    return result_dict


def MU_run(X,
           n_components=10,
           iter=100,
           regularizer=0,
           ini_loading=None,
           if_compute_recons_error=True,
           save_folder='Output_files',
           output_results=True):
    ALSDR = ALS_DR(X=X,
                        n_components=n_components,
                        ini_loading=None,
                        ini_A=None,
                        ini_B=None,
                        alpha=regularizer)

    result_dict = ALSDR.MU(iter=iter,
                           ini_loading=ini_loading,
                           if_compute_recons_error=if_compute_recons_error,
                           save_folder=save_folder,
                           output_results=output_results)
    return result_dict

def plot_benchmark_errors_list(results_list_dict, MU_result, name=1, save_folder=None):

    ALS_results_names = [name for name in results_list_dict.keys()]
    d1 = results_list_dict.get(ALS_results_names[0])
    n_components = d1.get('n_components')

    print('!!! ALS_results_names', ALS_results_names)

    time_records = {}
    ALS_errors = {}
    f_ALS_interpolated = {}
    # for i in np.arange(len(ALS_results_names)):
    #      ALS_errors.update({'ALS_errors' + str(i) : results_list_dict.get(str(ALS_results_names[i])).get('timed_errors_trials')})

    # max duration
    x_all_max = 0
    for i in np.arange(len(ALS_results_names)):
        ALS_errors0 = results_list_dict.get(ALS_results_names[i]).get('timed_errors_trials')
        x_all_max = max(x_all_max, max(ALS_errors0[:, :, -1][:, 0]))

    x_all = np.linspace(0, x_all_max, num=101, endpoint=True)

    for i in np.arange(len(ALS_results_names)):
        ALS_errors0 = results_list_dict.get(ALS_results_names[i]).get('timed_errors_trials')
        time_records.update({'x_all_ALS'+ str(i) : x_all[x_all < min(ALS_errors0[:, :, -1][:, 0])]})

    # x_all_common = x_all_ALS1[range(np.round(len(x_all_ALS1) // 1.1).astype(int))]
    # x_all_MU = x_all_common

    MU_errors = MU_result.get('timed_errors_trials')
    x_all_MU = x_all[x_all < min(MU_errors[:, :, -1][:, 0])]
    n_trials = MU_errors.shape[0]


    # interpolate data and have common carrier
    for j in np.arange(len(ALS_results_names)):
        f_ALS_interpolated0 = []
        ALS_errors0 = results_list_dict.get(ALS_results_names[j]).get('timed_errors_trials')
        for i in np.arange(ALS_errors0.shape[0]):
            f_ALS0 = interp1d(ALS_errors0[i, 0, :], ALS_errors0[i, 1, :], fill_value="extrapolate")
            x_all_ALS0 = time_records.get('x_all_ALS'+ str(j))
            f_ALS_interpolated0.append(f_ALS0(x_all_ALS0))

        f_ALS_interpolated0 = np.asarray(f_ALS_interpolated0)
        f_ALS_interpolated.update({'f_ALS_interpolated'+str(j): f_ALS_interpolated0})

    f_MU_interpolated = []
    for i in np.arange(MU_errors.shape[0]):
        f_MU = interp1d(MU_errors[i, 0, :], MU_errors[i, 1, :], fill_value="extrapolate")
        f_MU_interpolated.append(f_MU(x_all_MU))
    f_MU_interpolated = np.asarray(f_MU_interpolated)
    f_MU_avg = np.sum(f_MU_interpolated, axis=0) / f_MU_interpolated.shape[0]  ### axis-0 : trials
    f_MU_std = np.std(f_MU_interpolated, axis=0)

    # make figure
    color_list = ['r', 'b', 'c', 'k']
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    for i in np.arange(len(ALS_results_names)):
        f_ALS_interpolated0 = f_ALS_interpolated.get('f_ALS_interpolated' + str(i))
        f_ALS_avg0 = np.sum(f_ALS_interpolated0, axis=0) / f_ALS_interpolated0.shape[0]  ### axis-0 : trials
        f_ALS_std0 = np.std(f_ALS_interpolated0, axis=0)
        # print('!!!! f_ALS_avg0_' + str(i), f_ALS_avg0)

        x_all_ALS0 = x_all_ALS0 = time_records.get('x_all_ALS'+ str(i))
        color = color_list[i % len(color_list)]
        markers, caps, bars = axs.errorbar(x_all_ALS0, f_ALS_avg0, yerr=f_ALS_std0,
                                           fmt=color+'-', marker='*', label=ALS_results_names[i], errorevery=5)
        axs.fill_between(x_all_ALS0, f_ALS_avg0 - f_ALS_std0, f_ALS_avg0 + f_ALS_std0, facecolor=color, alpha=0.1)

    # Add MU plots
    markers, caps, bars = axs.errorbar(x_all_MU, f_MU_avg, yerr=f_MU_std,
                                       fmt='g-', marker='x', label='MU', errorevery=5)
    axs.fill_between(x_all_MU, f_MU_avg - f_MU_std, f_MU_avg + f_MU_std, facecolor='g', alpha=0.2)


    # min_max duration
    x_all_min_max = []
    for i in np.arange(len(ALS_results_names)):
        x_all_ALS0 = time_records.get('x_all_ALS'+ str(i))
        x_all_min_max.append(max(x_all_ALS0))

    x_all_min_max = min(x_all_min_max)
    axs.set_xlim(0, x_all_min_max)


    [bar.set_alpha(0.5) for bar in bars]
    # axs.set_ylim(0, np.maximum(np.max(f_OCPDL_avg + f_OCPDL_std), np.max(f_ALS_avg + f_ALS_std)) * 1.1)
    axs.set_xlabel('Elapsed time (s)', fontsize=14)
    axs.set_ylabel('Reconstruction error', fontsize=12)
    plt.suptitle('Reconstruction error benchmarks')
    axs.legend(fontsize=13)
    plt.tight_layout()
    plt.suptitle('Reconstruction error benchmarks', fontsize=13)
    plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.00, 0.00)
    if save_folder is None:
        root = 'Output_files_BCD'
    else:
        root = save_folder

    plt.savefig(root + '/benchmark_plot_errorbar' + '_ntrials_' + str(n_trials) + "_" + "_ncomps_" + str(
        n_components) + "_" + str(name) + ".pdf")




def plot_benchmark_errors(ALS_result0, ALS_result1, ALS_result2, MU_result, name=1, save_folder=None):
    n_components = ALS_result1.get('n_components')

    ALS_errors0 = ALS_result0.get('timed_errors_trials')  # shape (# trials) x (2 for time, error) x (iterations)
    ALS_errors1 = ALS_result1.get('timed_errors_trials')  # shape (# trials) x (2 for time, error) x (iterations)
    ALS_errors2 = ALS_result2.get('timed_errors_trials')
    MU_errors = MU_result.get('timed_errors_trials')
    n_trials = ALS_errors1.shape[0]

    print('!! ALS_errors0.shape', ALS_errors0.shape)
    print('!! ALS_errors1.shape', ALS_errors1.shape)
    print('!! ALS_errors2.shape', ALS_errors2.shape)
    print('!! MU_errors.shape', MU_errors.shape)
    print('!!!!! MU_errors', MU_errors)

    x_all_max = max(min(ALS_errors1[:, :, -1][:, 0]), min(MU_errors[:, :, -1][:, 0]),
                    min(ALS_errors2[:, :, -1][:, 0]))
    x_all = np.linspace(0, x_all_max, num=101, endpoint=True)
    x_all_ALS0 = x_all[x_all < min(ALS_errors0[:, :, -1][:, 0])]
    x_all_ALS1 = x_all[x_all < min(ALS_errors1[:, :, -1][:, 0])]
    x_all_ALS2 = x_all[x_all < min(ALS_errors2[:, :, -1][:, 0])]
    x_all_MU = x_all[x_all < min(MU_errors[:, :, -1][:, 0])]

    x_all_common = x_all_ALS1[range(np.round(len(x_all_ALS1) // 1.1).astype(int))]
    # x_all_MU = x_all_common

    print('!!! x_all', x_all)
    # interpolate data and have common carrier

    f_ALS_interpolated0 = []
    f_ALS_interpolated1 = []
    f_ALS_interpolated2 = []
    f_MU_interpolated = []

    for i in np.arange(MU_errors.shape[0]):
        f_ALS0 = interp1d(ALS_errors0[i, 0, :], ALS_errors0[i, 1, :], fill_value="extrapolate")
        f_ALS_interpolated0.append(f_ALS0(x_all_ALS0))

        f_ALS1 = interp1d(ALS_errors1[i, 0, :], ALS_errors1[i, 1, :], fill_value="extrapolate")
        f_ALS_interpolated1.append(f_ALS1(x_all_ALS1))

        f_ALS2 = interp1d(ALS_errors2[i, 0, :], ALS_errors2[i, 1, :], fill_value="extrapolate")
        f_ALS_interpolated2.append(f_ALS2(x_all_ALS2))

        f_MU = interp1d(MU_errors[i, 0, :], MU_errors[i, 1, :], fill_value="extrapolate")
        f_MU_interpolated.append(f_MU(x_all_MU))

    f_ALS_interpolated0 = np.asarray(f_ALS_interpolated0)
    f_ALS_interpolated1 = np.asarray(f_ALS_interpolated1)
    f_ALS_interpolated2 = np.asarray(f_ALS_interpolated2)
    f_MU_interpolated = np.asarray(f_MU_interpolated)

    f_ALS_avg0 = np.sum(f_ALS_interpolated0, axis=0) / f_ALS_interpolated0.shape[0]  ### axis-0 : trials
    f_ALS_std0 = np.std(f_ALS_interpolated0, axis=0)
    # print('!!! f_ALS_std0', f_ALS_std0)

    f_ALS_avg1 = np.sum(f_ALS_interpolated1, axis=0) / f_ALS_interpolated1.shape[0]  ### axis-0 : trials
    f_ALS_std1 = np.std(f_ALS_interpolated1, axis=0)
    # print('!!! f_ALS_std1', f_ALS_std1)

    f_ALS_avg2 = np.sum(f_ALS_interpolated2, axis=0) / f_ALS_interpolated2.shape[0]  ### axis-0 : trials
    f_ALS_std2 = np.std(f_ALS_interpolated2, axis=0)
    # print('!!! f_ALS_std2', f_ALS_std2)

    f_MU_avg = np.sum(f_MU_interpolated, axis=0) / f_MU_interpolated.shape[0]  ### axis-0 : trials
    f_MU_std = np.std(f_MU_interpolated, axis=0)
    print('!!! f_MU_avg', f_MU_avg)
    print('!!! f_MU_std', f_MU_std)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    markers, caps, bars = axs.errorbar(x_all_ALS0, f_ALS_avg0, yerr=f_ALS_std0,
                                       fmt='r-', marker='*', label='ALS_DR-0.5', errorevery=5)
    axs.fill_between(x_all_ALS0, f_ALS_avg0 - f_ALS_std0, f_ALS_avg0 + f_ALS_std0, facecolor='r', alpha=0.1)

    markers, caps, bars = axs.errorbar(x_all_ALS1, f_ALS_avg1, yerr=f_ALS_std1,
                                       fmt='b-', marker='*', label='ALS_DR-1', errorevery=5)
    axs.fill_between(x_all_ALS1, f_ALS_avg1 - f_ALS_std1, f_ALS_avg1 + f_ALS_std1, facecolor='b', alpha=0.1)

    markers, caps, bars = axs.errorbar(x_all_ALS2, f_ALS_avg2, yerr=f_ALS_std2,
                                       fmt='c-', marker='*', label='ALS', errorevery=5)
    axs.fill_between(x_all_ALS2, f_ALS_avg2 - f_ALS_std2, f_ALS_avg2 + f_ALS_std2, facecolor='c', alpha=0.1)

    markers, caps, bars = axs.errorbar(x_all_MU, f_MU_avg, yerr=f_MU_std,
                                       fmt='g-', marker='x', label='MU', errorevery=5)
    axs.fill_between(x_all_MU, f_MU_avg - f_MU_std, f_MU_avg + f_MU_std, facecolor='g', alpha=0.2)
    axs.set_xlim(0, min(max(x_all_ALS0), max(x_all_ALS1), max(x_all_ALS2), max(x_all_MU)))

    [bar.set_alpha(0.5) for bar in bars]
    # axs.set_ylim(0, np.maximum(np.max(f_OCPDL_avg + f_OCPDL_std), np.max(f_ALS_avg + f_ALS_std)) * 1.1)
    axs.set_xlabel('Elapsed time (s)', fontsize=14)
    axs.set_ylabel('Reconstruction error', fontsize=12)
    plt.suptitle('Reconstruction error benchmarks')
    axs.legend(fontsize=13)
    plt.tight_layout()
    plt.suptitle('Reconstruction error benchmarks', fontsize=13)
    plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.00, 0.00)
    if save_folder is None:
        root = 'Output_files_BCD'
    else:
        root = save_folder

    plt.savefig(root + '/benchmark_plot_errorbar' + '_ntrials_' + str(n_trials) + "_" + "_ncomps_" + str(
        n_components) + "_" + str(name) + ".pdf")


def main():
    loading = {}
    n_components = 5
    iter = 20
    num_repeat = 5
    save_folder = "Output_files/test1"
    # save_folder = "Output_files_BCD_twitter0"

    synthetic_data = True
    run_ALS = True
    run_MU = True
    plot_errors = True

    # Load data
    file_name = "Synthetic"
    if synthetic_data:
        np.random.seed(1)
        U0 = np.random.rand(100, 5*n_components)
        np.random.seed(2)
        U1 = np.random.rand(100, 5*n_components)
        np.random.seed(3)
        U2 = np.random.rand(300, 5*n_components)

        loading.update({'U0': U0})
        loading.update({'U1': U1})
        loading.update({'U2': U2})

        X = Out_tensor(loading) * 10
    else:
        path = "Data/Twitter/top_1000_daily/data_tensor_top1000.pickle"
        dict = pickle.load(open(path, "rb"))
        X = dict[1] * 10000
        file_name = "Twitter"

    print('X.shape', X.shape)

    # X = 10 * X/np.linalg.norm(X)

    search_radius_const = np.linalg.norm(X.reshape(-1,1),1)
    # search_radius_const = 1000

    loading_list = []
    for i in np.arange(num_repeat):
        loading_list.append(initialize_loading(data_shape=X.shape, n_components=n_components))


    if run_ALS:
        ALS_result_list_dict = {}
        ALS_subsample_ratio_list=[None]
        # beta_list = [1 / 2, 1, None]
        beta_list = [5, 1, 0.5, None]
        for subsample_ratio in ALS_subsample_ratio_list:
            print('!!! ALS subsample_ratio:', subsample_ratio)
            for beta in beta_list:
                print('!!! ALS initialized with beta:', beta)
                list_full_timed_errors = []
                iter1 = iter

                if subsample_ratio is not None:
                    iter1 = iter1

                for i in np.arange(num_repeat):

                    result_dict_ALS = ALS_run(X,
                                              n_components=n_components,
                                              iter=iter1,
                                              regularizer=0,
                                              # inverse regularizer on time mode (to promote long-lasting topics),
                                              # no regularizer on on words and tweets
                                              ini_loading=loading_list[i],
                                              beta=beta,
                                              search_radius_const=search_radius_const,
                                              subsample_ratio=subsample_ratio,
                                              if_compute_recons_error=True,
                                              save_folder=save_folder,
                                              output_results=True)
                    time_error = result_dict_ALS.get('time_error')
                    list_full_timed_errors.append(time_error.copy())
                    # print('!!! list_full_timed_errors', len(list_full_timed_errors))

                timed_errors_trials = np.asarray(
                    list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)
                result_dict_ALS.update({'timed_errors_trials': timed_errors_trials})

                save_filename = "ALS_result_" + "beta_" + str(beta) + "_" + "subsample_" + str(subsample_ratio)
                # np.save(save_folder + "/" + save_filename, result_dict_ALS)
                ALS_result_list_dict.update({str(save_filename): result_dict_ALS})
                np.save(save_folder + "/ALS_result_list_dict", ALS_result_list_dict)
                print('ALS_result_list_dict.keys()', ALS_result_list_dict.keys())
                result_dict_ALS = {}

    if run_MU:
        list_full_timed_errors = []
        print('!!! MU initialized')
        for i in np.arange(num_repeat):
            result_dict_MU = MU_run(X,
                                    n_components=n_components,
                                    iter=iter*1.5,
                                    regularizer=0,
                                    ini_loading=loading_list[i],
                                    if_compute_recons_error=True,
                                    save_folder=save_folder,
                                    output_results=True)
            time_error = result_dict_MU.get('time_error')
            list_full_timed_errors.append(time_error.copy())
            # print('!!! list_full_timed_errors', len(list_full_timed_errors))

        timed_errors_trials = np.asarray(
            list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)
        result_dict_MU.update({'timed_errors_trials': timed_errors_trials})

        np.save(save_folder + "/MU_result_" + str(file_name), result_dict_MU)
        print('result_dict_MU.keys()', result_dict_MU.keys())

    if plot_errors:
        save_filename = file_name + ".npy"

        ALS_result_list_dict = np.load(save_folder + "/ALS_result_list_dict.npy", allow_pickle=True).item()

        MU_result = np.load(save_folder + '/MU_result_' + save_filename, allow_pickle=True).item()
        plot_benchmark_errors_list(ALS_result_list_dict, MU_result, name=1, save_folder=save_folder)


        """
        ALS_result0 = np.load(save_folder + '/ALS_result_beta_0.5_' + save_filename, allow_pickle=True).item()
        ALS_result1 = np.load(save_folder + '/ALS_result_beta_1_' + save_filename, allow_pickle=True).item()
        ALS_result2 = np.load(save_folder + '/ALS_result_beta_None_' + save_filename, allow_pickle=True).item()

        MU_result = np.load(save_folder + '/MU_result_' + save_filename, allow_pickle=True).item()
        plot_benchmark_errors(ALS_result0, ALS_result1, ALS_result2, MU_result, name=file_name,
                              save_folder=save_folder)
        """


if __name__ == '__main__':
    main()
