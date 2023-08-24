from BMM_DR import ALS_DR
import numpy as np
import matplotlib.pyplot as plt
import pickle
from itertools import cycle
from scipy.interpolate import interp1d
import datetime
import scipy.sparse as sp

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def initialize_loading(data_shape=[10,10,10], n_components=5):
    ### loading = python dict of [U1, U2, \cdots, Un], each Ui is I_i x R array
    loading = {}
    for i in np.arange(len(data_shape)):  # n_modes = X.ndim -1 where -1 for the last `batch mode'
        loading.update({'U' + str(i): np.random.rand(data_shape[i], n_components)})
    return loading

def out(loading, drop_last_mode=False):
    ### given loading, take outer product of respected columns to get CPdict
    ### Use drop_last_mode for ALS
    CPdict = {}
    n_components = loading.get("U0").shape[1]
    for i in np.arange(n_components):
        A = np.array([1])
        if drop_last_mode:
            n_modes_multiplied = len(loading.keys()) - 1
        else:
            n_modes_multiplied = len(loading.keys())  # also equals self.X_dim - 1
        for j in np.arange(n_modes_multiplied):
            loading_factor = loading.get('U' + str(j))  ### I_i X n_components matrix
            # print('loading_factor', loading_factor)
            A = np.multiply.outer(A, loading_factor[:, i])
        A = A[0]
        CPdict.update({'A' + str(i): A})
    return CPdict

def Out_tensor(loading):
    ### given loading, take outer product of respected columns to get CPdict
    CPdict = out(loading, drop_last_mode=False)
    recons = np.zeros(CPdict.get('A0').shape)
    for j in np.arange(len(CPdict.keys())):
        recons += CPdict.get('A' + str(j))

    return recons


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
            nonnegativity=False,
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
                            alpha=regularizer,
                            search_radius_const=search_radius_const,
                            if_compute_recons_error=if_compute_recons_error,
                            save_folder=save_folder,
                            subsample_ratio=subsample_ratio,
                            nonnegativity=nonnegativity,
                            output_results=output_results)

    return result_dict


def MU_run(X,
           n_components=10,
           iter=100,
           regularizer=0,
           rho=0,
           delta=0,
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
                           rho=rho,
                           delta=delta,
                           if_compute_recons_error=if_compute_recons_error,
                           save_folder=save_folder,
                           output_results=output_results)
    return result_dict



def plot_benchmark_errors(full_result_list, save_path):

    time_records = []
    errors = []
    f_interpolated_list = []

    # max duration and time records
    x_all_max = 0
    for i in np.arange(len(full_result_list)):
        errors0 = full_result_list[i].get('timed_errors_trials')
        x_all_max = max(x_all_max, max(errors0[:, :, -1][:, 0]))

    x_all = np.linspace(0, x_all_max, num=101, endpoint=True)

    for i in np.arange(len(full_result_list)):
        errors0 = full_result_list[i].get('timed_errors_trials')
        time_records.append(x_all[x_all < min(errors0[:, :, -1][:, 0])])

    # interpolate data and have common carrier
    for i in np.arange(len(full_result_list)):
        errors0 = full_result_list[i].get('timed_errors_trials')
        f0_interpolated = []
        for j in np.arange(errors0.shape[0]): # trials for same setting
            f0 = interp1d(errors0[j, 0, :], errors0[j, 1, :], fill_value="extrapolate")
            x_all_0 = time_records[i]
            f0_interpolated.append(f0(x_all_0))
        f0_interpolated = np.asarray(f0_interpolated)
        f_interpolated_list.append(f0_interpolated)

    # make figure
    search_radius_const = full_result_list[0].get('search_radius_const')
    color_list = ['g', 'k', 'r', 'c', 'b','m']
    marker_list = ['*', '|', 'x', 'o', '+']
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    for i in np.arange(len(full_result_list)):
        f0_interpolated = f_interpolated_list[i]
        f_avg0 = np.sum(f0_interpolated, axis=0) / f0_interpolated.shape[0]  ### axis-0 : trials
        f_std0 = np.std(f0_interpolated, axis=0)

        x_all_0 = time_records[i]
        color = color_list[i % len(color_list)]
        marker = marker_list[i % len(marker_list)]

        result_dict = full_result_list[i]
        delta = result_dict.get("delta")
        if delta==0 or delta is None:
            label0 = result_dict.get("method")
        else:
            # label0 = result_dict.get("method") + " ($\\beta=${}, $c'=${:.0f})".format(beta, search_radius_const)
            label0 = result_dict.get("method") + " ($\\delta=${})".format(delta)

        markers, caps, bars = axs.errorbar(x_all_0, f_avg0, yerr=0.3*f_std0,
                                           fmt=color+'-', marker=marker, label=label0, errorevery=5)
        axs.fill_between(x_all_0, f_avg0 - 0.3*f_std0, f_avg0 + 0.3*f_std0, facecolor=color, alpha=0.1)
    # min_max duration
    x_all_min_max = []
    for i in np.arange(len(time_records)):
        x_all_ALS0 = time_records[i]
        x_all_min_max.append(max(x_all_ALS0))

    x_all_min_max = min(x_all_min_max)
    axs.set_xlim(0, x_all_min_max)
    axs.legend(ncol =2)

    [bar.set_alpha(0.5) for bar in bars]
    # axs.set_ylim(0, np.maximum(np.max(f_OCPDL_avg + f_OCPDL_std), np.max(f_ALS_avg + f_ALS_std)) * 1.1)
    axs.set_xlabel('Elapsed time (s)', fontsize=15)
    axs.set_ylabel('Relative Recons. Error', fontsize=15)
    axs.set_yscale('log')
    #plt.axhline(y=0.3*1e-2, color='k',linestyle='--',alpha=0.4)
    data_name = full_result_list[0].get('data_name')
    
    title = data_name
    plt.title(label=data_name + " (fixed $\\rho$)",loc="center",fontsize=15)
    #plt.suptitle(title, fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #axs.legend(fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(0.15, 0.1, 0.9, 0.9, 0.00, 0.00)

    plt.savefig(save_path, bbox_inches='tight')



def main(n_components = 2,
        iter = 200,
        num_repeat = 10,
        save_folder = "Output_files/MP_test1",
        data_name = "Synthetic--Sparse",#"Cifar10",#"Synthetic", # or "Twitter"
        nonnegativity=True,
        run_ALS = False,
        run_MU = True,
        plot_errors = True):


   
    # Load data
    loading = {}

    # Load data
    if data_name == "Synthetic--Dense":
        if nonnegativity:
            np.random.seed(400)
            U0 = np.random.rand(100, n_components)
            #M = sp.random(100,n_components,density=.2, format='csr')
            #U0 = M.A
            #np.random.seed(20)
            U1 = np.random.rand(50, n_components)
            #M = sp.random(50,n_components,density=.2, format='csr')
            #U1 = M.A
            #np.random.seed(210)
            #U2 = np.random.rand(30, n_components)
            #print(U0.shape)
            #print(U1.shape)
        else:
            np.random.seed(198)
            U0 = np.random.normal(size=(100, n_components))
            np.random.seed(20)
            U1 = np.random.normal(size=(100, n_components))
            np.random.seed(300)
            U2 = np.random.normal(size=(100, n_components))

        loading.update({'U0': U0})
        loading.update({'U1': U1})
        #loading.update({'U2': U2})

        X = Out_tensor(loading)
    elif data_name=="Synthetic--Sparse":
        if nonnegativity:
            np.random.seed(400)
            #U0 = np.random.rand(100, n_components)
            M = sp.random(100,n_components,density=.2, format='csr')
            U0 = M.A
            #np.random.seed(20)
            #U1 = np.random.rand(50, n_components)
            M = sp.random(50,n_components,density=.2, format='csr')
            U1 = M.A
            #np.random.seed(210)
            #U2 = np.random.rand(30, n_components)
            print(U0.shape)
            print(U1.shape)
        else:
            np.random.seed(198)
            U0 = np.random.normal(size=(100, n_components))
            np.random.seed(20)
            U1 = np.random.normal(size=(100, n_components))
            np.random.seed(300)
            U2 = np.random.normal(size=(100, n_components))

        loading.update({'U0': U0})
        loading.update({'U1': U1})
        #loading.update({'U2': U2})

        X = Out_tensor(loading)
    elif data_name == "Twitter":
        path = "Data/Twitter/top_1000_daily/data_tensor_top1000.pickle"
        dict = pickle.load(open(path, "rb"))
        X = dict[1] * 10000
        file_name = "Twitter"
    elif data_name =="Cifar10":
        X = np.load("data/cifar10_1000x32x32x3.npy")
        X= X[:,:,:,0]
        file_name="Cifar10"

    print('X.shape', X.shape)

    # X = 10 * X/np.linalg.norm(X)

    # search_radius_const = np.linalg.norm(X.reshape(-1,1),1)
    search_radius_const = np.linalg.norm(X.reshape(-1,1),1) / 300000
    print('search_radius_const:', search_radius_const)

    loading_list = []
    for i in np.arange(num_repeat):
        loading_list.append(initialize_loading(data_shape=X.shape, n_components=n_components))

    full_result_list = []
#should for num_repeat first then beta
    if run_ALS:
        # beta_list = [5, None]
        beta=None
        alpha_list = [5, 1, 0.5, 0.1, 0]
        for alpha in alpha_list:
            
            if not alpha==0:
                beta=0.1
            if alpha==0:
                beta=None
            
            print('!!! ALS initialized with beta:', beta)
            list_full_timed_errors = []
            results_dict = {}
            for i in np.arange(num_repeat):

                result_dict_ALS = ALS_run(X,
                                          n_components=n_components,
                                          iter=iter,
                                          regularizer=alpha,
                                          # inverse regularizer on time mode (to promote long-lasting topics),
                                          # no regularizer on on words and tweets
                                          ini_loading=loading_list[i],
                                          beta=beta,
                                          search_radius_const=search_radius_const,
                                          subsample_ratio=None,
                                          if_compute_recons_error=True,
                                          save_folder=save_folder,
                                          nonnegativity=nonnegativity,
                                          output_results=True)
                time_error = result_dict_ALS.get('time_error')
                list_full_timed_errors.append(time_error.copy())
                # print('!!! list_full_timed_errors', len(list_full_timed_errors))

            timed_errors_trials = np.asarray(
                list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)

            if alpha==0:
                results_dict.update({"method": "BCD"})
            else:
                results_dict.update({"method": "BMM"})
            results_dict.update({"data_name": data_name})
            results_dict.update({"beta": beta})
            results_dict.update({"alpha": alpha})
            results_dict.update({"search_radius_const": search_radius_const})
            results_dict.update({"n_components": n_components})
            results_dict.update({"iterations": iter})
            results_dict.update({"num_trials": num_repeat})
            results_dict.update({'timed_errors_trials': timed_errors_trials})
            full_result_list.append(results_dict.copy())

            save_path = save_folder + "/full_result_list_" + str(data_name)
            np.save(save_path, full_result_list)
            # print('full_result_list:', full_result_list)


    if run_MU:
        results_dict = {}
        print('!!! MU initialized')
        list_full_timed_errors = []
        iter1 = iter*1.5
        delta_list =[1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0]
        for delta in delta_list:
            if delta==0:
                rho=0
            else:
                rho=1e-8
            for i in np.arange(num_repeat):
                result_dict_MU = MU_run(X,
                                        n_components=n_components,
                                        iter=iter1,
                                        regularizer=0,
                                        rho=rho,
                                        delta=delta,
                                        ini_loading=loading_list[i],
                                        if_compute_recons_error=True,
                                        save_folder=save_folder,
                                        output_results=True)
                time_error = result_dict_MU.get('time_error')
                list_full_timed_errors.append(time_error.copy())
                # print('!!! list_full_timed_errors', len(list_full_timed_errors))
    
            timed_errors_trials = np.asarray(
                list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)
    
            if rho==0 and delta==0:
                results_dict.update({"method": "MU"})
            else:
                results_dict.update({"method": "MUR"})
            results_dict.update({"data_name": data_name})
            results_dict.update({"n_components": n_components})
            results_dict.update({"search_radius_const": search_radius_const})
            results_dict.update({"iterations": iter1})
            results_dict.update({"rho": rho})
            results_dict.update({"delta": delta})
            results_dict.update({"num_trials": num_repeat})
            results_dict.update({'timed_errors_trials': timed_errors_trials})
            full_result_list.append(results_dict.copy())
    
            save_path = save_folder + "/full_result_list_" + str(data_name)
            np.save(save_path, full_result_list)
        # print('full_result_list:', full_result_list)


    if plot_errors:
       full_result_list = np.load(save_folder + "/full_result_list_" + str(data_name) + ".npy", allow_pickle=True)
       full_result_list = full_result_list[::-1]
       n_trials = full_result_list[0].get("num_trials")
       n_components = full_result_list[0].get('n_components')
       search_radius_const = full_result_list[0].get('search_radius_const')
       data_name = full_result_list[0].get('data_name')
       time_now=datetime.datetime.now().strftime('%H_%M_%S')
       save_path = save_folder + "/full_result_error_plot" + '_ntrials_' + str(n_trials) + "_" + "_ncomps_" + str(
           n_components) + "_src_" + str(search_radius_const) + "_" + str(data_name) + time_now + ".pdf"
   
       plot_benchmark_errors(full_result_list, save_path=save_path)
       print('!!! plot saved')


if __name__ == '__main__':
    main()
