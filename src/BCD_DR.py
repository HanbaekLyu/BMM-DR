# (setq python-shell-interpreter "./venv/bin/python")

import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from sklearn.decomposition import SparseCoder
import time
from tqdm import trange

DEBUG = False


class ALS_DR():

    def __init__(self,
                 X, n_components=100,
                 iterations=500,
                 sub_iterations=10,
                 batch_size=20,
                 ini_CPdict=None,
                 ini_loading=None,
                 history=0,
                 ini_A=None,
                 ini_B=None,
                 alpha=None,
                 beta=None,
                 subsample=True):
        '''
        Alternating Least Squares with Diminishing Radius for Nonnegative CP Tensor Factorization
        X: data tensor (n-dimensional) with shape I_1 * I_2 * ... * I_n
        Seeks to find n loading matrices U0, ... , Un, each Ui is (I_i x r)
        n_components (int) = r = number of rank-1 CP factors
        iter (int): number of iterations where each iteration is a call to step(...)
        '''
        self.X = X
        self.n_modes = X.ndim - 1  ### -1 for discounting the last "batch_size" mode
        self.n_components = n_components
        self.batch_size = batch_size
        self.iterations = iterations

        if ini_CPdict is not None:
            self.CPdict = ini_CPdict
        else:
            self.CPdict = self.initialize_CPdict()

        if ini_loading is None:
            self.loading = self.initialize_loading()
        else:
            self.loading = ini_loading

        self.alpha = alpha

    def initialize_loading(self):
        ### loading = python dict of [U1, U2, \cdots, Un], each Ui is I_i x R array
        loading = {}
        for i in np.arange(self.n_modes):  # n_modes = X.ndim -1 where -1 for the last `batch mode'
            loading.update({'U' + str(i): np.random.rand(self.X.shape[i], self.n_components)})
        return loading

    def initialize_CPdict(self):
        ### CPdict = python dict of [A1, A2, \cdots, AR], R=n_components, each Ai is a rank-1 tensor
        CPdict = {}
        for i in np.arange(self.n_components):
            CPdict.update({'A' + str(i): np.zeros(shape=self.X.shape[:-1])})
            ### Exclude last "batch size" mode
        return CPdict

    def out(self, loading, drop_last_mode=False):
        ### given loading, take outer product of respected columns to get CPdict
        ### Use drop_last_mode for ALS
        CPdict = {}
        for i in np.arange(self.n_components):
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

    def get_A_U(self, A, loading, j):
        ### Compute (n_componetns) x (n_components) intermediate aggregation matrix
        A_U = A
        for k in np.arange(self.n_modes):
            if k != j:
                U = loading.get('U' + str(k))  ### I_{k} x n_components loading matrix
                U = U.T @ U  ### (n_componetns) x (n_components) matrix
                A_U = A_U * U
        return A_U

    def get_B_U(self, B, loading, j):
        ### Compute (n_components) x (I_j) intermediate aggregation matrix
        ### B has size (I_1 * I_2 * ... * I_n) x (n_componetns)

        B_U = np.zeros(shape=(self.n_components, self.X.shape[j]))
        for r in np.arange(self.n_components):
            B_r = B[:, r].reshape(self.X.shape[:-1])
            for i in np.arange(self.n_modes):
                if i != j:
                    U = loading.get('U' + str(i))  ### I_{k} x n_components loading matrix
                    B_r = tl.tenalg.mode_dot(B_r, U[:, r], i)
                    B_r = np.expand_dims(B_r, axis=i)  ## mode_i product above kills axis i
            B_U[r, :] = B_r.reshape(self.X.shape[j])
        return B_U

    def update_code_within_radius(self, X, W, H0, r, alpha=0, sub_iter=2, stopping_diff=0.1):
        '''
        Find \hat{H} = argmin_H ( | X - WH| + alpha|H| ) within radius r from H0
        Use row-wise projected gradient descent
        Do NOT sparsecode the whole thing and then project -- instable
        12/5/2020 Lyu
        Using sklearn's SparseCoder for ALS seems unstable (errors could increase due to initial overshooting and projection)
        '''

        A = W.T @ W
        B = W.T @ X
        H1 = H0.copy()
        i = 0
        dist = 1
        while (i<sub_iter) and (dist>stopping_diff):
            H1_old = H1.copy()
            for k in np.arange(H0.shape[0]):
                grad = (np.dot(A[k,:], H1) - B[k,:]+alpha*np.ones(H0.shape[1]))
                # H1[k, :] = H1[k,:] - (1 / (A[k, k] + np.linalg.norm(grad, 2))) * grad
                H1[k, :] = H1[k,:] - (1 / ( ((i+5)**(0.5))* (A[k, k] + 1))) * grad
                # use i+10 to ensure monotonicity (but gets slower)
                H1[k,:] = np.maximum(H1[k,:], np.zeros(shape=(H1.shape[1],)))  # nonnegativity constraint
                if r is not None:  # usual sparse coding without radius restriction
                    d = np.linalg.norm(H1 - H0, 2)
                    H1 = H0 + (r/max(r, d))*(H1 - H0)
                H0 = H1

            dist = np.linalg.norm(H1 - H1_old, 2)/np.linalg.norm(H1_old, 2)
            # print('!!! dist', dist)
            H1_old = H1
            i = i+1
            # print('!!!! i', i)  # mostly the loop finishes at i=1 except the first round

        return H1

    def inner_product_loading_code(self, loading, code):
        ### loading = [U1, .. , Un], Ui has shape Ii x R
        ### code = shape R x batch_size
        ### compute the reconstructed tensor of size I1 x I2 x .. x In x batch_size
        ### from the given loading = [U1, .. , Un] and code = [c1, .., c_batch_size], ci = (R x 1)
        recons = np.zeros(shape=self.X.shape[:-1])
        recons = np.expand_dims(recons, axis=-1)  ### Now shape I1 x I2 x .. x In x 1
        CPdict = self.out(loading)
        for i in np.arange(code.shape[1]):
            A = np.zeros(self.X.shape[:-1])
            for j in np.arange(len(loading.keys())):
                A = A + CPdict.get('A' + str(j)) * code[j, i]
            A = np.expand_dims(A, axis=-1)
            recons = np.concatenate((recons, A), axis=-1)
        recons = np.delete(recons, obj=0, axis=-1)
        # print('recons.shape', recons.shape)
        return recons

    def compute_reconstruction_error(self, X, loading, is_X_full_tensor=False):
        ### X data tensor, loading = loading matrices
        ### Find sparse code and compute reconstruction error
        ### If X is full tensor,
        c = self.sparse_code_tensor(X, self.out(loading))
        # print('X.shape', X.shape)
        # print('c.shape', c.shape)
        recons = self.inner_product_loading_code(loading, c.T)
        error = np.linalg.norm((X - recons).reshape(-1, 1), ord=2)
        return error

    def ALS(self,
            iter=100,
            ini_loading=None,
            beta = None,
            if_compute_recons_error=False,
            save_folder='Output_files',
            search_radius_const = 1000,
            output_results=False):
        '''
        Given data tensor X and initial loading matrices W_ini, find loading matrices W by Alternating Least Squares
        with Diminishing Radius
        '''

        X = self.X


        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', ini_loading.shape, '\n')

        n_modes = len(X.shape)
        if ini_loading is not None:
            loading = ini_loading
        else:
            loading = {}
            for i in np.arange(X.ndim):
                loading.update({'U' + str(i): np.random.rand(X.shape[i], self.n_components)})

        result_dict = {}
        time_error = np.zeros(shape=[0, 2])
        elapsed_time = 0

        for step in trange(int(iter)):
            start = time.time()

            for mode in np.arange(n_modes):
                X_new = np.swapaxes(X, mode, -1)

                U = loading.get('U' + str(mode))  # loading matrix to be updated
                loading_new = loading.copy()
                loading_new.update({'U' + str(mode): loading.get('U' + str(n_modes - 1))})
                loading_new.update({'U' + str(n_modes - 1): U})

                X_new_mat = X_new.reshape(-1, X_new.shape[-1])
                CPdict = self.out(loading_new, drop_last_mode=True)

                W = np.zeros(shape=(X_new_mat.shape[0], self.n_components))
                for j in np.arange(self.n_components):
                    W[:, j] = CPdict.get('A' + str(j)).reshape(-1, 1)[:, 0]

                if beta is None:  # usual nonnegative sparse coding
                    Code = self.update_code_within_radius(X_new_mat, W, U.T, r=None, alpha=0)
                else:
                    if search_radius_const is None:
                            search_radius_const = 10000

                    search_radius = search_radius_const * (float(step+1))**(-beta)/np.log(float(step+2))

                    # sparse code within radius
                    Code = self.update_code_within_radius(X_new_mat, W, U.T, search_radius, alpha=0)

                    # print('!!!!! search_radius_const', search_radius_const)
                U_new = Code.T.reshape(U.shape)

                loading.update({'U' + str(mode): U_new})

                #print('!!! Iteration %i: %i th loading matrix updated..' % (step, mode))

            end = time.time()
            elapsed_time += end - start

            if if_compute_recons_error:
                CPdict = self.out(loading, drop_last_mode=False)
                recons = np.zeros(X.shape)
                for j in np.arange(len(loading.keys())):
                    recons += CPdict.get('A' + str(j))
                error = np.linalg.norm((X - recons).reshape(-1, 1), ord=2)
                time_error = np.append(time_error, np.array([[elapsed_time, error]]), axis=0)
                print('!!! Reconstruction error at iteration %i = %f.3' % (step, error))

        result_dict.update({'loading': loading})
        # result_dict.update({'CPdict': self.out(loading)}) ### this is very expensive
        result_dict.update({'time_error': time_error.T})
        result_dict.update({'iter': iter})
        result_dict.update({'n_components': self.n_components})
        np.save(save_folder + "/ALS_result_", result_dict)

        if output_results:
            return result_dict
        else:
            return loading

    def MU(self,
            iter=100,
            ini_loading=None,
            if_compute_recons_error=False,
            save_folder='Output_files',
            output_results=False):
        '''
        Given data tensor X and initial loading matrices W_ini, find loading matrices W by Multiplicative Update
        Ref: Shashua, Hazan, "Non-Negative Tensor Factorization with Applications to Statistics and Computer Vision" (2005)
        '''

        X = self.X

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', ini_loading.shape, '\n')

        n_modes = len(X.shape)
        if ini_loading is not None:
            loading = ini_loading
        else:
            loading = {}
            for i in np.arange(X.ndim):
                loading.update({'U' + str(i): np.random.rand(X.shape[i], self.n_components)})

        result_dict = {}
        time_error = np.zeros(shape=[0, 2])
        elapsed_time = 0

        for step in trange(int(iter)):
            start = time.time()

            for mode in np.arange(n_modes):
                # print('!!! X.shape', X.shape)
                X_new = np.swapaxes(X, mode, -1)
                U = loading.get('U' + str(mode))  # loading matrix to be updated
                loading_new = loading.copy()
                loading_new.update({'U' + str(mode): loading.get('U' + str(n_modes - 1))})
                loading_new.update({'U' + str(n_modes - 1): U})
                # Now update the last loading matrix U = 'U' + str(n_modes - 1)) by MU
                # Matrize X along the last mode to get a NMF problem V \approx W*H, and use MU in LEE & SEUNG (1999)

                # Form dictionary matrix
                CPdict = self.out(loading_new, drop_last_mode=True)
                # print('!!! X_new.shape', X_new.shape)
                W = np.zeros(shape=(len(X_new.reshape(-1, X_new.shape[-1])), self.n_components))
                for j in np.arange(self.n_components):
                    W[:, j] = CPdict.get('A' + str(j)).reshape(-1, 1)[:, 0]

                V = X_new.reshape(-1, X_new.shape[-1])
                # print('!!! W.shape', W.shape)
                # print('!!! U.shape', U.shape)
                # print('!!! V.shape', V.shape)
                U_new = U.T * (W.T @ V)/(W.T @ W @ U.T)
                loading.update({'U' + str(mode): U_new.T})

                # print('!!! Iteration %i: %i th loading matrix updated..' % (step, mode))

            end = time.time()
            elapsed_time += end - start

            if if_compute_recons_error:
                CPdict = self.out(loading, drop_last_mode=False)
                recons = np.zeros(X.shape)
                for j in np.arange(len(loading.keys())):
                    recons += CPdict.get('A' + str(j))
                error = np.linalg.norm((X - recons).reshape(-1, 1), ord=2)
                time_error = np.append(time_error, np.array([[elapsed_time, error]]), axis=0)
                # print('!!! Reconstruction error at iteration %i = %f.3' % (step, error))

        result_dict.update({'loading': loading})
        # result_dict.update({'CPdict': self.out(loading)}) ### this is very expensive
        result_dict.update({'time_error': time_error.T})
        result_dict.update({'iter': iter})
        result_dict.update({'n_components': self.n_components})
        np.save(save_folder + "/MU_result_", result_dict)

        if output_results:
            return result_dict
        else:
            return loading