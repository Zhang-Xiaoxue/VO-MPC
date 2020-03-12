# %%
import numpy as np
import math
from scipy.special import erf
# from casadi import *
import json
import cvxpy

disable_internal_warnings = True

# %%
global sysParam
sysParam = {"deltaT":0.02, "N_pred": 25, "runtime":10, "radius": 0.1, "num_robot": 6, \
    "nx": 4, "nu": 2, "ndim":2, "mass": 1}

#%%
class Quadcopter():
    def __init__(self, x_init):
        self.A = np.array([[1,0,sysParam['deltaT'],0],[0,1,0,sysParam['deltaT']],[0,0,1,0],[0,0,0,1]]) 
        self.B = np.array([[0,0],[0,0],[sysParam['deltaT']/sysParam['mass'],0],[0,sysParam['deltaT']/sysParam['mass']]])
        self.W = 0.1*np.identity(sysParam['nx'])  # noise covariance

        # Definite intial conditions
        if any(x_init) == None:
            self.x = np.zeros((sysParam['nx'],1))
        else:
            self.x = x_init
    
    def model(self, u):
        noise = np.random.multivariate_normal([0,0,0,0], self.W).reshape(-1,1)
        previous_x = self.x
        self.x = np.squeeze(self.A@previous_x.reshape(-1,1) + self.B@u.reshape(-1,1) + noise)

# if __name__ == '__main__':
#     x_init = np.zeros((sysParam['nx'],1))
#     quad = Quadcopter(x_init=x_init)
#     quad.model(np.ones((sysParam['nu'],1)))

# %%
class Controller():
    def __init__(self):
        pass

    def comp_vectors (self, p_i, p_j):
        """ compute the two tangent vectors and norm vectors """
        W_v = 0.1*np.identity(int(sysParam['nx']/2))  # noise covariance
        delta_1, delta_2 = 0.1, 0.01
        radius = sysParam['radius'] + sysParam['radius']

        p_ij = p_j - p_i
        norm_p_ij = cvxpy.atoms.norm(p_ij)
        sin_alpha = radius / norm_p_ij
        cos_alpha = cvxpy.sqrt(norm_p_ij**2 - radius**2) / norm_p_ij
        
        tangent_trans_1 = np.array([[cos_alpha, sin_alpha], [sin_alpha, -cos_alpha]])
        T_ij_1 = tangent_trans_1@p_ij
        N_ij_1 = np.array([[0,1],[1,0]])@(T_ij_1)
        kappa_1 = np.power(2*N_ij_1.T@W_v@N_ij_1, 0.5)*erf(1-2*delta_1)

        tangent_trans_2 = np.array([[cos_alpha, -sin_alpha], [sin_alpha, cos_alpha]])    
        T_ij_2 = tangent_trans_2@(p_ij)
        N_ij_2 = np.array([[0,-1],[1,0]])@(T_ij_2)
        kappa_2 = np.power(2*N_ij_2.T@W_v@N_ij_2, 0.5)*erf(1-2*delta_2)
        return N_ij_1, kappa_1, N_ij_2, kappa_2

    def control(self, p_i, p_others, v_others, idx_host, xr):
        deltaT = sysParam['deltaT']
        N = sysParam['N_pred']
        nx = sysParam['nx']
        nu = sysParam['nu']
        num_robot = sysParam['num_robot']
        Q = np.zeros([nx,nx])
        Q[0,0] = 1
        Q[1,1] = 1
        R = np.zeros([nu,nu])
        ndim = 2 # dimension = 2
        
        # Define optimzation variables
        x = cvxpy.Variable((nx, N + 1)) # size of variable 4 x (T+1)
        u = cvxpy.Variable((nu, N)) # size of variable 4 x T

        cost = 0.0
        constr = []

        velo_upper_bound = 5
        velo_lower_bound = -5

        # Loop through horizon
        for t in range(N):
            cost += cvxpy.quad_form(x[:, t + 1] - xr[:, t + 1 ], Q) # This does xT * Q * x
            cost += cvxpy.quad_form(u[:, t], R) # This does uT * R * u

            constr += [ x[:, t + 1] == globals().get('quad_'+str(idx_host)).A @x[:, t] + \
                globals().get('quad_'+str(idx_host)).B@u[:, t] + np.random.multivariate_normal([0,0,0,0], 0.1*np.identity(nx)) ]# Contraint to follow dynamics
            # constr += [x[:, t + 1] == A * x[:, t] + B * (u[:, t] + u_eq)] # Contraint to follow dynamics

            # constrain position
            constr += [x[0, t + 1] <= 100]
            constr += [x[0, t + 1] >= -100]
            constr += [x[1, t + 1] <= 100]
            constr += [x[1, t + 1] >= -100]
            # Contrain velocity
            constr += [x[2, t + 1] <= 100]
            constr += [x[2, t + 1] >= -100]
            constr += [x[3, t + 1] <= 100]
            constr += [x[3, t + 1] >= -100]

            # Constrain u force 
            constr += [u[:, t] <= 1000]
            constr += [u[:, t] >= -1000]

            # Constrain velocity obstacles
            for j in range(sysParam['num_robot']-1):
                p_i = x[0:ndim, t+1]
                p_j = p_others[:,j]
                v_j = v_others[:,j]
                N_ij_1, kappa_1, N_ij_2, kappa_2 = self.comp_vectors(p_i, p_j)
                v_i = x[ndim:nx, t+1]
                constr += [ kappa_1 <= np.dot(N_ij_1,v_j) - np.dot(N_ij_1,v_i)  ]
                constr += [ kappa_2 <= np.dot(N_ij_2,v_j) - np.dot(N_ij_2,v_i)  ]

        constr += [x[:, 0] == globals().get('quad_'+ str(idx_host)).x[:,0]] # Contraint for initial conditions

        # Create cvxpy optimization problem
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)

        prob.solve(verbose=False) # Solve optimization

        # If optimal solution reached
        if prob.status == cvxpy.OPTIMAL:
            ox = self.arrayflatten(x.value[0, :]) # result for x for all timesteps
            dx = self.arrayflatten(x.value[1, :])
            theta = self.arrayflatten(x.value[2, :])
            dtheta = self.arrayflatten(x.value[3, :])
            # if traj_choice == 5:
            #     dist_obs = (d.value[:,:]).T
            # ou = self.arrayflatten(u.value[:, :])
            ou = u.value
        else:
            print ("ERROR: Optimization not OPTIMAL")
            print ("Status", prob.status)
            print ("Try other solvers: OSQP")
            prob.solve(verbose=True, solver=cvxpy.OSQP) # Solve optimization , 'OSQP'
            if prob.status == cvxpy.OPTIMAL:
                ox = self.arrayflatten(x.value[0, :]) # result for x for all timesteps
                dx = self.arrayflatten(x.value[1, :])
                theta = self.arrayflatten(x.value[2, :])
                dtheta = self.arrayflatten(x.value[3, :])
                # if traj_choice == 5:
                #     dist_obs = (d.value[:,:]).T
                ou = u.value
            else:
                print ("ERROR: Optimization not OPTIMAL")
                print ("Status", prob.status)
                print ("Try other solvers: GORUBI")
                prob.solve(verbose=True, solver=cvxpy.GUROBI) # Solve optimization , 'OSQP'
                if prob.status == cvxpy.OPTIMAL:
                    ox = self.arrayflatten(x.value[0, :]) # result for x for all timesteps
                    dx = self.arrayflatten(x.value[1, :])
                    theta = self.arrayflatten(x.value[2, :])
                    dtheta = self.arrayflatten(x.value[3, :])
                    # if traj_choice == 5:
                    #     dist_obs = (d.value[:,:]).T
                    ou = u.value

                else:
                    print("ERROR: Optimization not OPTIMAL")
                    print ("Status", prob.status)
                    print("Try other solvers: ECOS")
                    prob.solve(verbose=True, solver=cvxpy.ECOS) # Solve optimization 
                    if prob.status == cvxpy.OPTIMAL:
                        ox = self.arrayflatten(x.value[0, :]) # result for x for all timesteps
                        dx = self.arrayflatten(x.value[1, :])
                        theta = self.arrayflatten(x.value[2, :])
                        dtheta = self.arrayflatten(x.value[3, :])
                        # if traj_choice == 5:
                        #     dist_obs = (d.value[:,:]).T
                        ou = u.value
                    else:
                        print ("ERROR: Optimization not OPTIMAL")
                        print ("Status", prob.status)
                        print ("Try other solvers: MOSEK")
                        prob.solve(verbose=True, solver=cvxpy.MOSEK) # Solve optimization , 'OSQP'
                        if prob.status == cvxpy.OPTIMAL:
                            ox = self.arrayflatten(x.value[0, :]) # result for x for all timesteps
                            dx = self.arrayflatten(x.value[1, :])
                            theta = self.arrayflatten(x.value[2, :])
                            dtheta = self.arrayflatten(x.value[3, :])
                            # if traj_choice == 5:
                            #     dist_obs = (d.value[:,:]).T
                            ou = u.value
                        else:
                            print("ERROR: Optimization not OPTIMAL")
                            print ("Status", prob.status)
                            print("Try other solvers: SCS")
                            prob.solve(verbose=True, solver=cvxpy.SCS) # Solve optimization , 'OSQP'
                            if prob.status == cvxpy.OPTIMAL:
                                ox = self.arrayflatten(x.value[0, :]) # result for x for all timesteps
                                dx = self.arrayflatten(x.value[1, :])
                                theta = self.arrayflatten(x.value[2, :])
                                dtheta = self.arrayflatten(x.value[3, :])
                                # if traj_choice == 5:
                                #     dist_obs = (d.value[:,:]).T
                                ou = u.value
                            else:
                                print ("ERROR: Optimization not OPTIMAL")
                                print ("Status", prob.status)
                                print ("Try many times, But still Not work!")
                                # if traj_choice == 5:
                                #     return 0, 0, 0, 0, 0, 0
                                # else:
                                return 0, 0
        return ox, ou

    def arrayflatten(self, x):
        return np.array(x).flatten()

# %%
""" Compute reference states """
def reference(X_init_all, X_end_all, idx_host):
    locals()['_xr_x_'+str(idx_host)] = np.linspace(X_init_all[0,idx_host], \
        X_end_all[0,idx_host], int(sysParam['runtime']/sysParam['deltaT']))
    locals()['_xr_y_'+str(idx_host)] = np.linspace(X_init_all[1,idx_host], \
        X_end_all[1,idx_host], int(sysParam['runtime']/sysParam['deltaT']))
    locals()['_xr_v_'+str(idx_host)] = np.zeros([int(sysParam['nx']-sysParam['ndim']), int(sysParam['runtime']/sysParam['deltaT'])])
    
    locals()['xr_'+str(idx_host)] = np.vstack([locals().get('_xr_x_'+str(idx_host)), locals().get('_xr_y_'+str(idx_host)), locals().get('_xr_v_'+str(idx_host))])
    return locals()['xr_'+str(idx_host)]

# X_init_all = np.zeros((sysParam['nx'],sysParam['num_robot'])) 
# if sysParam['num_robot'] == 6:
#     X_init_all[0,:] = [-4, -2,  2,  4,  -2,   2]
#     X_init_all[1,:] = [ 0,  2,  2,  0,  -2,  -2]
# res = reference(X_init_all,idx_host=1)

# %%
if __name__ == '__main__':

    X_init_all = np.zeros((sysParam['nx'],sysParam['num_robot'])) 
    X_end_all = np.zeros((sysParam['nx'],sysParam['num_robot'])) 
    if sysParam['num_robot'] == 6:
        X_init_all[0,:] = [-4, -2,  2,  4,  -2,   2]
        X_init_all[1,:] = [ 0,  2,  2,  0,  -2,  -2]
        X_end_all[0,:]  = [ 4, -2,  2, -4,  -2,   2]
        X_end_all[1,:]  = [ 0, -2, -2,  0,   2,   2]
    
    # Initiallize
    for idx_host in range(sysParam['num_robot']):
        globals()['xr_'+str(idx_host)] = reference(X_init_all, X_end_all, idx_host)
        locals()['x_init_'+str(idx_host)] = X_init_all[:,idx_host]
        globals()['quad_'+str(idx_host)] = Quadcopter(locals()['x_init_'+str(idx_host)])
        globals()['controller_'+str(idx_host)] = Controller()
    
        locals()['u_result_'+str(idx_host)] = []
        locals()['x_result_'+str(idx_host)] = []
    
    for iter in range(int(sysParam['runtime']/sysParam['deltaT']-sysParam['N_pred'])):
        print("************************** " + "Iteration Times: ", str(iter) + " ***************************")
        
        for idx_host in range(sysParam['num_robot']):

            p_i = globals()['quad_'+str(idx_host)].x[0:sysParam['ndim']]
            j = 0
            p_others = np.zeros([sysParam['ndim'], (sysParam['num_robot']-1)])
            v_others = np.zeros([sysParam['ndim'], (sysParam['num_robot']-1)])
            for idx_j in [m for m in range(sysParam['num_robot']) if m != idx_host]:
                p_others[:,j] = globals()['quad_'+str(idx_j)].x[0:sysParam['ndim']]
                v_others[:,j] = globals()['quad_'+str(idx_j)].x[sysParam['ndim']:sysParam['nx']]
                j += 1
            
            u, state_pred = globals().get('controller_'+str(idx_host)).control(p_i, p_others, v_others, idx_host, locals().get('xr_'+str(idx_host))[:,iter:int(iter+(sysParam['N_pred']+1))] )
            globals().get('quad_'+str(idx_host)).model(u)

            print ('-------------------')
            print ("iteration #", iter)
            print ("time: ", iter*sysParam['deltaT'])
            print ('number of robot: ', idx_host)
            print ('control input: ', u)
            print ('states: ', globals().get('quad_'+str(idx_host)).x.reshape(-1,1))
            print ("reference loc: ", locals().get('xr_'+str(idx_host))[:,iter])
            
            locals()['u_result_'+str(idx_host)].append(u.tolist())
            locals()['x_result_'+str(idx_host)].append(globals().get('quad_'+str(idx_host)).x.tolist())


    
    """ save data """
    results = {}   
    for idx_host in range(sysParam['num_robot']):
        results['x_results_'+str(idx_host)] = locals().get('x_result_'+str(idx_host))
        results['u_results_'+str(idx_host)] = locals().get('u_result_'+str(idx_host))
        results['x_reference_'+str(idx_host)] = globals().get('xr_'+str(idx_host)).tolist()

    with open('results_data.json', 'w') as json_file:  
        json.dump(results, json_file)

# %%
