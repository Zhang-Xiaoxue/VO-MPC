# %%
import numpy as np
import math
from scipy.special import erf
from casadi import *
import json

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
        norm_p_ij = norm_1(p_ij)
        sin_alpha = radius / norm_p_ij
        cos_alpha = np.sqrt(norm_p_ij**2 - radius**2) / norm_p_ij
        
        tangent_trans_1 = np.array([[cos_alpha, sin_alpha], [sin_alpha, -cos_alpha]])
        T_ij_1 = tangent_trans_1@p_ij
        N_ij_1 = np.array([[0,1],[1,0]])@(T_ij_1)
        kappa_1 = np.sqrt(2*N_ij_1.T@W_v@N_ij_1)*erf(1-2*delta_1)

        tangent_trans_2 = np.array([[cos_alpha, -sin_alpha], [sin_alpha, cos_alpha]])    
        T_ij_2 = tangent_trans_2@(p_ij)
        N_ij_2 = np.array([[0,-1],[1,0]])@(T_ij_2)
        kappa_2 = np.sqrt(2*N_ij_2.T@W_v@N_ij_2)*erf(1-2*delta_2)
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

        states = SX.sym('ss', nx, 1)
        controls = SX.sym('u', nu, 1)
        f = Function('f', [states, controls], [globals().get('quad_'+str(idx_host)).A @states + \
            globals().get('quad_'+str(idx_host)).B@controls + np.random.multivariate_normal([0,0,0,0], np.identity(nx))])

        X = SX.sym('X', nx, (N+1))
        U = SX.sym('U', nu, N)
        P = SX.sym('P', (nx+N*(nx+nu)))

        """ define obj and constr for MPC """
        obj = 0
        constr = []
        st = X[:,0]
        constr = vertcat(constr, st-P[0:nx])
        for k in range(N):
            st = X[:,k]
            con = U[:,k]
            P_st_ref = P[((nx+nu)*(k+1)-nu) : ((nx+nu)*(k+1)+nx-nu)]
            # P_u_ref = P[((nx+nu)*(k+1)+nx-nu):((nx+nu)*(k+1)+nx)]
            # obj += (st-P_st_ref).T @ Q @ (st-P_st_ref) + (con-P_u_ref).T @ R @ (con-P_u_ref)
            obj += (st-P_st_ref).T @ Q @ (st-P_st_ref) + (con).T @ R @ (con)
            st_next = X[:,k+1]
            f_value = f(st, con)
            noise = np.random.multivariate_normal([0,0,0,0], np.identity(nx))
            st_next_value = globals().get('quad_'+str(idx_host)).A @ st + globals().get('quad_'+str(idx_host)).B@con + noise
            constr = vertcat(constr, st_next-st_next_value)
            
            for j in range(sysParam['num_robot']-1):
                p_i = st[0:ndim]
                p_j = p_others[:,j]
                v_j = v_others[:,j]
                N_ij_1, kappa_1, N_ij_2, kappa_2 = self.comp_vectors(p_i, p_j)
                v_i = st[ndim:nx]
                constr = vertcat(constr, kappa_1 + dot(N_ij_1,v_j) - dot(N_ij_1,v_i) )
                constr = vertcat(constr, kappa_2 + dot(N_ij_2,v_j) - dot(N_ij_2,v_i) )
        
        """ formulate NLP problem and solver """
        OPT_variables = vertcat( X.reshape((nx*(N+1),1)), U.reshape((nu*N,1)) )  # (412,1)   
        nlp_prob = dict(f=obj, x=OPT_variables, g=constr, p=P)
        opts = {'ipopt.max_iter':1000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, \
                'ipopt.acceptable_obj_change_tol':1e-6}
        solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

        """ set arguments, i.e. constraints (inequality Bounds and enquality constraints) """
        args = {}

        g_lim_low = np.zeros((1,nx*(N+1)+(num_robot-1)*2*N))
        for k in range(N):
            g_lim_low[:, (2*nx+k*(nx+(num_robot-1)*2)) : (2*nx+k*nx+(k+1)*(num_robot-1)*2)] = -inf
        g_lim_upp = np.zeros((1,nx*(N+1)+(num_robot-1)*2*N))
        args['lbg'] = g_lim_low
        args['ubg'] = g_lim_upp
                
        x_lim_low = DM((nx*(N+1)+nu*N),1)
        # begin to set bounds for states
        x_lim_low[0:(nx*(N+1)):nx,0] = -10
        x_lim_low[1:(nx*(N+1)):nx,0] = -10
        x_lim_low[2:(nx*(N+1)):nx,0] = -100
        x_lim_low[3:(nx*(N+1)):nx,0] = -100

        # begin to set bounds for control inputs
        x_lim_low[(nx*(N+1)) :, 0] = -1000

        x_lim_upp = DM((nx*(N+1)+nu*N),1)
        # begin to set bounds for states
        x_lim_upp[0:(nx*(N+1)):nx,0] = 10
        x_lim_upp[1:(nx*(N+1)):nx,0] = 10
        x_lim_upp[2:(nx*(N+1)):nx,0] = 100
        x_lim_upp[3:(nx*(N+1)):nx,0] = 100

        # begin to set bounds for control inputs
        x_lim_upp[(nx*(N+1)) :, 0] = 1000

        args['lbx'] = x_lim_low
        args['ubx'] = x_lim_upp


        """ begin to compute (mpc problem) """
        x0 = globals().get('quad_'+ str(idx_host)).x # initial condition  (nx, )
        xs = xr[:,0] # reference position  (4, )  xr:(4,26)
        u0 = np.zeros((N,nu)) # (25, 4)
        X0 = repmat(x0, 1, (N+1)).T # initialize the states decision variables (26,4)

        args['p'] = DM(nx+N*(nx+nu),1) # (154, 1)
        args['p'][0:nx] = x0

        for k in range(N):
            args['p'][((nx+nu)*(k+1)-nu) : ((nx+nu)*(k+1)+nx-nu)] = xr[:,k]
        
        # initialize optimization decision variables
        args['x0'] = vertcat(X0.reshape(((nx*(N+1)), 1)), u0.reshape(((nu*N), 1)))  #(154, 1)

        sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=args['p'])
        
        solx = sol['x']
        solx_u = np.array(solx)[(nx*(N+1)):(solx.shape[0]+1)]
        u = solx_u.reshape((N,nu))
        control_input = u[0,:]
                
        X0 = solx[0:nx*(N+1)].reshape(((N+1), nx))
        # shift trajectory to initialize the next step
        X0 = vertcat(X0[1:,:], X0[-1,:])
        state_pred = np.array(X0)

        return control_input, state_pred

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
