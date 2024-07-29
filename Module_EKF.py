"""Import Python toolbox"""
import numpy as np
from numpy.linalg import inv
import time
import scipy
from scipy.integrate import solve_ivp
from Module_CRTBP import CRTBP_dynamics, CRTBP_STM_dynamics, CRTBP_STT_dynamics, CRTBP_DSTT_dynamics
from Module_STT import STT_Pred_mu_P, DSTT_Pred_mu_P
from rich.console import Console
from rich.table import Column, Table

def EKF_Measurement_y(xr, xe, STD):
    """Measurement function"""
    """True measurements"""
    y = xr[1] + np.random.randn(1) * STD[0]
    """Estimated measurements"""
    h = xe[1]
    dy = y - h
    H = np.matrix([
        [0, 1, 0, 0, 0, 0]
    ])
    R = np.matrix(STD[0] ** 2)
    return dy, H, R

def EKF_Process(x0, errR, errV, nav_time, STD, UnitL, UnitV, mu):
    """Module of the EKF"""
    """Parameter setting"""
    DIM = 6
    RelTol = 10**-12
    AbsTol = 10**-12
    """Initial state"""
    r0 = x0[:3]
    v0 = x0[3:]
    """Parameters for orbit determination"""
    # r0e = r0 + errR * np.random.randn(3)
    # v0e = v0 + errV * np.random.randn(3)
    r0e = r0 + errR * np.ones(3)
    v0e = v0 + errV * np.ones(3)
    x0e = np.array([r0e[0], r0e[1], r0e[2], v0e[0], v0e[1], v0e[2]])
    error = np.array([errR, errR, errR, errV, errV, errV])
    P = np.diag(error ** 2)
    Q = np.zeros([DIM, DIM])
    """Begin navigation"""
    number = len(nav_time)
    true_orbit = np.zeros([DIM, number])
    estimated_orbit = np.zeros([DIM, number])
    true_orbit[:, 0] = x0
    estimated_orbit[:, 0] = x0e
    time_cost = np.zeros([number - 1])
    for k in range(number - 1):
        print("==== iter %d ====" % k)
        """state update"""
        xr = true_orbit[:, k]
        xe = estimated_orbit[:, k]
        t0 = nav_time[k]
        tf = nav_time[k + 1]
        """Propagate the true orbit"""
        t_eval = [t0, tf]
        check = solve_ivp(CRTBP_dynamics, [t0, tf], xr, args=(mu, ), method='RK45',
                          t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xr = check.y.T[-1, :]
        true_orbit[:, k + 1] = xr.T
        """Calculate the STM"""
        start = time.time()
        X = np.concatenate((xe, np.eye(6).reshape(36)))
        check = solve_ivp(CRTBP_STM_dynamics, [t0, tf], X, args=(mu, ), method='RK45',
                          t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xe = check.y.T[-1, :6].T
        STM = check.y.T[-1, 6:].reshape(6, 6)
        """Propagate the prior covariance"""
        P = np.matmul(np.matmul(STM, P), STM.T) + Q
        """Measurement update"""
        dy, H, R = EKF_Measurement_y(xr, xe, STD)
        Pzz = np.matmul(np.matmul(H, P), H.T) + R
        Pxz = np.matmul(P, H.T)
        K_k = np.matmul(Pxz, np.linalg.inv(Pzz))
        """Update estimations"""
        estimated_orbit[:, k + 1] = xe + np.matmul(K_k, dy)
        P = P - np.matmul(np.matmul(K_k, Pzz), K_k.T)
        time_cost[k] = time.time() - start
    error = estimated_orbit - true_orbit
    error[:3, :] = error[:3, :] * UnitL
    error[3:, :] = error[3:, :] * UnitV * 1e6
    return error, time_cost

def SEKF_Process(x0, errR, errV, nav_time, STD, UnitL, UnitV, mu):
    """Module of the SEKF"""
    """Parameter setting"""
    DIM = 6
    RelTol = 10**-12
    AbsTol = 10**-12
    """Initial state"""
    r0 = x0[:3]
    v0 = x0[3:]
    """Parameters for orbit determination"""
    # r0e = r0 + errR * np.random.randn(3)
    # v0e = v0 + errV * np.random.randn(3)
    r0e = r0 + errR * np.ones(3)
    v0e = v0 + errV * np.ones(3)
    x0e = np.array([r0e[0], r0e[1], r0e[2], v0e[0], v0e[1], v0e[2]])
    error = np.array([errR, errR, errR, errV, errV, errV])
    P = np.diag(error ** 2)
    Q = np.zeros([DIM, DIM])
    """Begin navigation"""
    number = len(nav_time)
    true_orbit = np.zeros([DIM, number])
    estimated_orbit = np.zeros([DIM, number])
    true_orbit[:, 0] = x0
    estimated_orbit[:, 0] = x0e
    time_cost = np.zeros([number - 1])
    # number = 100
    for k in range(number - 1):
        print("==== iter %d ====" % k)
        """state update"""
        xr = true_orbit[:, k]
        xe = estimated_orbit[:, k]
        t0 = nav_time[k]
        tf = nav_time[k + 1]
        """Propagate the true orbit"""
        t_eval = [t0, tf]
        check = solve_ivp(CRTBP_dynamics, [t0, tf], xr, args=(mu, ), method='RK45',
                          t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xr = check.y.T[-1, :]
        true_orbit[:, k + 1] = xr.T
        """Calculate the STM"""
        start = time.time()
        X = np.concatenate((xe, np.eye(6).reshape(36), np.zeros([6, 6, 6]).reshape(6 ** 3)))
        check = solve_ivp(CRTBP_STT_dynamics, [t0, tf], X, args=(mu, ), method='RK45',
                          t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xe = check.y.T[-1, :6].T
        STM = check.y.T[-1, 6:42].reshape(6, 6)
        STT = check.y.T[-1, 42:].reshape(6, 6, 6)
        """Propagate the prior covariance"""
        mf, Pf = STT_Pred_mu_P(P, STM, STT)
        P = Pf + Q
        xe = xe + mf
        """Measurement update"""
        dy, H, R = EKF_Measurement_y(xr, xe, STD)
        Pzz = np.matmul(np.matmul(H, P), H.T) + R
        Pxz = np.matmul(P, H.T)
        K_k = np.matmul(Pxz, np.linalg.inv(Pzz))
        """Update estimations"""
        estimated_orbit[:, k + 1] = xe + np.matmul(K_k, dy)
        P = P - np.matmul(np.matmul(K_k, Pzz), K_k.T)
        time_cost[k] = time.time() - start
    error = estimated_orbit - true_orbit
    error[:3, :] = error[:3, :] * UnitL
    error[3:, :] = error[3:, :] * UnitV * 1e6
    return error, time_cost

def SDEKF_Process(x0, errR, errV, nav_time, STD, UnitL, UnitV, mu):
    """Module of the SDEKF"""
    """Parameter setting"""
    DIM = 6
    RelTol = 10**-12
    AbsTol = 10**-12
    """Initial state"""
    r0 = x0[:3]
    v0 = x0[3:]
    """Parameters for orbit determination"""
    # r0e = r0 + errR * np.random.randn(3)
    # v0e = v0 + errV * np.random.randn(3)
    r0e = r0 + errR * np.ones(3)
    v0e = v0 + errV * np.ones(3)
    x0e = np.array([r0e[0], r0e[1], r0e[2], v0e[0], v0e[1], v0e[2]])
    error = np.array([errR, errR, errR, errV, errV, errV])
    P = np.diag(error ** 2)
    Q = np.zeros([DIM, DIM])
    """Begin navigation"""
    number = len(nav_time)
    true_orbit = np.zeros([DIM, number])
    estimated_orbit = np.zeros([DIM, number])
    true_orbit[:, 0] = x0
    estimated_orbit[:, 0] = x0e
    time_cost = np.zeros([number - 1])
    RMatrix = np.eye(DIM)
    dim_R = DIM
    # number = 100
    for k in range(number - 1):
        print("==== iter %d ====" % k)
        """state update"""
        xr = true_orbit[:, k]
        xe = estimated_orbit[:, k]
        t0 = nav_time[k]
        tf = nav_time[k + 1]
        """Propagate the true orbit"""
        t_eval = [t0, tf]
        check = solve_ivp(CRTBP_dynamics, [t0, tf], xr, args=(mu, ), method='RK45',
                          t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xr = check.y.T[-1, :]
        true_orbit[:, k + 1] = xr.T
        """Calculate the STM"""
        start = time.time()
        X = np.concatenate((xe, np.eye(6).reshape(36), np.zeros([6, dim_R, dim_R]).reshape(6 * (dim_R ** 2))))
        check = solve_ivp(CRTBP_DSTT_dynamics, [t0, tf], X, args=(mu, RMatrix, dim_R), method='RK45',
                          t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xe = check.y.T[-1, :6].T
        STM = check.y.T[-1, 6:42].reshape(6, 6)
        DSTT = check.y.T[-1, 42:].reshape(6, dim_R, dim_R)
        """Propagate the prior covariance"""
        mf, Pf = DSTT_Pred_mu_P(P, STM, DSTT, RMatrix, dim_R)
        P = Pf + Q
        xe = xe + mf
        """Measurement update"""
        dy, H, R = EKF_Measurement_y(xr, xe, STD)
        Pzz = np.matmul(np.matmul(H, P), H.T) + R
        Pxz = np.matmul(P, H.T)
        K_k = np.matmul(Pxz, np.linalg.inv(Pzz))
        """Update estimations"""
        estimated_orbit[:, k + 1] = xe + np.matmul(K_k, dy)
        P = P - np.matmul(np.matmul(K_k, Pzz), K_k.T)
        time_cost[k] = time.time() - start
        """Update R"""
        RMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        dim_R = 5
        """
        RMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ])
        dim_R = 3
        """
    error = estimated_orbit - true_orbit
    error[:3, :] = error[:3, :] * UnitL
    error[3:, :] = error[3:, :] * UnitV * 1e6
    return error, time_cost

if __name__ == '__main__':
    """Main test"""
    """Load data"""
    data = scipy.io.loadmat("NRHO_Scenario_Data.mat")
    UnitL = data["UnitL"][0, 0]
    UnitV = data["UnitV"][0, 0]
    UnitT = data["UnitT"][0, 0]
    mu = data["mu"][0, 0]
    nav_time = data["nav_time"][0]
    period = data["period"][0, 0]
    x0 = data["x0"].T[0]
    """EKF"""
    errR = 2.5e-5
    errV = 1e-6
    STD = np.array([1e-3 / UnitL])
    error, time_cost = EKF_Process(x0, errR, errV, nav_time, STD, UnitL, UnitV, mu)
    np.save("EKF_CPU_time.py", time_cost)
    error_EKF = error
    print((abs(error_EKF)).mean(axis=1))
    t_EKF = time_cost.mean()
    error, time_cost = SEKF_Process(x0, errR, errV, nav_time, STD, UnitL, UnitV, mu)
    np.save("SEKF_CPU_time.py", time_cost)
    error_SEKF = error
    print((abs(error_SEKF)).mean(axis=1))
    t_SEKF = time_cost.mean()
    error, time_cost = SDEKF_Process(x0, errR, errV, nav_time, STD, UnitL, UnitV, mu)
    np.save("SDEKF_CPU_time.py", time_cost)
    error_SDEKF = error
    print((abs(error_SDEKF)).mean(axis=1))
    t_SDEKF = time_cost.mean()
    """Table"""
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Method", justify="left")
    table.add_column("One-step CPU time (s)", justify="left")
    table.add_row(
        "[red bold]EKF[/red bold]", "[bold]%.4f s[/bold]" % t_EKF
    )
    table.add_row(
        "[blue bold]SEKF[/blue bold]", "[bold]%.4f s[/bold]" % t_SEKF
    )
    table.add_row(
        "[green bold]MDSTT-SEKF[/green bold]", "[bold]%.4f s[/bold]" % t_SDEKF
    )
    console.print(table)