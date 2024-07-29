import numpy as np
import math
import scipy
from scipy.integrate import solve_ivp
from Module_STT import STM_pred, STT_pred

def CRTBP_dynamics(t, y, mu):
    """the dyanmics of the CRTBP model"""
    r1 = math.sqrt((mu + y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2)
    r2 = math.sqrt((1 - mu - y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2)
    m1 = 1 - mu
    m2 = mu
    dydt = np.array([
        y[3],
        y[4],
        y[5],
        y[0] + 2 * y[4] + m1 * (-mu - y[0]) / (r1 ** 3) + m2 * (1 - mu - y[0]) / (r2 ** 3),
        y[1] - 2 * y[3] - m1 * (y[1]) / (r1 ** 3) - m2 * y[1] / (r2 ** 3),
        -m1 * y[2] / (r1 ** 3) - m2 * y[2] / (r2 ** 3)
    ])
    return dydt

def CRTBP_STM_dynamics(t, y, mu):
    """the dyanmics of the CRTBP model (with STM)"""
    x = y[:6]
    STM = y[6:].reshape(6, 6)
    """x"""
    dxdt = CRTBP_dynamics(t, x, mu)
    """STM"""
    A = cal_1st_tensor(x, mu)
    dSTM = np.matmul(A, STM).reshape(36)
    dy = np.concatenate((dxdt, dSTM))
    return dy

def CRTBP_STT_dynamics(t, y, mu):
    """the dyanmics of the CRTBP model (with STM and STT)"""
    x = y[:6]
    STM = y[6:42].reshape(6, 6)
    STT = y[42:].reshape(6, 6, 6)
    """x"""
    dxdt = CRTBP_dynamics(t, x, mu)
    """STM"""
    N1 = cal_1st_tensor(x, mu)
    dSTM = np.matmul(N1, STM).reshape(36)
    """STT"""
    N2 = cal_2rd_tensor(x, mu)
    dSTT = np.zeros([6, 6, 6])
    for i in range(6):
        for a in range(6):
            for b in range(6):
                for alpha in range(6):
                    dSTT[i, a, b] = dSTT[i, a, b] + N1[i, alpha] * STT[alpha, a, b]
                    for beta in range(6):
                        dSTT[i, a, b] = dSTT[i, a, b] + N2[i, alpha, beta] * STM[alpha, a] * STM[beta, b]
    dSTT = dSTT.reshape(6 ** 3)
    dy = np.concatenate((dxdt, dSTM, dSTT))
    return dy

def CRTBP_DSTT_dynamics(t, y, mu, R, dim):
    """the dyanmics of the CRTBP model (with STM and DSTT)"""
    x = y[:6]
    STM = y[6:42].reshape(6, 6)
    DSTT = y[42:].reshape(6, dim, dim)
    """x"""
    dxdt = CRTBP_dynamics(t, x, mu)
    """STM"""
    N1 = cal_1st_tensor(x, mu)
    dSTM = np.matmul(N1, STM).reshape(36)
    """DSTM"""
    DSTM = np.zeros([6, dim])
    for i in range(6):
        for k1 in range(dim):
            for l1 in range(dim):
                DSTM[i, k1] = DSTM[i, k1] + STM[i, l1] * R[k1, l1]
    """DSTT"""
    N2 = cal_2rd_tensor(x, mu)
    dSTT = np.zeros([6, dim, dim])
    for i in range(6):
        for a in range(dim):
            for b in range(dim):
                for alpha in range(6):
                    dSTT[i, a, b] = dSTT[i, a, b] + N1[i, alpha] * DSTT[alpha, a, b]
                    for beta in range(6):
                        dSTT[i, a, b] = dSTT[i, a, b] + N2[i, alpha, beta] * DSTM[alpha, a] * DSTM[beta, b]
    dSTT = dSTT.reshape(6 * (dim ** 2))
    dy = np.concatenate((dxdt, dSTM, dSTT))
    return dy

def cal_1st_tensor(x, mu):
    """the first-order tensor of the CRTBP dynamcis"""
    rx = x[0]
    ry = x[1]
    rz = x[2]
    daxdrx = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) - mu / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) + (3 * mu * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) + 1
    daxdry = (3 * mu * ry * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    daxdrz = (3 * mu * rz * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    daxdvx = 0
    daxdvy = 2
    daxdvz = 0
    daydrx = (3 * mu * ry * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * ry * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2))
    daydry = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) - mu / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) + (3 * mu * ry ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + 1
    daydrz = (3 * mu * ry * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    daydvx = -2
    daydvy = 0
    daydvz = 0
    dazdrx = (3 * mu * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2))
    dazdry = (3 * mu * ry * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    dazdrz = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) - mu / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) + (3 * mu * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    dazdvx = 0
    dazdvy = 0
    dazdvz = 0
    """Jacobi matrix"""
    A = np.zeros([6, 6])
    A[:3, 3:] = np.eye(3)
    A[3:, :] = np.array([
        [daxdrx, daxdry, daxdrz, daxdvx, daxdvy, daxdvz],
        [daydrx, daydry, daydrz, daydvx, daydvy, daydvz],
        [dazdrx, dazdry, dazdrz, dazdvx, dazdvy, dazdvz],
    ])
    return A

def cal_2rd_tensor(x, mu):
    """the second-order tensor of the CRTBP dynamcis"""
    rx = x[0]
    ry = x[1]
    rz = x[2]
    A = np.zeros([6, 6, 6])
    """elements of A"""
    daxdrxrx = (3 * mu * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + (3 * mu * (2 * mu + 2 * rx - 2)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * (mu + rx - 1) * (2 * mu + 2 * rx - 2) ** 2) / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx) ** 2) / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdrxry = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdrxrz = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdrxvx = 0
    daxdrxvy = 0
    daxdrxvz = 0
    daxdryrx = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdryry = (3 * mu * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + (15 * ry ** 2 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * ry ** 2 * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daxdryrz = (15 * ry * rz * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * ry * rz * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daxdryvx = 0
    daxdryvy = 0
    daxdryvz = 0
    daxdrzrx = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdrzry = (15 * ry * rz * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * ry * rz * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daxdrzrz = (3 * mu * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + (15 * rz ** 2 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * rz ** 2 * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daxdrzvx = 0
    daxdrzvy = 0
    daxdrzvz = 0
    daxdvxrx = 0
    daxdvxry = 0
    daxdvxrz = 0
    daxdvxvx = 0
    daxdvxvy = 0
    daxdvxvz = 0
    daxdvyrx = 0
    daxdvyry = 0
    daxdvyrz = 0
    daxdvyvx = 0
    daxdvyvy = 0
    daxdvyvz = 0
    daxdvzrx = 0
    daxdvzry = 0
    daxdvzrz = 0
    daxdvzvx = 0
    daxdvzvy = 0
    daxdvzvz = 0
    daydrxrx = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * (2 * mu + 2 * rx - 2) ** 2) / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry * (mu - 1) * (2 * mu + 2 * rx) ** 2) / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydrxry = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * ry ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydrxrz = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydrxvx = 0
    daydrxvy = 0
    daydrxvz = 0
    daydryrx = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * ry ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydryry = (9 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (9 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 3) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 3 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydryrz = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 2 * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 2 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydryvx = 0
    daydryvy = 0
    daydryvz = 0
    daydrzrx = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydrzry = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 2 * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 2 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydrzrz = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydrzvx = 0
    daydrzvy = 0
    daydrzvz = 0
    daydvxrx = 0
    daydvxry = 0
    daydvxrz = 0
    daydvxvx = 0
    daydvxvy = 0
    daydvxvz = 0
    daydvyrx = 0
    daydvyry = 0
    daydvyrz = 0
    daydvyvx = 0
    daydvyvy = 0
    daydvyvz = 0
    daydvzrx = 0
    daydvzry = 0
    daydvzrz = 0
    daydvzvx = 0
    daydvzvy = 0
    daydvzvz = 0
    dazdrxrx = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz * (2 * mu + 2 * rx - 2) ** 2) / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz * (mu - 1) * (2 * mu + 2 * rx) ** 2) / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdrxry = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdrxrz = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * rz ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdrxvx = 0
    dazdrxvy = 0
    dazdrxvz = 0
    dazdryrx = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdryry = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 2 * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 2 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdryrz = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdryvx = 0
    dazdryvy = 0
    dazdryvz = 0
    dazdrzrx = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * rz ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdrzry = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdrzrz = (9 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (9 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz ** 3) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * rz ** 3 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdrzvx = 0
    dazdrzvy = 0
    dazdrzvz = 0
    dazdvxrx = 0
    dazdvxry = 0
    dazdvxrz = 0
    dazdvxvx = 0
    dazdvxvy = 0
    dazdvxvz = 0
    dazdvyrx = 0
    dazdvyry = 0
    dazdvyrz = 0
    dazdvyvx = 0
    dazdvyvy = 0
    dazdvyvz = 0
    dazdvzrx = 0
    dazdvzry = 0
    dazdvzrz = 0
    dazdvzvx = 0
    dazdvzvy = 0
    dazdvzvz = 0
    A[3] = np.array([
        [daxdrxrx, daxdrxry, daxdrxrz, daxdrxvx, daxdrxvy, daxdrxvz],
        [daxdryrx, daxdryry, daxdryrz, daxdryvx, daxdryvy, daxdryvz],
        [daxdrzrx, daxdrzry, daxdrzrz, daxdrzvx, daxdrzvy, daxdrzvz],
        [daxdvxrx, daxdvxry, daxdvxrz, daxdvxvx, daxdvxvy, daxdvxvz],
        [daxdvyrx, daxdvyry, daxdvyrz, daxdvyvx, daxdvyvy, daxdvyvz],
        [daxdvzrx, daxdvzry, daxdvzrz, daxdvzvx, daxdvzvy, daxdvzvz],
    ])
    A[4] = np.array([
        [daydrxrx, daydrxry, daydrxrz, daydrxvx, daydrxvy, daydrxvz],
        [daydryrx, daydryry, daydryrz, daydryvx, daydryvy, daydryvz],
        [daydrzrx, daydrzry, daydrzrz, daydrzvx, daydrzvy, daydrzvz],
        [daydvxrx, daydvxry, daydvxrz, daydvxvx, daydvxvy, daydvxvz],
        [daydvyrx, daydvyry, daydvyrz, daydvyvx, daydvyvy, daydvyvz],
        [daydvzrx, daydvzry, daydvzrz, daydvzvx, daydvzvy, daydvzvz],
    ])
    A[5] = np.array([
        [dazdrxrx, dazdrxry, dazdrxrz, dazdrxvx, dazdrxvy, dazdrxvz],
        [dazdryrx, dazdryry, dazdryrz, dazdryvx, dazdryvy, dazdryvz],
        [dazdrzrx, dazdrzry, dazdrzrz, dazdrzvx, dazdrzvy, dazdrzvz],
        [dazdvxrx, dazdvxry, dazdvxrz, dazdvxvx, dazdvxvy, dazdvxvz],
        [dazdvyrx, dazdvyry, dazdvyrz, dazdvyvx, dazdvyvy, dazdvyvz],
        [dazdvzrx, dazdvzry, dazdvzrz, dazdvzvx, dazdvzvy, dazdvzvz],
    ])
    return A

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
    """Propagate orbit"""
    t0 = 0
    tf = period / 2
    t_eval = [t0, tf]
    RelTol = 10 ** -8
    AbsTol = 10 ** -8
    print("===== Nominal orbit =====")
    check = solve_ivp(CRTBP_dynamics, [t0, tf], x0, args=(mu,), method='RK45',
                      t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    xf = check.y.T[-1, :]
    """Propagate deviation orbit"""
    errR = 2.5e-5
    errV = 1e-6
    dx0 = np.array([errR, errR, errR, errV, errV, errV])
    print("===== Deviation orbit =====")
    check = solve_ivp(CRTBP_dynamics, [t0, tf], x0 + dx0, args=(mu,), method='RK45',
                      t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    xf_ = check.y.T[-1, :]
    """Propagate the STM"""
    y0_STM = np.concatenate((x0, np.eye(6).reshape(36)))
    print("===== STM =====")
    check = solve_ivp(CRTBP_STM_dynamics, [t0, tf], y0_STM, args=(mu,), method='RK45',
                      t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    STM = check.y.T[-1, 6:].reshape(6, 6)
    """Propagate the STT"""
    y0_STT = np.concatenate((x0, np.eye(6).reshape(36), np.zeros([6, 6, 6]).reshape(6 ** 3)))
    print("===== STT =====")
    check = solve_ivp(CRTBP_STT_dynamics, [t0, tf], y0_STT, args=(mu,), method='RK45',
                      t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    STT = check.y.T[-1, 42:].reshape(6, 6, 6)
    """Accuracy validation"""
    dxf = xf_ - xf
    dxf1 = STM_pred(STM, dx0)
    dxf2 = STM_pred(STM, dx0) + STT_pred(STT, dx0, dx0)
    RE_STM = abs(dxf - dxf1) / abs(dxf) * 100
    RE_STT = abs(dxf - dxf2) / abs(dxf) * 100
    print("RE_STM =", RE_STM)
    print("RE_STT =", RE_STT)