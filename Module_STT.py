from math import *
import numpy as np
from jplephem.spk import SPK
from scipy.integrate import solve_ivp
import time

def func_STM_Dyn(rs, rd, miu):
    """
    Calculate the first-order derivatives of the dynamics
    """
    rsx = rs[0]  # spacecraft
    rsy = rs[1]
    rsz = rs[2]
    rdx = rd[0]  # disturbed body
    rdy = rd[1]
    rdz = rd[2]
    d_ax_x = miu*(rdx - rsx)*(3*rdx - 3*rsx)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2) - miu/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(3/2)
    d_ax_y = miu*(rdx - rsx)*(3*rdy - 3*rsy)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ax_z = miu*(rdx - rsx)*(3*rdz - 3*rsz)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ay_x = miu*(3*rdx - 3*rsx)*(rdy - rsy)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ay_y = miu*(rdy - rsy)*(3*rdy - 3*rsy)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2) - miu/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(3/2)
    d_ay_z = miu*(rdy - rsy)*(3*rdz - 3*rsz)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_az_x = miu*(3*rdx - 3*rsx)*(rdz - rsz)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_az_y = miu*(3*rdy - 3*rsy)*(rdz - rsz)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_az_z = miu*(rdz - rsz)*(3*rdz - 3*rsz)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2) - miu/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(3/2)
    STM_dyn = np.array([
        [d_ax_x, d_ax_y, d_ax_z],
        [d_ay_x, d_ay_y, d_ay_z],
        [d_az_x, d_az_y, d_az_z],
    ])
    return STM_dyn

def func_STT_Dyn(rs, rd, miu):
    """
    Calculate the second-order derivatives of the dynamics
    """
    rsx = rs[0]  # spacecraft
    rsy = rs[1]
    rsz = rs[2]
    rdx = rd[0]  # disturbed body
    rdy = rd[1]
    rdz = rd[2]
    d_ax_xx = 3*miu*(rdx - rsx)*(5*(rdx - rsx)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 3)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ax_xy = 3*miu*(rdy - rsy)*(5*(rdx - rsx)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ax_xz = 3*miu*(rdz - rsz)*(5*(rdx - rsx)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ax_yx = 3*miu*(rdy - rsy)*(5*(rdx - rsx)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ax_yy = 3*miu*(rdx - rsx)*(5*(rdy - rsy)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ax_yz = 15*miu*(rdx - rsx)*(rdy - rsy)*(rdz - rsz)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(7/2)
    d_ax_zx = 3*miu*(rdz - rsz)*(5*(rdx - rsx)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ax_zy = 15*miu*(rdx - rsx)*(rdy - rsy)*(rdz - rsz)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(7/2)
    d_ax_zz = 3*miu*(rdx - rsx)*(5*(rdz - rsz)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ay_xx = 3*miu*(rdy - rsy)*(5*(rdx - rsx)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ay_xy = 3*miu*(rdx - rsx)*(5*(rdy - rsy)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ay_xz = 15*miu*(rdx - rsx)*(rdy - rsy)*(rdz - rsz)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(7/2)
    d_ay_yx = 3*miu*(rdx - rsx)*(5*(rdy - rsy)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ay_yy = 3*miu*(rdy - rsy)*(5*(rdy - rsy)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 3)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ay_yz = 3*miu*(rdz - rsz)*(5*(rdy - rsy)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ay_zx = 15*miu*(rdx - rsx)*(rdy - rsy)*(rdz - rsz)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(7/2)
    d_ay_zy = 3*miu*(rdz - rsz)*(5*(rdy - rsy)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_ay_zz = 3*miu*(rdy - rsy)*(5*(rdz - rsz)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_az_xx = 3*miu*(rdz - rsz)*(5*(rdx - rsx)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_az_xy = 15*miu*(rdx - rsx)*(rdy - rsy)*(rdz - rsz)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(7/2)
    d_az_xz = 3*miu*(rdx - rsx)*(5*(rdz - rsz)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_az_yx = 15*miu*(rdx - rsx)*(rdy - rsy)*(rdz - rsz)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(7/2)
    d_az_yy = 3*miu*(rdz - rsz)*(5*(rdy - rsy)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_az_yz = 3*miu*(rdy - rsy)*(5*(rdz - rsz)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_az_zx = 3*miu*(rdx - rsx)*(5*(rdz - rsz)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_az_zy = 3*miu*(rdy - rsy)*(5*(rdz - rsz)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 1)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    d_az_zz = 3*miu*(rdz - rsz)*(5*(rdz - rsz)**2/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2) - 3)/((rdx - rsx)**2 + (rdy - rsy)**2 + (rdz - rsz)**2)**(5/2)
    STT_dyn = np.zeros([6, 6, 6])
    STT_dyn[3, :3, :3] = np.array([
        [d_ax_xx, d_ax_xy, d_ax_xz],
        [d_ax_yx, d_ax_yy, d_ax_yz],
        [d_ax_zx, d_ax_zy, d_ax_zz]
    ])
    STT_dyn[4, :3, :3] = np.array([
        [d_ay_xx, d_ay_xy, d_ay_xz],
        [d_ay_yx, d_ay_yy, d_ay_yz],
        [d_ay_zx, d_ay_zy, d_ay_zz]
    ])
    STT_dyn[5, :3, :3] = np.array([
        [d_az_xx, d_az_xy, d_az_xz],
        [d_az_yx, d_az_yy, d_az_yz],
        [d_az_zx, d_az_zy, d_az_zz]
    ])
    return STT_dyn

def STM_pred(STM, x):
    """
    Calculate the first-order term using STM
    """
    pred = np.mat(STM) * np.mat(x).T
    return pred.T.A[0]

def STT_pred(STT, x, y):
    """
    Calculate the second-order term using STT
    """
    lx = len(x)
    lpred = len(STT)
    pred = np.zeros([lpred])
    for i in range(lpred):
        for k1 in range(lx):
            for k2 in range(lx):
                pred[i] = pred[i] + 1 / 2 * STT[i, k1, k2] * x[k1] * y[k2]
    return pred

def STM_Combine(STM, B):
    """
    Combine two STMs
    """
    pred = np.mat(STM) * np.mat(B)
    return pred.A

def STT_Combine(N1, N2, STM, STT):
    """
    Combine two STTs
    """
    len_i = len(N1[:,0])
    len_y = len(N1[0,:])
    len_x = len(STM[0,:])
    STT_Combined = np.zeros([len_i, len_x, len_x])
    for i in range(len_i):
        for a in range(len_x):
            for b in range(len_x):
                for alpha in range(len_y):
                    STT_Combined[i, a, b] = STT_Combined[i, a, b] + N1[i, alpha] * STT[alpha, a, b]
                    for beta in range(len_y):
                        STT_Combined[i, a, b] = STT_Combined[i, a, b] \
                                                + N2[i, alpha, beta] * STM[alpha, a] * STM[beta, b]
    return STT_Combined

def STT_Pred_mu_P(P0, STM, STT):
    """Calculate the mean and covariance using the STTs"""
    n, m = len(STM), np.size(STM[0])
    """Mean"""
    mf = np.zeros([n])
    for i in range(n):
        for i1 in range(m):
            for i2 in range(m):
                mf[i] += STT[i, i1, i2] * P0[i1, i2] / 2
    """Covariance"""
    Pf = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            Pf[i, j] = -mf[i] * mf[j]
            """First-order"""
            for a in range(m):
                for b in range(m):
                    Pf[i, j] = Pf[i, j] + STM[i, a] * STM[j, b] * P0[a, b]
            """Second-order"""
            for a in range(m):
                for b in range(m):
                    for alpha in range(m):
                        for beta in range(m):
                            Pf[i, j] = Pf[i, j] + STT[i, a, b] * STT[j, alpha, beta] * (P0[a, b] * P0[alpha, beta] + P0[a, alpha] * P0[b, beta] + P0[a, beta] * P0[b, alpha]) / 4
    return mf, Pf

def DSTT_Pred_mu_P(P0, STM, DSTT, R, dim):
    """Calculate the mean and covariance using the DSTTs"""
    n, m = len(STM), np.size(STM[0])
    R = np.mat(R)
    P0R = np.matmul(np.matmul(R, P0), R.T)
    """Mean"""
    mf = np.zeros([n])
    for i in range(n):
        for i1 in range(dim):
            for i2 in range(dim):
                mf[i] += DSTT[i, i1, i2] * P0R[i1, i2] / 2
    """Covariance"""
    Pf = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            Pf[i, j] = -mf[i] * mf[j]
            """First-order"""
            for a in range(m):
                for b in range(m):
                    Pf[i, j] = Pf[i, j] + STM[i, a] * STM[j, b] * P0[a, b]
            """Second-order"""
            for a in range(dim):
                for b in range(dim):
                    for alpha in range(dim):
                        for beta in range(dim):
                            Pf[i, j] = Pf[i, j] + DSTT[i, a, b] * DSTT[j, alpha, beta] * (P0R[a, b] * P0R[alpha, beta] + P0R[a, alpha] * P0R[b, beta] + P0R[a, beta] * P0R[b, alpha]) / 4
    return mf, Pf