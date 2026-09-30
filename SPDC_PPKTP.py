import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.ticker as mticker
import scipy as sp
from scipy.special import erfi, erf
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
import Type2_integrals as t2_int


L = 2e-3
lambda_p = 405e-9
lambda_s = lambda_i = 2*lambda_p
poling_period = 10e-6
theta_s = theta_i = np.deg2rad(3)

Wp = 310e-6 #beam waist of pump laser (in meters)
Wi = Ws = 145.4e-6 #beam waist of signal/idler (in meters)

Bp = 30e12 #pump spectral bandwith (in Hz)
dws = dwi = 10e12 #allowed angular frequency range of signal and idler by Bandpass Filter (in Hz)
dwp = dws+dwi
E0 = 8.854187817e-12 #F/m (permittivity of free space)
c = 299792458 #m/s (speed of light)

eta_s = 1 #efficiency of filters
eta_i = 1 #efficiency of filters

d = (2*7.6e-12)/np.pi

t = 140e-15 #pulse duration of femtosecond pulse


temp_fit_coeff_n1 = pd.DataFrame({
    'z': [9.9587e-6, 9.9228e-6, -8.9603e-6, 4.1010e-6],
    'y': [6.2897e-6, 6.3061e-6, -6.0629e-6, 2.6486e-6]
})

temp_fit_coeff_n2 = pd.DataFrame({
    'z': [-1.1882e-8, 10.459e-8, -9.8136e-8, 3.1481e-8],
    'y': [-0.14445e-8, 2.2244e-8, -3.5770e-8, 1.3470e-8]
})

fit_coeff_list = [temp_fit_coeff_n1, temp_fit_coeff_n2]

def factorial(x):
    if x==0:
        return 1
    else:
        return x * factorial(x-1)

def w(l):
    return (2 * np.pi * c) / l

def n_12(l, n12, xyz):
    n = 0
    for m in range(4):
        n += (fit_coeff_list[n12 - 1][xyz][m] / (l ** m))
    return n

def nx(l,T):
    A, B, C, D = 2.1146, 0.89188, 0.20861, 0.01320
    RI = np.sqrt(A + (B / (1 - ((C/(l * 1e6))**2))) - D * ((l * 1e6) ** 2))
    
    RI += ((0.1717 / l**3) - (0.5353 / l**2) + (0.8416 / l) + 0.1627)*1e-5*(T-25)

    return RI

def ny(l, T):
    if l == 406.2e-9:
        A, B, C, D = 2.1518, 0.87862, 0.21801, 0.01327
        RI = np.sqrt(A + (B / (1 - ((C/(l * 1e6))**2))) - D * ((l * 1e6) ** 2))
        RI += 3.5e-5 * (T-25)
        
    elif l==405e-9:
        A, B, C, D = 2.1518, 0.87862, 0.21801, 0.01327
        RI = np.sqrt(A + (B / (1 - ((C/(l * 1e6))**2))) - D * ((l * 1e6) ** 2))
        RI += 6.3e-5 * (T-25)
    
    else: 
        A, B, C, D = 2.1518, 0.87862, 0.21801, 0.01327
        RI = np.sqrt(A + (B / (1 - ((C/(l * 1e6))**2))) - D * ((l * 1e6) ** 2))
        RI += (n_12(l * 1e6, 1, 'y') * (T - 25) + n_12(l * 1e6, 2, 'y') * ((T - 25) ** 2))

    return RI

def nz(l, T):
    A, B, C, D = 2.3136, 1.00012, 0.23831, 0.01679
    RI = np.sqrt(A + (B / (1 - ((C/(l * 1e6))**2))) - D * ((l * 1e6) ** 2))
    RI += (n_12(l * 1e6, 1, 'z') * (T - 25) + n_12(l * 1e6, 2, 'z') * ((T - 25) ** 2))

    return RI

def Kj(lj, nj):
    return (nj * 2 * np.pi) / lj

def Nj(noj): 
    return noj / c

def A(Wp, Ws, Wi): #Defiend constant A
    return (1 / Wp ** 2) + (1 / Ws ** 2) + (1 / Wi ** 2)

def C(Wp, Ws, Wi): #Defiend constant C
    return (1 / Wp ** 2) + (np.cos(theta_s)**2 / Ws ** 2) + (np.cos(theta_i)**2 / Wi ** 2)

def alpha_j(Woj,n=0,m=0):
    Denominator = ((2 ** (n + m)) * factorial(n) * factorial(m) * np.pi * (Woj ** 2))
    return np.sqrt(2 / Denominator)

def Wp_Ws_relation(Ni,Ns, Np, W0_p,L): #Relation between pump and signal/idler beam Waist
    alpha = 0.455
    cos_theta_s = np.cos(theta_s)
    cos_theta_i = np.cos(theta_i)
    sin_theta_s = np.sin(theta_s)
    sin_theta_i = np.sin(theta_i)

    term1 = np.sqrt(cos_theta_s ** 2 + cos_theta_i ** 2)

    numerator = Ni * Ns * sin_theta_i * sin_theta_s
    denominator = (1 / Bp ** 2) + (alpha ** 2 * L ** 2 * (Np - Ni * cos_theta_i) * (Np - Ns * cos_theta_s))

    term2 = (numerator / denominator) - (1 / W0_p ** 2)

    # return term1 / np.sqrt(term2)

    result = np.full_like(term2, np.nan)  # Initialize an array of zeros with the same shape as term2
    positive_indices = term2 > 0  # Find indices where term2 > 0
    result[positive_indices] = term1 / np.sqrt(term2[positive_indices])  # Compute only for positive term2
    return result

def DSApart(ws, wi, Bp, wso, wio): # Phase matching function
    return np.exp(-(((wi + ws - wso - wio)*t) ** 2 / (8*np.log(2))))

def Integration_part(A, C, L, kp, ks, ki, dky, dkz, Bp, wi, wso, wio, dws, dwi, Wop, Wos, Woi, theta_s, theta_i, inte, n, m):
    ws_min, ws_max = wso - (dws / 2), wso + (dws / 2)
    wi_min, wi_max = wio - ((dwi) / 2), wio + ((dwi) / 2)
    
    result, error = sp.integrate.dblquad(
        lambda ws, wi: DSApart(ws, wi, Bp, wso, wio),
        ws_min, ws_max,
        lambda ws: wi_min, lambda ws: wi_max
    )

    return result * inte

def dkz_zero(lp, temp, angle_s = 0):
    # Calculate refractive indices at temperature T
    ls = li = 2*lp
    ni = np.sqrt(((nx(li, temp)**2) * (ny(li, temp)**2) ) / ( ((ny(li, temp)**2) * (np.sin(angle_s)**2)) + ((nx(li,temp)**2) * (np.cos(angle_s)**2))))
    ns = nz(ls, temp)
    n_p = ny(lp, temp)

    Kp = Kj(lp, n_p)
    Ks = Kj(ls, ns)
    Ki = Kj(li, ni)
    K = 2*np.pi/poling_period

    return Kp - Ks * np.cos(angle_s) - Ki * np.cos(angle_s) - K

def Phase_Match_Temp(lp):
    # Initial guess for temperature (°C)
    initial_guess = 80

    # Solve for temperature where dky = 0
    temp_solution = fsolve(lambda temp: dkz_zero(lp, temp, theta_s), initial_guess)[0]

    print("Temperature where Δk_y = 0:", temp_solution, "°C")

    # Plot Δk_y vs Temperature
    temp_range = np.linspace(0, 150, 1000)
    dky_values = [dkz_zero(lp, T, theta_s) for T in temp_range]

    plt.plot(temp_range, dky_values)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Δk_y (1/m)')
    plt.title('Phase Mismatch Δk_y vs Temperature')
    plt.axhline(0, color='red', linestyle='--', label=f'Δk_y = 0 at Temp: {temp_solution} °C')
    plt.grid(True)
    plt.legend()
    plt.show()

    return temp_solution

def Bandwidth(temp, lam_p, df=20e12):
    FWHMs = []
    Ls = np.linspace(0.1e-3, 30e-3, 100)

    Wop = 119.22e-6
    Wos = Woi = Wop*np.sqrt(2)

    ni = np.sqrt(((nx(lambda_i, temp_sol)**2) * (ny(lambda_i, temp_sol)**2) ) / ( ((ny(lambda_i, temp_sol)**2) * (np.sin(theta_i)**2)) + ((nx(lambda_i, temp_sol)**2) * (np.cos(theta_i)**2))))
    ns = ny(lambda_s, temp_sol)
    n_p = ny(lambda_p, temp_sol)

    for L in Ls:
        center_freq = c / lam_p
        vp = np.linspace(center_freq-df, center_freq + df, 1000)
        lp = c/vp

        L_corr = L*(1 + (6.7e-6)*(temp-25) + (11e-9)*(temp-25)**2)

        dk_values = []
        for lps in lp:
            dk_values.append(dkz_zero(lps, temp))
        dkz = np.array(dk_values)


        z = t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=0) * t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L_corr,0, dkz, theta_s, theta_i, Wop, Wos, Woi, m=0)
        z = abs(z) ** 2
        final_term = Integration_part(A(Wop, Wos, Woi), C(Wop, Wos, Woi), L_corr, Kj(lambda_p,n_p), Kj(lambda_s,ns), Kj(lambda_i,ni), 0, dkz, Bp, wi, wos, woi, dws, dwi, Wop, Wos, Woi, theta_s, theta_i, z,n=0, m=0)

        half_max = max(final_term)/2 
        crossings = np.where(np.diff(np.sign(final_term - half_max)))[0]

        if len(crossings) >= 2:
            # Interpolate to find more accurate crossing points
            f_interp_left = interp1d(final_term[crossings[0]:crossings[0] + 2], vp[crossings[0]:crossings[0] + 2])
            v_left = f_interp_left(half_max)

            f_interp_right = interp1d(final_term[crossings[1]:crossings[1] + 2], vp[crossings[1]:crossings[1] + 2])
            v_right = f_interp_right(half_max)

            FWHM_v = (v_right - v_left) * 1e-9  # GHz
            FWHM_l = ((c / v_left) - (c / v_right)) * 1e9  # nm
            FWHMs.append(FWHM_l)

            center_freq = (v_left + v_right) / 2
            FWHM_freq = v_right - v_left

    plt.plot(Ls, FWHMs)
    plt.xlabel('L (m)')
    plt.ylabel('FWHM (nm)')
    plt.title('Bandwidth vs Length')
    plt.grid(True)
    plt.show() 
    #     fig, ax1 = plt.subplots()

    #     # Plot full curve
    #     ax1.plot(vp, final_term, label='Sinc term')

    #     # Highlight FWHM portion of the curve
    #     fwhm_mask = (vp >= v_left) & (vp <= v_right)
    #     ax1.plot(vp[fwhm_mask], final_term[fwhm_mask], color='blue', linewidth=2.5, label='FWHM region')

    #     ax1.set_xlabel('vp (Frequency in Hz)')
    #     ax1.set_ylabel('sinc_term')
    #     ax1.set_title(f'Sinc Term vs vp at Temp = {temp}°C')
    #     ax1.grid(True)

    #     # Mark FWHM region on frequency axis (bottom)
    #     ax1.axvline(v_left, color='red', linestyle='--', label='FWHM freq start')
    #     ax1.axvline(v_right, color='red', linestyle='--', label='FWHM freq end')
    #     ax1.fill_betweenx([0, 1], v_left, v_right, color='red', alpha=0.2)

    #     # Create a second x-axis on top for wavelength
    #     ax2 = ax1.twiny()
    #     ax2.set_xlim(lp[0], lp[-1])

    #     freq_ticks = ax1.get_xticks()
    #     wavelength_ticks = np.interp(freq_ticks, vp, lp)
    #     ax2.set_xticks(wavelength_ticks)
    #     ax2.set_xticklabels([f"{wl*1e9:.1f}" for wl in wavelength_ticks])
    #     ax2.set_xlabel('Wavelength (nm)')

    #     # Mark FWHM region on wavelength axis (top)
    #     left_wavelength = c / v_left
    #     right_wavelength = c / v_right
    #     ax2.axvline(left_wavelength, color='green', linestyle='--', label='FWHM wavelength start')
    #     ax2.axvline(right_wavelength, color='green', linestyle='--', label='FWHM wavelength end')

    #     ax1.plot([], [], ' ')
    #     ax1.legend(loc='upper right', fontsize=8)
    #     plt.tight_layout()
    #     plt.show()

    # else:
    #     print("Could not find two crossings at half maximum.")



def Angle_Temp(lp):
    angle_s = np.linspace(0, np.deg2rad(3), 1000)

    temp_solutions = list()

    for angle in angle_s:
        initial_guess = 80
        temp_solution = fsolve(lambda temp: dkz_zero(lp, temp, angle), initial_guess)[0]
        temp_solutions.append(temp_solution)

    angle_s = np.rad2deg(angle_s)
    plt.plot(temp_solutions, angle_s)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Angle (°)')
    plt.title('Angle vs Temperature')
    # plt.axhline(0, color='red', linestyle='--', label=f'Δk_y = 0 at Temp: {temp_solution} °C')
    plt.grid(True)
    # plt.legend()
    plt.show()

    return temp_solution

# Angular Frequency
ws = wi = w(lambda_s)  
wp = w(lambda_p)
wos = ws
woi = wi

temp_sol = Phase_Match_Temp(lambda_p)
Bandwidth(temp_sol, lambda_p)

# Refractive Index
ni = np.sqrt(((nx(lambda_i, temp_sol)**2) * (ny(lambda_i, temp_sol)**2) ) / ( ((ny(lambda_i, temp_sol)**2) * (np.sin(theta_i)**2)) + ((nx(lambda_i, temp_sol)**2) * (np.cos(theta_i)**2))))
ns = ny(lambda_s, temp_sol)
n_p = ny(lambda_p, temp_sol)

nos = ns
noi = ni
nop = n_p

# Wavevector
dky = (Kj(lambda_s,ns) * np.sin(np.radians(theta_s))) - (Kj(lambda_i,ni) * np.sin(np.radians(theta_i)))
dkz = dkz_zero(lambda_p, temp_sol, theta_s)

Ns = Nj(nos)
Ni = Nj(noi)
Np = Nj(nop)
    
# Bandwidth(temp_sol, lambda_p)
# angle_temp_solution = Angle_Temp(lambda_p)

# def JSI():
#     Wop, Wos, Woi = Wp, Ws, Wi

#     lambda_s= lambda_i = np.linspace(790, 830, 1000)
#     ws_vals = w(lambda_s)
#     wi_vals = w(lambda_i)
#     ws, wi = np.meshgrid(ws_vals, wi_vals, indexing='ij')  # shape (1000, 1000)


#     ni = np.sqrt(((nx(lambda_i, temp_sol)**2) * (ny(lambda_i, temp_sol)**2) ) / ( ((ny(lambda_i, temp_sol)**2) * (np.sin(theta_i)**2)) + ((nx(lambda_i, temp_sol)**2) * (np.cos(theta_i)**2))))
#     ns = nz(lambda_s, temp_sol)
#     n_p = ny(lambda_p, temp_sol)

#     dky = (Kj(lambda_s,ns) * np.sin(np.radians(theta_s))) - (Kj(lambda_i,ni) * np.sin(np.radians(theta_i)))
#     dkz = dkz_zero(lambda_p, temp_sol, theta_s)

#     z = t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=0) * t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=0)
#     z = abs(z) ** 2
#     JSA = Integration_part(A(Wop, Wos, Woi), C(Wop, Wos, Woi), L, Kj(lambda_p,n_p), Kj(lambda_s,ns), Kj(lambda_i,ni), dky, dkz, Bp, wi, ws, wi, dws, dwi, Wop, Wos, Woi, theta_s, theta_i, z,n=0, m=0)

#     plt.figure(figsize=(7, 6))
#     sns.heatmap(JSA, cmap='PRGn', cbar=True, xticklabels=200, yticklabels=200)

#     # Optional: Map tick positions to wavelength values
#     xticks_locs = np.linspace(0, len(wi_vals)-1, 5, dtype=int)
#     yticks_locs = np.linspace(0, len(ws_vals)-1, 5, dtype=int)
#     plt.xticks(xticks_locs, np.round(wi_vals[xticks_locs], 1))
#     plt.yticks(yticks_locs, np.round(ws_vals[yticks_locs], 1))

#     plt.title("Joint Spectral Intensity (PRGn colormap)")
#     plt.xlabel("Woi (nm)")
#     plt.ylabel("Wos (nm)")
#     plt.tight_layout()
#     plt.show()

def Calc_Max_R():
    Wop = np.linspace(75e-6,2e-3,10000)
    Woi = Wos = Wp_Ws_relation(Ni,Ns,Np,Wop,L)
    # Wop = 100e-6
    # Woi = Wos = Wop*np.sqrt(2)

    z = t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=0) * t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=0)
    z = abs(z) ** 2

    # Calculation of Pair Production Rate
    Numerator = ( eta_s * eta_i * (d**2) * (alpha_j(Woi)**2) * (alpha_j(Wos)**2) * (alpha_j(Wop)**2) * ws * wi)
    Denominator = np.sqrt(2) * np.sqrt(np.pi**3) * E0 * (c**3) * ns * ni * n_p * Bp
    integration_part = Integration_part(A(Wop, Wos, Woi), C(Wop, Wos, Woi), L, Kj(lambda_p,n_p), Kj(lambda_s,ns), Kj(lambda_i,ni), dky, dkz, Bp, wi, wos, woi, dws, dwi, Wop, Wos, Woi, theta_s, theta_i, z,n=0, m=0)

    R = (Numerator/Denominator) * integration_part

    max = np.max(R)
    print("MAX R:", max*1e-3)

    index = np.where(R==max)
    print("Wop for max R:", Wop[index][0]*(10**6))

    plt.figure(figsize=(6, 4))  # Standard size for paper figures

    # Plot the curve
    plt.plot(Wop * 1e6, R * 1e-3, color='blue', linewidth=1.5, label='R/P vs Wop')

    # Labels and title
    plt.xlabel(r'$W_{o,p}$ (µm)', fontsize=12)
    plt.ylabel(r'$R/P$ (pairs/s·mW)', fontsize=12)
    plt.title(r'Pair Rate per Pump Power vs $W_{o,p}$', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Annotate maximum point
    plt.annotate(
        fr'Max $R$: {max * 1e-3:.2f} pairs/s·mW' + f'\nat $W_{{o,p}}$: {Wop[index][0] * 1e6:.2f} µm',
        xy=(Wop[index][0] * 1e6, max * 1e-3),
        xytext=(30, -30),
        textcoords='offset points',
        fontsize=10,
        color='black',
        arrowprops=dict(arrowstyle='->', color='black'),
        bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
    )

    # Tick formatting
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()
    
    return index[0], max, Wop[index][0]

def R_vs_Wsp(Wop):
    Woi = Wos = np.linspace(1e-6, Wop, 10000)
    W_sp = Wos/Wop

    z = t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=0) * t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=0)
    z = abs(z) ** 2
    # Calculation of Pair Production Rate
    Numerator = ( eta_s * eta_i * (d**2) * (alpha_j(Woi)**2) * (alpha_j(Wos)**2) * (alpha_j(Wop)**2) * ws * wi)
    Denominator = np.sqrt(2) * np.sqrt(np.pi**3) * E0 * (c**3) * ns * ni * n_p * Bp
    integration_part = Integration_part(A(Wop, Wos, Woi), C(Wop, Wos, Woi), L, Kj(lambda_p,n_p), Kj(lambda_s,ns), Kj(lambda_i,ni), dky, dkz, Bp, wi, wos, woi, dws, dwi, Wop, Wos, Woi, theta_s, theta_i, z,n=0, m=0)

    R = (Numerator/Denominator) * integration_part

    plt.figure(figsize=(6, 4))  # Standard paper figure size

    # Plot the curve
    plt.plot(W_sp, R * 1e-3, color='blue', linewidth=1.5)

    # Axis labels and title with LaTeX formatting
    plt.title(r'Pair Rate per Pump Power vs $W_{o,s} / W_{o,p}$', fontsize=14)
    plt.xlabel(r'$W_{o,s} / W_{o,p}$', fontsize=12)
    plt.ylabel(r'$R/P$ (pairs/s·mW)', fontsize=12)

    # Grid for better readability
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Ticks formatting
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Layout adjustment for tight and clean output
    plt.tight_layout()
    plt.show()

def R_vs_P(Wop):
    Woi = Wos = Wp_Ws_relation(Ni,Ns,Np,Wop,L)

    z = t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=0) * t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=0)
    z = abs(z) ** 2

    # Calculation of Pair Production Rate
    Numerator = ( eta_s * eta_i * (d**2) * (alpha_j(Woi)**2) * (alpha_j(Wos)**2) * (alpha_j(Wop)**2) * ws * wi)
    Denominator = np.sqrt(2) * np.sqrt(np.pi**3) * E0 * (c**3) * ns * ni * n_p * Bp
    integration_part = Integration_part(A(Wop, Wos, Woi), C(Wop, Wos, Woi), L, Kj(lambda_p,n_p), Kj(lambda_s,ns), Kj(lambda_i,ni), dky, dkz, Bp, wi, wos, woi, dws, dwi, Wop, Wos, Woi, theta_s, theta_i, z,n=0, m=0)

    P = np.linspace(0.1e-3,10e-3,100000)

    R = (Numerator/Denominator) * integration_part * P

    plt.figure(figsize=(6, 4))  # Paper-friendly size

    # Plot curve
    plt.plot(P * 1e3, R, color='blue', linewidth=1.5)

    # Title and axes with LaTeX-style formatting
    plt.title(r'Pair Rate vs Pump Power', fontsize=14)
    plt.xlabel(r'$P$ (mW)', fontsize=12)
    plt.ylabel(r'$R$ (pairs/s)', fontsize=12)

    # Grid and ticks
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Layout
    plt.tight_layout()
    plt.show()

def R_vs_L(Wop):
    Ll = np.linspace(1e-3,10e-3,10000)
    Wos = Woi = Wop*np.sqrt(2)                                                                          


    I = 1j  # Imaginary unit in numpy
    D = (np.sin(2 * theta_s)**2 / Wos**2) - (np.sin(2 * theta_i)**2 / Woi**2)
    F = (np.sin(theta_s)**2 / Wos**2) + (np.sin(theta_i)**2 / Woi**2)
    H = F - (D**2/(4*C(Wop, Wos, Woi)))

    z = (np.pi*(erf((1j*dkz + H*Ll)/(2*np.sqrt(H))) - 1j*erfi((dkz + 1j*H*Ll)/(2*np.sqrt(H)))))/(2*np.sqrt(C(Wop, Wos, Woi))*np.exp((C(Wop, Wos, Woi)*dkz**2 + dky**2*H)/(4*C(Wop, Wos, Woi)*H))*np.sqrt(H))
    z *= np.sqrt(np.pi/ A(Wop, Wos, Woi))
    z = abs(z) ** 2

    Numerator = ( eta_s * eta_i * (d**2) * (alpha_j(Woi)**2) * (alpha_j(Wos)**2) * (alpha_j(Wop)**2) * ws * wi)
    Denominator = np.sqrt(2) * np.sqrt(np.pi**3) * E0 * (c**3) * ns * ni * n_p * Bp
    integration_part = Integration_part(A(Wop, Wos, Woi), C(Wop, Wos, Woi), Ll, Kj(lambda_p,n_p), Kj(lambda_s,ns), Kj(lambda_i,ni), dky, 1, Bp, wi, wos, woi, dws, dwi, Wop, Wos, Woi, theta_s, theta_i, z,n=0, m=0)

    R = (Numerator/Denominator) * integration_part

    plt.figure(figsize=(6, 4))  # Paper-sized figure

    # Plotting the curve
    plt.plot(Ll * 1e6, R * 1e-3, color='blue', linewidth=1.5)

    # Title and axes labels using LaTeX formatting
    plt.title(r'Pair Rate per Pump Power vs $L$', fontsize=14)
    plt.xlabel(r'$L$ (µm)', fontsize=12)
    plt.ylabel(r'$R/P$ (pairs/s·mW)', fontsize=12)

    # Grid and tick formatting
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Tight layout for publication
    plt.tight_layout()
    plt.show()

def Heralding_Eff(Wop):
    Woi = Wos = np.linspace(0.2*Wop, 1e7*Wop, 100) #Max may lie out of this range
    W_sp = Wos/Wop
   
    x_n = [t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=0),
           t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=1),
           t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=2),
           t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=3), 
           t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=4),
           t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=5),
           t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=6),
           t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=7), 
           t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=8),
           t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=9), 
           t2_int.int_dx(A(Wop, Wos, Woi), Wop, Wos, Woi, n=10)]
    
    y_n_s = [t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=0), 
           t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=1),
           t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=2),
           t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=3),
           t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=4),
           t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=5),
           t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=6),
           t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=7),
           t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=8),
           t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=9),
           t2_int.int_dydz_s(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=10)]
    
    y_n_i = [t2_int.int_dydz_i(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=0), 
           t2_int.int_dydz_i(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=1),
           t2_int.int_dydz_i(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=2),
           t2_int.int_dydz_i(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=3),
           t2_int.int_dydz_i(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=4),
           t2_int.int_dydz_i(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=5),
           t2_int.int_dydz_i(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=6),
           t2_int.int_dydz_i(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=7),
           t2_int.int_dydz_i(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=8),
           t2_int.int_dydz_i(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=9),
           t2_int.int_dydz_i(A(Wop, Wos, Woi),C(Wop, Wos, Woi),L,dky, dkz, theta_s, theta_i, Wop, Wos, Woi, m=10)]

    z = x_n[0] * y_n_s[0]
    z = abs(z) ** 2

    # Calculation of Pair Production Rate
    Numerator = ( eta_s * eta_i * (d**2) * (alpha_j(Woi)**2) * (alpha_j(Wos)**2) * (alpha_j(Wop)**2) * ws * wi)
    Denominator = np.sqrt(2) * np.sqrt(np.pi**3) * E0 * (c**3) * ns * ni * n_p * Bp
    integration_part = Integration_part(A(Wop, Wos, Woi), C(Wop, Wos, Woi), L, Kj(lambda_p,n_p), Kj(lambda_s,ns), Kj(lambda_i,ni), dky, dkz, Bp, wi, wos, woi, dws, dwi, Wop, Wos, Woi, theta_s, theta_i, z, n=0, m=0)

    R = (Numerator/Denominator) * integration_part

    Rs = 0
    Ri = 0
    n_max = 10
    m_max = 3
    for n in range(n_max+1):
        for m in range(m_max+1):
        # Calculation of Pair Production Rate
            z_s = x_n[n] * y_n_s[m]
            z_s = abs(z_s) ** 2

            z_i = x_n[n] * y_n_i[m]
            z_i = abs(z_i) ** 2

            Numerator = ( eta_s * eta_i * (d**2) * (alpha_j(Woi,n,m)**2) * (alpha_j(Wos)**2) * (alpha_j(Wop)**2) * ws * wi)
            Denominator = np.sqrt(2) * np.sqrt(np.pi**3) * E0 * (c**3) * ns * ni * n_p * Bp
            integration_part = Integration_part(A(Wop, Wos, Woi), C(Wop, Wos, Woi), L, Kj(lambda_p,n_p), Kj(lambda_s,ns), Kj(lambda_i,ni), dky, dkz, Bp, wi, wos, woi, dws, dwi, Wop, Wos, Woi, theta_s, theta_i, z_i, n=0, m=0)
            M = (Numerator/Denominator) * integration_part
            Rs += M

            Numerator = ( eta_s * eta_i * (d**2) * (alpha_j(Woi)**2) * (alpha_j(Wos,n,m)**2) * (alpha_j(Wop)**2) * ws * wi)
            Denominator = np.sqrt(2) * np.sqrt(np.pi**3) * E0 * (c**3) * ns * ni * n_p * Bp
            integration_part = Integration_part(A(Wop, Wos, Woi), C(Wop, Wos, Woi), L, Kj(lambda_p,n_p), Kj(lambda_s,ns), Kj(lambda_i,ni), dky, dkz, Bp, wi, wos, woi, dws, dwi, Wop, Wos, Woi, theta_s, theta_i, z_s, n=0, m=0)
            Ri += (Numerator/Denominator) * integration_part

            print(n,m)

    HE = (R / np.sqrt(Rs * Ri))
    max_index = np.argmax(HE)
    HE_max = HE[max_index]
    print(HE_max)
    

    # Plot HE vs Wos/Wop
    plt.figure(figsize=(6, 4))  # Suitable size for publication

    # Plot the curve
    plt.plot(W_sp, HE, color='green', linewidth=1.5)

    # Title and axis labels with LaTeX-style formatting
    plt.title(r'Heralding Efficiency vs $W_{o,s} / W_{o,p}$', fontsize=14)
    plt.xlabel(r'$W_{o,s} / W_{o,p}$', fontsize=12)
    plt.ylabel('Heralding Efficiency (HE)', fontsize=12)

    # Set y-axis limits to [0, 1]
    # plt.ylim(0.00, 1.00)

    # Grid and tick styling
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Adjust layout to avoid label clipping
    plt.tight_layout()
    plt.show()

index, max_R, Wop_max = Calc_Max_R()
# R_vs_Wsp(Wop_max)
# R_vs_P(Wop_max)
# R_vs_L(Wop_max)
# Heralding_Eff(Wop_max)