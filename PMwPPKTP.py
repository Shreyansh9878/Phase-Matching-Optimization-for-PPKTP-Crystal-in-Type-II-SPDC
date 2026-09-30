import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import sys

sys.stdout.reconfigure(encoding="utf-8")

L = 2e-3
lambda_p = 405e-9
poling_period = 10e-6
theta_s = theta_i = np.deg2rad(0)
c = 299792458

temp_fit_coeff_n1 = pd.DataFrame({
    'z': [9.9587e-6, 9.9228e-6, -8.9603e-6, 4.1010e-6],
    'y': [6.2897e-6, 6.3061e-6, -6.0629e-6, 2.6486e-6]
})

temp_fit_coeff_n2 = pd.DataFrame({
    'z': [-1.1882e-8, 10.459e-8, -9.8136e-8, 3.1481e-8],
    'y': [-0.14445e-8, 2.2244e-8, -3.5770e-8, 1.3470e-8]
})

fit_coeff_list = [temp_fit_coeff_n1, temp_fit_coeff_n2]

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

def factorial(x):
    if x==0:
        return 1
    else:
        return x * factorial(x-1)

def Kj(lj, nj):
    return (nj * 2 * np.pi) / lj

def A(Wp, Ws, Wi): #Defiend constant A
    return (1 / Wp ** 2) + (1 / Ws ** 2) + (1 / Wi ** 2)

def C(Wp, Ws, Wi): #Defiend constant C
    return (1 / Wp ** 2) + (np.cos(theta_s)**2 / Ws ** 2) + (np.cos(theta_i)**2 / Wi ** 2)

def alpha_j(Woj,n=0,m=0):
    Denominator = ((2 ** (n + m)) * factorial(n) * factorial(m) * np.pi * (Woj ** 2))
    return np.sqrt(2 / Denominator)

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

def dky_zero(lp, temp, angle_s = 0):
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
    temp_solution = fsolve(lambda temp: dky_zero(lp, temp, theta_s), initial_guess)[0]

    print("Temperature where Δk_y = 0:", temp_solution, "°C")

    # Plot Δk_y vs Temperature
    temp_range = np.linspace(0, 150, 1000)
    dky_values = [dky_zero(lp, T) for T in temp_range]

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
    center_freq = c / lam_p
    vp = np.linspace(center_freq-df, center_freq + df, 1000)
    lp = c/vp

    L_corr = L*(1 + (6.7e-6)*(temp-25) + (11e-9)*(temp-25)**2)

    dk_values = []
    for lps in lp:
        dk_values.append(dky_zero(lps, temp))
    dk_values = np.array(dk_values)

    x = (dk_values*L_corr)/2
    sinc_term = np.abs(np.sin(x)/x)**2

    jsa = Integration_part(A(Wop, Wos, Woi), C(Wop, Wos, Woi), L, Kj(lambda_p,n_p), Kj(lambda_s,ns), Kj(lambda_i,ni), dky, dkz, Bp, wi, wos, woi, dws, dwi, Wop, Wos, Woi, theta_s, theta_i, z,n=0, m=0)


    half_max=0.5
    crossings = np.where(np.diff(np.sign(sinc_term - half_max)))[0]

    if len(crossings) >= 2:
        # Interpolate to find more accurate crossing points
        f_interp_left = interp1d(sinc_term[crossings[0]:crossings[0] + 2], vp[crossings[0]:crossings[0] + 2])
        v_left = f_interp_left(half_max)

        f_interp_right = interp1d(sinc_term[crossings[1]:crossings[1] + 2], vp[crossings[1]:crossings[1] + 2])
        v_right = f_interp_right(half_max)

        FWHM_v = (v_right - v_left) * 1e-9  # GHz
        FWHM_l = ((c / v_left) - (c / v_right)) * 1e9  # nm

        fwhm_label = f"FWHM: {FWHM_v:.3f} GHz OR {FWHM_l:.3f} nm"
        print(fwhm_label)

        center_freq = (v_left + v_right) / 2
        FWHM_freq = v_right - v_left

        fig, ax1 = plt.subplots()

        # Plot full curve
        ax1.plot(vp, sinc_term, label='Sinc term')

        # Highlight FWHM portion of the curve
        fwhm_mask = (vp >= v_left) & (vp <= v_right)
        ax1.plot(vp[fwhm_mask], sinc_term[fwhm_mask], color='blue', linewidth=2.5, label='FWHM region')

        ax1.set_xlabel('vp (Frequency in Hz)')
        ax1.set_ylabel('sinc_term')
        ax1.set_title(f'Sinc Term vs vp at Temp = {temp}°C')
        ax1.grid(True)

        # Mark FWHM region on frequency axis (bottom)
        ax1.axvline(v_left, color='red', linestyle='--', label='FWHM freq start')
        ax1.axvline(v_right, color='red', linestyle='--', label='FWHM freq end')
        ax1.fill_betweenx([0, 1], v_left, v_right, color='red', alpha=0.2)

        # Create a second x-axis on top for wavelength
        ax2 = ax1.twiny()
        ax2.set_xlim(lp[0], lp[-1])

        freq_ticks = ax1.get_xticks()
        wavelength_ticks = np.interp(freq_ticks, vp, lp)
        ax2.set_xticks(wavelength_ticks)
        ax2.set_xticklabels([f"{wl*1e9:.1f}" for wl in wavelength_ticks])
        ax2.set_xlabel('Wavelength (nm)')

        # Mark FWHM region on wavelength axis (top)
        left_wavelength = c / v_left
        right_wavelength = c / v_right
        ax2.axvline(left_wavelength, color='green', linestyle='--', label='FWHM wavelength start')
        ax2.axvline(right_wavelength, color='green', linestyle='--', label='FWHM wavelength end')

        ax1.plot([], [], ' ', label=fwhm_label)
        ax1.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

    else:
        print("Could not find two crossings at half maximum.")

def Angle_Temp(lp):
    angle_s = np.linspace(0, np.deg2rad(3), 7)

    temp_solutions = list()

    for angle in angle_s:
        initial_guess = 80
        temp_solution = fsolve(lambda temp: dky_zero(lp, temp, angle), initial_guess)[0]
        temp_solutions.append(temp_solution)

    print(temp_solutions)
    angle_s = np.rad2deg(angle_s)
    
    theta_vals = np.linspace(-3.2, 3.2, 300)
    X, Y = np.meshgrid(theta_vals, theta_vals)
    r = np.sqrt(X**2 + Y**2)

    # Plot setup
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw each circular contour
    for angle, temp, color in zip(angle_s, temp_solutions,
                               ['white','purple', 'blue', 'green', 'orange', 'red', 'black']):
        if angle == 0:
            ax.plot(0, 0, 'o', color=color)  # Mark it
            ax.text(0, 0.12, f"{temp:.1f} °C", fontsize=10, fontweight='bold', ha='center', va='bottom', color = 'red')
        else:
            cs = ax.contour(X, Y, r, levels=[angle], colors=[color], linewidths=2)
            labels = ax.clabel(cs, fmt={angle: f"{temp:.1f} °C"}, fontsize=10)
            for txt in labels:
                txt.set_fontweight('bold')

        # Instead of using cs.collections (which can fail), add dummy line to legend
        ax.plot([], [], color=color, linewidth=2, label=f"{temp:.1f} °C")

    # Axes formatting
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect('equal')
    ax.set_xlabel("Angle θy (°)", fontsize=12)
    ax.set_ylabel("Angle θz (°)", fontsize=12)

    # Temperature direction arrow
    ax.annotate('T', xy=(2.0, 2.1), xytext=(2.7, 2.8),
                arrowprops=dict(arrowstyle='->'), fontsize=12)

    # ax.legend(title="Temperature", fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.show()

temp_sol = Phase_Match_Temp(lambda_p)
Bandwidth(temp_sol, lambda_p, 10e12)
Angle_Temp(lambda_p)