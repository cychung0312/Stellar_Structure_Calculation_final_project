import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.interpolate import RectBivariateSpline
import pandas as pd
import matplotlib.pyplot as plt
def calculate_stellar_density(P, T, X, Y, Z):
    """
    Calculates density (rho) for a fully ionized plasma including radiation pressure.
    Units: P (dyn/cm^2), T (K), rho (g/cm^3)
    """
    # Constants (CGS units)
    k = 1.380649e-16      # Boltzmann constant (erg/K)
    m_H = 1.6735e-24      # Hydrogen atom mass (g)
    a = 7.5657e-15        # Radiation constant (erg/cm^3/K^4)
    
    # 1. Calculate Mean Molecular Weight (mu) for complete ionization
    inv_mu = 2.0*X + 0.75*Y + 0.5*Z
    mu = 1.0 / inv_mu
    
    # 2. Calculate Radiation Pressure
    P_rad = (1.0/3.0) * a * T**4
    
    P_gas = np.maximum(P - P_rad, 1e-5 * P) 
    beta = P_gas / P
    
    # 4. Calculate Density from Gas Pressure: P_gas = (rho * k * T) / (mu * m_H)
    rho = (P_gas * mu * m_H) / (k * T)
    
    return rho, beta


def load_opacity_table(filename):
    """
    Parses opacity table (derivative-inclusive format)
    Returns: logT, logR, logKappa (as 1D arrays)
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            # Only process data rows (starts with a number)
            # A more robust check for numeric starting lines
            if len(parts) >= 4 and parts[0][0].isdigit() or (parts[0].startswith('-') and parts[0][1].isdigit()):
                try:
                    # Column mapping: 0:logT, 1:logR, 2:logRho, 3:logKappa
                    row = [float(parts[0]), float(parts[1]), float(parts[3])]
                    data.append(row)
                except ValueError:
                    continue
    data = np.array(data)
    return data[:, 0], data[:, 1], data[:, 2]

def create_interpolator(logT_d, logR_d, logK_d):
    # Determine the unique grid points
    uT = np.unique(logT_d)
    uR = np.unique(logR_d)
    
    # Reshape the 1D opacity data into a 2D grid
    K_grid = logK_d.reshape(len(uT), len(uR))
    
    # Create the spline object (kx=1, ky=1 is linear and very stable)
    return RectBivariateSpline(uT, uR, K_grid, kx=1, ky=1), uT, uR


def get_interpolated_kappa(target_logRho, target_logT):
    target_logT = np.clip(target_logT, T_grid.min(), T_grid.max())
    # Calculate logR = logRho - 3*logT + 18
    logR = target_logRho - 3 * (target_logT - 6.0)
    target_logR = np.clip(logR, R_grid.min(), R_grid.max())

    return 10**kappa_spline.ev(target_logT, target_logR)
    

def calculate_epsilon(rho, T, X, Y, Z):
    """
    Calculates nuclear energy generation rate (erg/g/s) for H-burning.
    """
    T7 = T / 1e7
    T9 = T / 1e9
    
    # 1. Weak Screening Factor (f11) for pp-chain
    # KWW Eq 18.56, 18.57 approximation
    # For fully ionized gas
    z1 = z2 = 1.0 
    zeta = 2.0 * X + 1.5 * Y + Z 
    mu_e = 2 / (1+X)
    exponent = 5.92e-3 * z1 * z2 * np.sqrt(zeta * rho / (T7**3))
    if exponent >= 1:
        exponent = 0.0205*((z1+z2)**(5/3)-z1**(5/3)-z2**(5/3))*((rho/mu_e)**(1/3)/T7)
    f11 = np.exp(exponent)
    
    # 2. pp-Chain Energy Generation (KWW Eq 18.63)
    psi = 1.0       # branching factor (approx 1 for T < 2e7 K).
    g11 = (1 + 3.82*T9 + 1.51*T9**2 + 0.144*T9**3 - 0.0114*T9**4)
    eps_pp = 2.57e4 * psi * f11 * g11 * rho * X**2 * (T9**(-2/3)) * np.exp(-3.381 / (T9**(1/3)))

    # 3. CNO Cycle Energy Generation (KWW Eq 18.65)
    # Assumes CN and NO cycles are in equilibrium (X_CNO approx Z/2)
    X_CNO = Z  
    g14_1 = (1 - 2.00*T9 + 3.41*T9**2 - 2.43*T9**3)
    eps_cno = 8.24e25 * g14_1 * rho * X * X_CNO * (T9**(-2/3)) * np.exp((-15.231 * (T9**(-1/3))) - (T9/0.8)**2 )    
    return eps_pp, eps_cno

def load1(m, Pc, Tc, X, Y, Z):
    # 1. State variables at center
    rho_c, beta_c = calculate_stellar_density(Pc, Tc, X, Y, Z)
    kappa_c = get_interpolated_kappa(np.log10(rho_c), np.log10(Tc))
    eps_pp, eps_cno = calculate_epsilon(rho_c, Tc, X, Y, Z)
    epsilon_c = eps_pp + eps_cno

    # 2. Geometry and Luminosity
    r = (3.0 * m / (4.0 * np.pi * rho_c))**(1/3)
    l = epsilon_c * m

    # 3. Pressure drop
    G = 6.67430e-8
    P = Pc - (3.0 * G / (8.0 * np.pi)) * (4.0 * np.pi * rho_c / 3.0)**(4/3) * m**(2/3)

    # Determine if core is convective at m=0
    a_rad = 7.5657e-15
    c_light = 2.9979e10
    G = 6.67430e-8
    
    # Calculate nabla_rad at center using the limit l/m as m->0 = epsilon_c
    nabla_rad = (3.0 * kappa_c * epsilon_c * Pc) / (16.0 * np.pi * a_rad * c_light * G * Tc**4)
    nabla_ad = 0.4
    
    # 4. Temperature drop
    if nabla_rad > nabla_ad:
        # --- CONVECTIVE CORE ---
        term_conv = (np.pi / 6.0)**(1/3) * G * (nabla_ad * rho_c**(4/3) / Pc) * m**(2/3)
        T = np.exp(np.log(Tc) - term_conv)

    else:
        # --- RADIATIVE CORE ---
        term = (1.0 / (2.0 * a_rad * c_light)) * (3.0 / (4.0 * np.pi))**(2/3) * kappa_c * epsilon_c * (rho_c**(4/3)) * (m**(2/3))
        T = (Tc**4 - term)**0.25

    return np.array([P, T, r, l])



def load2(L_star, R_star, M_star, X, Y, Z):

    G = 6.67430e-8
    sigma_sb = 5.67037e-5
    m_H = 1.6735e-24
    k = 1.380649e-16      # Boltzmann constant (erg/K)
    

    # 1. Luminosity is constant in the envelope
    l = L_star
    r = R_star

    # 2. Temperature at tau = 2/3 is Teff (per Eddington approx)
    T_eff = (L_star / (4.0 * np.pi * (R_star**2) * sigma_sb))**0.25
    T = T_eff

    # Simple fixed-point iteration to find consistent P and kappa
    g = (G * M_star) / (R_star**2)
    P_surf = 1.0e5 # Initial guess
    for i in range(15):
        inv_mu = 1.0 * X + 0.25 * Y # Approximation for neutral gas
        mu = 1.0 / inv_mu
        rho = (P_surf * mu * m_H) / (k * T)
        kappa = get_interpolated_kappa(np.log10(rho), np.log10(T))
        
        # Eq 11.13: P = (g * tau) / kappa, with tau = 2/3
        new_P = (2.0/3.0) * (g / kappa)

        if np.abs((new_P - P_surf) / P_surf) < 1e-8:
            break
        P_surf = new_P

    P = P_surf 
    return np.array([P, T, r, l])

def derivs(m, y, X, Y, Z):
    """
    Stellar structure derivatives for use with shootf/solve_ivp.
    y = [P, T, r, l]
    Returns [dl/dm, dP/dm, dr/dm, dT/dm]
    """
    P, T, r, l = y
    
    # Constants (CGS)
    G = 6.67430e-8
    a_rad = 7.5657e-15
    c_light = 2.9979e10
    
    # 1. Calculate local physical properties
    rho, beta = calculate_stellar_density(P, T, X, Y, Z)
    kappa = get_interpolated_kappa(np.log10(rho), np.log10(T))
    eps_pp, eps_cno = calculate_epsilon(rho, T, X, Y, Z)
    epsilon = eps_pp + eps_cno
    
    # 2. Conservation of Mass and Hydrostatic Equilibrium
    # dr/dm = 1 / (4 * pi * r^2 * rho)
    dr_dm = 1.0 / (4.0 * np.pi * (r**2) * rho)
    
    # dP/dm = -G * m / (4 * pi * r^4)
    dP_dm = -(G * m) / (4.0 * np.pi * (r**4))
    
    # 3. Energy Generation
    # dl/dm = epsilon
    dl_dm = epsilon
    
    # 4. Energy Transport (Stability Check)
    # Radiative gradient
    nabla_rad = (3.0 * kappa * l * P) / (16.0 * np.pi * a_rad * c_light * G * m * T**4)

    # Adiabatic gradient (Assumes fully ionized ideal gas)
    nabla_ad = 0.4 
    
    # Schwarzschild Criterion
    nabla = np.minimum(nabla_rad, nabla_ad)
    dT_dm = -(G * m * T) / (4.0 * np.pi * (r**4) * P) * nabla

    return np.array([dP_dm, dT_dm, dr_dm, dl_dm])


def shootf(vec, M_star, X, Y, Z, m_fit_frac=0.6):
    """
    Computes the difference between core and surface integrations at m_fit.
    vec = [log(Pc), log(Tc), log(R_star), log(L_star)]
    """
#    vec = np.clip(vec, 
#                  [10.0, 5.0, 9.0, 30.0],   # Lower bounds (M-dwarfs)
#                  [20.0, 8.5, 13.0, 40.0])
    
    # 1. Unpack and delog guesses (using logs helps Newton-Raphson stability)
    Pc, Tc = 10**vec[0], 10**vec[1]
    R_star, L_star = 10**vec[2], 10**vec[3]

    m_fit = m_fit_frac * M_star
    m_eps = 1e-10 * M_star # Small offset from center to avoid 1/r singularity

    try:
        # 2. Outward Integration (Center -> Fitting Point)
        y_core_start = load1(m_eps, Pc, Tc, X, Y, Z)
        sol_core = solve_ivp(derivs, (m_eps, m_fit), y_core_start, 
                         args=(X, Y, Z), 
                         method='Radau', rtol=1e-10)
    
        # 3. Inward Integration (Surface -> Fitting Point)
        y_surf_start = load2(L_star, R_star, M_star, X, Y, Z)

        sol_surf = solve_ivp(derivs, (M_star, m_fit), y_surf_start, 
                         args=(X, Y, Z), 
                         method='Radau', rtol=1e-10)
    
        print(f"DEBUG: P_surf = {y_surf_start[0]:.2e} | P_core = {y_core_start[0]:.2e}")
        print(f"DEBUG: T_surf = {y_surf_start[1]:.2e} | T_core = {y_core_start[1]:.2e}")

        print(f"DEBUG: P_surf_at_fit = {sol_surf.y[0,-1]:.2e} | P_core_at_fit = {sol_core.y[0,-1]:.2e}")
        print(f"DEBUG: T_surf_at_fit = {sol_surf.y[1,-1]:.2e} | T_core_at_fit = {sol_core.y[1,-1]:.2e}")

    
        # 4. Check for failures
        if not sol_core.success or not sol_surf.success:
            # Return a massive penalty to tell the solver "wrong direction"
            return np.array([1e15, 1e15, 1e15, 1e15])
    
        # 5. Calculate Residuals [P, T, r, l]
        # We normalize by the surface guesses to make the error dimensionless
        res = (sol_core.y[:, -1] - sol_surf.y[:, -1]) / sol_core.y[:, -1]
        res = np.zeros(4)
        res[0] = (sol_core.y[0, -1] - sol_surf.y[0, -1]) / sol_core.y[0, -1] # Pressure
        res[1] = (sol_core.y[1, -1] - sol_surf.y[1, -1]) / sol_core.y[1, -1] # Temperature
        res[2] = (sol_core.y[2, -1] - sol_surf.y[2, -1]) / R_sun
        res[3] = (sol_core.y[3, -1] - sol_surf.y[3, -1]) / L_sun
        if np.isnan(res).any():
            return np.array([1e15, 1e15, 1e15, 1e15])
        else: 
            return res
        
    except Exception:
        return np.array([1e15] * 4)

def run_solver(M_star, init_guess, X, Y, Z):

    # initial guess: [log_Pc, log_Tc, log_R_star, log_L_star]
       
    method_choice = 'hybr' if M_star < 0.7 else 'lm'
    opts = {'xtol': 1e-10} if method_choice == 'hybr' else {'ftol': 1e-10, 'xtol': 1e-10}
    
    result = root(shootf, init_guess, args=(M_star, X, Y, Z), 
                  method=method_choice, 
                  options=opts)
        
    if result.success:
        print("Convergence Successful!")
        return 10**result.x
    else:
        print(f"Convergence Failed: {result.message}")
        return None
    

# Constants for conversion
M_sun = 1.989e33
L_sun = 3.828e33
R_sun = 6.957e10

# Initial guesses
def get_homology_guess(M_star):
    m_ratio = M_star / M_sun
    
    # Scale based on 1.0 M_sun success values
    Pc = 2.47e17 * (m_ratio**(-2))
    Tc = 1.57e7  * (m_ratio**0.5)
    R  = 0.8 * R_sun * (m_ratio**0.7)
    L  = 0.8 * L_sun * (m_ratio**3.5) # Mass-Luminosity relation
    
    return np.log10(np.array([Pc, Tc, R, L]))

# Use the scaled guess
M_star = 1.0 * M_sun
initial_guess = get_homology_guess(M_star)

X = 0.70
Y = 0.28
Z = 0.02

logT_d, logR_d, logK_d = load_opacity_table('popI_op.txt')
kappa_spline, T_grid, R_grid = create_interpolator(logT_d, logR_d, logK_d)
final_params = run_solver(M_star, initial_guess, X, Y, Z)

if final_params is not None:
    import pandas as pd

if final_params is not None:
    Pc_f, Tc_f, R_f, L_f = final_params
    m_fit = 0.5 * M_star
    m_eps = 1e-10 * M_star

    # 1. Perform a final, high-resolution integration (dense_output for smooth arrays)
    y_c_start = load1(m_eps, Pc_f, Tc_f, X, Y, Z)
    sol_c = solve_ivp(derivs, (m_eps, m_fit), y_c_start, args=(X, Y, Z), method='Radau', rtol=1e-10)
    
    y_s_start = load2(L_f, R_f, M_star, X, Y, Z)
    sol_s = solve_ivp(derivs, (M_star, m_fit), y_s_start, args=(X, Y, Z), method='Radau', rtol=1e-10)

    # 2. Merge and reverse surface integration (it went M_star -> m_fit)
    m_full = np.concatenate([sol_c.t, sol_s.t[::-1]])
    P_full = np.concatenate([sol_c.y[0], sol_s.y[0][::-1]])
    T_full = np.concatenate([sol_c.y[1], sol_s.y[1][::-1]])
    r_full = np.concatenate([sol_c.y[2], sol_s.y[2][::-1]])
    l_full = np.concatenate([sol_c.y[3], sol_s.y[3][::-1]])

    # 3. Reconstruct secondary variables
    table_data = []
    
    for i in range(len(m_full)):
        m, P, T, r, l = m_full[i], P_full[i], T_full[i], r_full[i], l_full[i]
        
        # Physics evaluations
        rho, beta = calculate_stellar_density(P, T, X, Y, Z)
        kappa = get_interpolated_kappa(np.log10(rho), np.log10(T))
        eps_pp, eps_cno = calculate_epsilon(rho, T, X, Y, Z)
        epsilon = eps_pp + eps_cno
        
        # Calculate gradients (nabla)
        a_rad = 7.5657e-15
        c_light = 2.9979e10
        G = 6.67430e-8
        
        # Avoid division by zero at the center for nabla_rad
#        if m < 1e-5 * M_star:
#            nabla_rad = (3.0 * kappa * epsilon * P) / (16.0 * np.pi * a_rad * c_light * G * T**4)
#        else:
        nabla_rad = (3.0 * kappa * l * P) / (16.0 * np.pi * a_rad * c_light * G * m * T**4)
        
        nabla_ad = 0.4
        nabla_act = min(nabla_rad, nabla_ad)
        nature = "CONV" if nabla_rad > nabla_ad else "RAD"

        table_data.append([m, r, rho, T, P, l, epsilon, kappa, nabla_ad, nabla_rad, nabla_act, nature])

    # 4. Save to CSV
    cols = ['m', 'r', 'rho', 'T', 'P', 'l', 'epsilon', 'kappa', 'grad_ad', 'grad_rad', 'grad_act', 'nature']
    df = pd.DataFrame(table_data, columns=cols)
    filename = f"stellar_profile_{M_star/M_sun:.1f}Msun.csv"
    df.to_csv(filename, index=False)
    
    print(f"Machine-readable table saved as {filename}")

    sigma_sb = 5.67037e-5
    T_eff = (L_f / (4.0 * np.pi * (R_f**2) * sigma_sb))**0.25

    print("\n" + "="*45)
    print("      FINAL CONVERGED STELLAR MODEL")
    print("="*45)
    print(f"{'Parameter':<25} {'Value':<15} {'Units':<10}")
    print("-" * 45)
    print(f"{'Central Pressure (Pc)':<25} {Pc_f:<15.4e} dyn/cm^2")
    print(f"{'Central Temperature (Tc)':<25} {Tc_f:<15.4e} K")
    print(f"{'Total Radius (R*)':<25} {R_f / R_sun:<15.4f} R_sun")
    print(f"{'Total Luminosity (L*)':<25} {L_f / L_sun:<15.4f} L_sun")
    print(f"{'Effective Temp (Teff)':<25} {T_eff:<15.1f} K")
    print("-" * 45)
    print(f"Log10(Tc): {np.log10(Tc_f):.4f} | Log10(Pc): {np.log10(Pc_f):.4f}")
    print("="*45 + "\n")
else:
    print("\n[!] Convergence failed. No results to display.")

