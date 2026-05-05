import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
M_sun = 1.989e33
L_sun = 3.828e33
R_sun = 6.957e10

M_star = 1.0 * M_sun
filename = f"stellar_profile_{M_star/M_sun:.1f}Msun.csv"
df = pd.read_csv(filename)
r_norm = df['r'] / df['r'].max()

# --- Figure 1: Global Structure ---
plt.figure(figsize=(8, 6))
plt.plot(r_norm, df['P']/df['P'].max(), label='Pressure ($P/P_c$)')
plt.plot(r_norm, df['T']/df['T'].max(), label='Temperature ($T/T_c$)')
plt.plot(r_norm, df['m']/df['m'].max(), label='Mass ($m/M_*$)')
plt.plot(r_norm, df['l']/df['l'].max(), label='Luminosity ($l/L_*$)')
plt.xlabel('$r/R_*$')
plt.ylabel('Normalized Value')
plt.title('Internal Structure Profile')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'structure_profile_{M_star/M_sun:.1f}Msun.png')

# --- Figure 2: Energy & Opacity ---
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()
# Use r'...' for raw strings
ax1.plot(df['m']/df['m'].max(), df['epsilon'], 'r-', label=r'$\epsilon$ (Energy)')
ax2.plot(r_norm, np.log10(df['kappa']), 'b--', label=r'$\log_{10}(\kappa)$')
ax1.set_ylabel(r'$\epsilon$ (erg/g/s)', color='r')
ax2.set_ylabel(r'$\log_{10}(\kappa)$', color='b')
ax1.set_xlabel('$m/M_*$')
plt.title('Energy Generation and Opacity')
plt.savefig(f'energy_opacity_{M_star/M_sun:.1f}Msun.png')

# --- Figure 3: Stability ---
plt.figure(figsize=(8, 6))
plt.plot(r_norm, df['grad_rad'], 'r-', label=r'$\nabla_{rad}$')
plt.plot(r_norm, df['grad_ad'], 'b--', label=r'$\nabla_{ad}$')
plt.fill_between(r_norm, df['grad_rad'], df['grad_ad'], 
                 where=(df['grad_rad'] > df['grad_ad']), 
                 color='gray', alpha=0.3, label='Convective Zone')
plt.ylim(0, 1) # Focus on the physical range
plt.xlabel('$r/R_*$')
plt.ylabel(r'$\nabla$')
plt.title('Schwarzschild Stability Criterion')
plt.legend()
plt.savefig(f'stability_gradient_{M_star/M_sun:.1f}Msun.png')
