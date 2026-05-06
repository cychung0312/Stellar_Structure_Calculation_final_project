import mesa_reader as mr
import matplotlib.pyplot as plt
import numpy as np

# Load the profile

p = mr.MesaData('profile2.data')
print("Available columns in your file:")
print(p.bulk_names)

if hasattr(p, 'header_names'):
    print(f"{'Header Item':25} | {'Value'}")
    print("-" * 40)
    for key in p.header_names:
        # Pass the key to the header function
        print(f"{key:25} | {p.header(key)}")
else:
    # If header_names isn't there, this will show you all available attributes
    print("Could not find 'header_names'. Attributes on this object are:")
    print(dir(p))

# 1. Central Values
# In readmesa, data is often stored in a dictionary-like structure
# Central values are at the end of the arrays (index -1)
tc_mesa = 10**p.logT[-1]
rhoc_mesa = 10**p.logRho[-1]
pc_mesa = 10**p.logP[-1]

# 2. Global Parameters (from the header)
# readmesa stores header info in the .header attribute
total_l = p.header('photosphere_L')
total_r = p.header('photosphere_r')
teff_mesa = p.header('Teff')

print(f"--- MESA Results (via readmesa) ---")
print(f"Central Temperature (Tc): {tc_mesa:.4e} K")
print(f"Central Density (rhoc):   {rhoc_mesa:.4f} g/cm^3")
print(f"Central Pressure (Pc):    {pc_mesa:.4e} dyn/cm^2")
print(f"Total Luminosity (L):     {total_l:.4f} Lsun")
print(f"Total Radius (R):         {total_r:.4f} Rsun")
print(f"Effective Temp (Teff):    {teff_mesa:.2f} K")

# 3. Quick Comparison 
my_tc = 2.0303e+07
my_pc = 1.5264e+17
my_rhoc = 56.09311
my_Teff = 8753.9
my_R = 1.6889
my_L = 15.0896

print(f"\n--- Comparison Residuals ---")
print(f"Tc Difference: {abs(tc_mesa - my_tc)/tc_mesa * 100:.2f}%")
print(f"Pc Difference: {abs(pc_mesa - my_pc)/pc_mesa * 100:.2f}%")
print(f"rhoc Difference: {abs(rhoc_mesa - my_rhoc)/rhoc_mesa * 100:.2f}%")
print(f"Teff Difference: {abs(teff_mesa - my_Teff)/teff_mesa * 100:.2f}%")
print(f"R Difference: {abs(total_r - my_R)/total_r * 100:.2f}%")
print(f"L Difference: {abs(total_l - my_L)/total_l* 100:.2f}%")
