"""Available XC functionals.

LYNX supports LDA, GGA, mGGA, and hybrid functionals via libxc.
"""
import lynx

# List available functionals
functionals = [
    ("LDA",    ["LDA_PZ", "LDA_PW"]),
    ("GGA",    ["PBE", "PBEsol", "RPBE"]),
    ("mGGA",   ["SCAN", "RSCAN", "R2SCAN"]),
    ("Hybrid", ["PBE0", "HSE06"]),
]

print("Available XC functionals:")
for family, names in functionals:
    print(f"\n  {family}:")
    for name in names:
        xc = lynx.xc.get(name)
        flags = []
        if xc.is_gga: flags.append("GGA")
        if xc.is_mgga: flags.append("mGGA")
        if xc.is_hybrid: flags.append(f"hybrid(alpha={xc.exx_fraction})")
        print(f"    {name:10s} -> {xc}  [{', '.join(flags) or 'LDA'}]")

# Use string shorthand or explicit object
calc1 = lynx.DFT(xc="PBE")
calc2 = lynx.DFT(xc=lynx.xc.HSE06(alpha=0.25, omega=0.11))
print(f"\ncalc1.xc = {calc1.xc}")
print(f"calc2.xc = {calc2.xc}")
