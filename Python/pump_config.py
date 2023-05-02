import numpy as np

threshold = 7.06 # in ampere
slope = 0.41 # in watt per ampere

def power_curve(cur):
    return (cur - threshold) * slope

currents = np.arange(12,33, 4)
wattage = power_curve(currents)

print(currents)
print(wattage)
print(power_curve(46))
