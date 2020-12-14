# The heat index is what you commonly see on a weather display as the 
# "Feels Like" temperature, it is the result of a linear regression
# by Rothfuz, https://www.wpc.ncep.noaa.gov/html/heatindex.shtml

import numpy as np

print("Temperature:")
Temp = float(input())
print("Temerature Units (C/F):")
Units = input()
print("Humidity %:")
RH = float(input())


def FeelsLike(Temp, Units, RH):
    HI = []
    if Units == "F":
        T = Temp
    if Units == "C":
        T = (Temp * 9/5) + 32 # convert C to F for calc

    HI = (0.50 * (T + 61 + ((T - 68) * 1.20) + (RH * 0.094))) # simple HI is calculated before the full regression,
    mean_HI = np.mean([HI, T])                                # eq, if the average of HI and Temp are below 80F
    if mean_HI > 80:                                          # then the result is accurate enough
        HI = -42.379 + 2.04901523*T + 10.14333127*RH\
            - .22475541*T*RH - .00683783*T*T\
                - .05481717*RH*RH + .00122874*T*T*RH\
                    + .00085282*T*RH*RH - .00000199*T*T*RH*RH
    elif RH <= 13 and 80 <= T and T <= 120:
            HI = HI - ((13 - RH)/4)*np.sqrt((17- abs(T - 95)) / 17)
    elif RH >= 85 and 80 <= T and T <= 87:
            HI = HI + ((RH - 85) / 10) * ((87 - T) / 5)

    if Units == "F":
        return HI
    if Units == "C":
        HI = (HI - 32) * 5/9 # convert back to celsius if necessary
        return HI

print("Feels Like", "{:.2f}".format(FeelsLike(Temp, Units, RH)), "degrees")
