import pandas as pd
import numpy as np
import os
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt

def spectral_analysis(df, column, plot=True):
    y = df[column].values
    times = pd.to_datetime(df["datetime"])
    ts = (times - times.iloc[0]).dt.total_seconds().values / 3600

    ls = LombScargle(ts, y)
    frequency, power = LombScargle(ts, y).autopower(minimum_frequency = 1 / 72, maximum_frequency=1)  # e.g. min period: 4h)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(1/frequency, power)
        ax.axvline(x=24, color='red', linestyle=':', label='24h period')
        ax.set_xlabel('Period [h]')
        ax.set_ylabel('Power')
        ax.set_title('LombScargle Movement Speed [72h]')
        ax.legend()
        plt.show()

    day_frequency = (1 / 24)
    max_power_idx = np.argmax(power)
    max_frequency = frequency[max_power_idx]
    max_power = power[max_power_idx]
    circadian_power = ls.power(day_frequency)


    return dict(max_power=max_power, max_frequency=max_frequency,
                circadian_power=circadian_power, max_frequency_h=1 / max_frequency)

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.getcwd(), "..", "data", "velocity_2088_2019.csv"))
    print(spectral_analysis(df, column="velocity", sampling_rate=6))

