"""Small one-way RF link-budget helpers for scripted ESM experiments."""

import math

C_MPS = 299_792_458.0


def watts_to_dbm(power_w):
    power_w = float(power_w)
    if power_w <= 0.0:
        raise ValueError("power_w must be positive")
    return 10.0 * math.log10(power_w * 1000.0)


def free_space_path_loss_db(frequency_hz, range_km):
    frequency_hz = float(frequency_hz)
    range_m = float(range_km) * 1000.0
    if frequency_hz <= 0.0 or range_m <= 0.0:
        raise ValueError("frequency_hz and range_km must be positive")
    wavelength_m = C_MPS / frequency_hz
    return 20.0 * math.log10(4.0 * math.pi * range_m / wavelength_m)


def received_power_dbm(
    tx_peak_power_w,
    tx_peak_gain_dbi,
    relative_pattern_gain_db,
    frequency_hz,
    range_km,
    rx_antenna_gain_dbi,
    system_loss_db=0.0,
):
    return (
        watts_to_dbm(tx_peak_power_w)
        + float(tx_peak_gain_dbi)
        + float(relative_pattern_gain_db)
        - free_space_path_loss_db(frequency_hz, range_km)
        + float(rx_antenna_gain_dbi)
        - float(system_loss_db)
    )
