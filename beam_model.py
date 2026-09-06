import math

import numpy as np


def wrap_angle_deg(angle_deg):
    """Wrap an angle to [-180, 180)."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


class RotatingSincBeam:
    """Simple rotating antenna model for emitter-behaviour simulation.

    The beamwidth parameter is the full -3 dB power beamwidth. The normalized
    sinc-squared power pattern is scaled so gain=0.5 at +/- beamwidth/2.
    This is intentionally a compact engineering model, not a full antenna model.
    """

    # Solve sinc(x)^2 = 0.5 where numpy.sinc(x)=sin(pi*x)/(pi*x).
    _HALF_POWER_SINC_X = 0.4429464706894523

    def __init__(
        self,
        beamwidth_deg=3.0,
        scan_rate_rpm=30.0,
        initial_azimuth_deg=0.0,
        sidelobe_floor_db=-45.0,
        fixed_azimuth_deg=None,
    ):
        self.beamwidth_deg = float(beamwidth_deg)
        self.scan_rate_rpm = float(scan_rate_rpm)
        self.initial_azimuth_deg = float(initial_azimuth_deg)
        self.sidelobe_floor_db = float(sidelobe_floor_db)
        self.fixed_azimuth_deg = None if fixed_azimuth_deg is None else float(fixed_azimuth_deg)

    @property
    def scan_period_s(self):
        if self.fixed_azimuth_deg is not None or self.scan_rate_rpm == 0.0:
            return math.inf
        return 60.0 / abs(self.scan_rate_rpm)

    def azimuth_deg(self, time_s):
        if self.fixed_azimuth_deg is not None:
            return self.fixed_azimuth_deg % 360.0
        deg_per_s = self.scan_rate_rpm * 6.0
        return (self.initial_azimuth_deg + deg_per_s * float(time_s)) % 360.0

    def power_gain(self, target_bearing_deg, time_s):
        delta_deg = wrap_angle_deg(float(target_bearing_deg) - self.azimuth_deg(time_s))
        half_bw = max(0.5 * self.beamwidth_deg, 1e-9)
        x = self._HALF_POWER_SINC_X * delta_deg / half_bw
        gain = float(np.sinc(x) ** 2)
        floor = 10.0 ** (self.sidelobe_floor_db / 10.0)
        return max(gain, floor)

    def voltage_gain(self, target_bearing_deg, time_s):
        return math.sqrt(self.power_gain(target_bearing_deg, time_s))

    def gain_db(self, target_bearing_deg, time_s):
        return 10.0 * math.log10(self.power_gain(target_bearing_deg, time_s))
