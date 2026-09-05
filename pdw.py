# pdw.py

from dataclasses import dataclass


@dataclass
class PDW:
    pdw_id: int
    toa_s: float
    pulse_width_s: float
    frequency_hz: float
    amplitude_dbfs: float

    def __str__(self):
        return (
            f"PDW {self.pdw_id:06d}  "
            f"TOA={self.toa_s:12.9f} s  "
            f"PW={self.pulse_width_s * 1e6:8.3f} us  "
            f"FREQ={self.frequency_hz / 1e6:10.3f} MHz  "
            f"AMP={self.amplitude_dbfs:7.2f} dBFS"
        )