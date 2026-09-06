# pdw.py

from dataclasses import dataclass, field


@dataclass
class PDW:
    pdw_id: int
    toa_s: float
    pulse_width_s: float
    frequency_hz: float
    amplitude_dbfs: float
    modulation_type: str = "UNKNOWN"
    modulation_bandwidth_hz: float = 0.0
    modulation_confidence: float = 0.0
    modulation_scores: dict = field(default_factory=dict)
    modulation_rejection_reason: str = ""

    def __str__(self):
        return (
            f"PDW {self.pdw_id:06d}  "
            f"TOA={self.toa_s:12.9f} s  "
            f"PW={self.pulse_width_s * 1e6:8.3f} us  "
            f"FREQ={self.frequency_hz / 1e6:10.3f} MHz  "
            f"AMP={self.amplitude_dbfs:7.2f} dBFS  "
            f"MOD={self.modulation_type:11s}  "
            f"MC={100.0 * self.modulation_confidence:5.1f}%  "
            f"BW={self.modulation_bandwidth_hz / 1e6:6.3f} MHz"
        )
