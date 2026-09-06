"""Continuous block processor used by the operator UI and, later, an SDR source.

The processor owns the fast front-end boundary:
    IQ block -> pulse detector -> waveform classifier/PDW extractor -> PDW stream

It deliberately has no knowledge of scripted radar modes or antenna commands.
"""

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S
from pdw_extractor import PDWExtractor
from pulse_detector import PulseDetector


class StreamingESMProcessor:
    def __init__(self, source):
        self.source = source
        self.detector = PulseDetector(
            DETECTION_THRESHOLD, source.sample_rate_hz, MIN_PULSE_WIDTH_S
        )
        self.extractor = PDWExtractor(source.sample_rate_hz, source.center_frequency_hz)
        self.total_blocks = 0
        self.total_pdws = 0

    def reset(self):
        self.source.reset()
        self.extractor.next_pdw_id = 1
        self.total_blocks = 0
        self.total_pdws = 0

    def process_block(self):
        iq, metadata = self.source.read_block()
        pulses = self.detector.detect(iq)
        pdws = [
            self.extractor.extract(iq, pulse, block_start_time_s=metadata["start_time_s"])
            for pulse in pulses
        ]
        self.total_blocks += 1
        self.total_pdws += len(pdws)
        return pdws, metadata

    def process_blocks(self, count):
        out = []
        last_metadata = None
        for _ in range(int(count)):
            pdws, last_metadata = self.process_block()
            out.extend(pdws)
        return out, last_metadata
