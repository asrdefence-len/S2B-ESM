import numpy as np


BARKER13 = np.array(
    [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
    dtype=np.float64,
)

# Fixed generic biphase example. This is deliberately not a Barker sequence.
BIPHASE16 = np.array(
    [1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1],
    dtype=np.float64,
)


def _frank_phases(order=4):
    phases = []
    for row in range(order):
        for column in range(order):
            phases.append(2.0 * np.pi * row * column / order)
    return np.asarray(phases, dtype=np.float64)


FRANK16_PHASES = _frank_phases(4)


def _chip_indices(num_samples, chip_count):
    indices = np.floor(np.arange(num_samples) * chip_count / num_samples).astype(int)
    return np.clip(indices, 0, chip_count - 1)


def cw(num_samples):
    return np.ones(num_samples, dtype=np.complex64)


def lfm(num_samples, normalized_bandwidth=0.30):
    """Baseband centred LFM candidate.

    normalized_bandwidth is total swept bandwidth as a fraction of sample rate.
    """
    if num_samples <= 1:
        return cw(num_samples)
    t = np.linspace(-0.5, 0.5, num_samples, endpoint=False)
    phase = np.pi * normalized_bandwidth * num_samples * t * t
    return np.exp(1j * phase).astype(np.complex64)


def barker13(num_samples):
    chips = _chip_indices(num_samples, len(BARKER13))
    return BARKER13[chips].astype(np.complex64)


def biphase(num_samples):
    chips = _chip_indices(num_samples, len(BIPHASE16))
    return BIPHASE16[chips].astype(np.complex64)


def polyphase(num_samples):
    chips = _chip_indices(num_samples, len(FRANK16_PHASES))
    return np.exp(1j * FRANK16_PHASES[chips]).astype(np.complex64)


def candidate_library(num_samples, lfm_bandwidths=None):
    if lfm_bandwidths is None:
        lfm_bandwidths = (0.10, 0.20, 0.30, 0.40)

    candidates = [
        {"family": "CW", "subtype": "CW", "samples": cw(num_samples)},
        {"family": "BIPHASE", "subtype": "BARKER13", "samples": barker13(num_samples)},
        {"family": "BIPHASE", "subtype": "GENERIC_BIPHASE", "samples": biphase(num_samples)},
        {"family": "POLYPHASE", "subtype": "FRANK16_LIKE", "samples": polyphase(num_samples)},
    ]

    for bandwidth in lfm_bandwidths:
        candidates.append(
            {
                "family": "LFM",
                "subtype": f"LFM_{bandwidth:.2f}FS",
                "samples": lfm(num_samples, bandwidth),
            }
        )

    return candidates
