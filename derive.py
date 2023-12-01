import numpy as np
from AudioSeg import windows


def derivative(sample, sample_rate, talk_range, window_size=0.2, threshold=5000):
    window_size = int(window_size * sample_rate)
    sample_sectors = [sample[s:e].astype(np.float64) for (s, e) in talk_range]
    for s in sample_sectors:
        cnt = 0
        signal_windows = list(windows(s, window_size, window_size))
        for sw in signal_windows:
            local_max = np.argmax(sw)
            if local_max == sw.shape[0] - 1:
                continue
            local_min = np.argmin(sw[local_max+1:])
            local_min += local_max+1
            
            dx = (sw[local_min] - sw[local_max]) / (local_min - local_max)
            if abs(dx) > threshold:
                cnt += 1
        if cnt >= 2:
            return True
    
    return False