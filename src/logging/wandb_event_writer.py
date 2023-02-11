from collections import defaultdict
import wandb
from detectron2.utils.events import EventWriter, get_event_storage

class WandBWriter(EventWriter):

    """
    Write training mettics to wandb

    """
    def __init__(self, window_size=20) -> None:
        self._window_size = window_size
        self._last_write = -1


    def write(self):
        
        storage = get_event_storage()
        to_save = defaultdict(dict)

        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            # keep scalars that have not been written
            if iter <= self._last_write:
                continue
            to_save[iter][k] = v
        if len(to_save):
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)

        for itr, scalars_per_iter in to_save.items():
            scalars_per_iter["iteration"] = itr
            wandb.log(scalars_per_iter)
