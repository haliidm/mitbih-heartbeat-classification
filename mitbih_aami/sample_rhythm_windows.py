"""Annotation-driven AFIB/AFL N-window supplementation for MIT-BIH."""
import numpy as np
import wfdb
from .label_config import FS, WINDOW_HALF, WINDOW_SAMPLES


def rhythm_intervals(annotation, length):
    changes = [(int(t), str(note).replace("\x00", "").strip())
               for t, note in zip(annotation.sample, annotation.aux_note)
               if str(note).replace("\x00", "").strip().startswith("(")]
    result = []
    for i, (start, rhythm) in enumerate(changes):
        end = changes[i+1][0] if i+1 < len(changes) else length
        if rhythm in ("(AFIB", "(AFL"):
            result.append((max(0,start),min(length,end),rhythm[1:]))
    return result


def add_rhythm_n_windows(dataset, splits, database_dir, cap=6, seed=42):
    """Append disjoint N-only windows outside ALL original centered windows.

    Original train/validation/test assignments remain unchanged. Supplemental
    windows are split within each record/rhythm when sufficient windows exist.
    No labels are altered and no model predictions are used for selection.
    """
    if cap < 1:
        raise ValueError("rhythm N window cap must be >=1")
    original_count = len(dataset.items)
    audit, summary = [], []
    rng = np.random.default_rng(seed)
    by_record = [[] for _ in dataset.records]
    for ri,center,_,_ in dataset.items:
        by_record[ri].append(center)
    for ri,record in enumerate(dataset.records):
        ann = wfdb.rdann(str(database_dir / record.name), "atr")
        episodes = rhythm_intervals(ann,len(record.signal))
        if not episodes:
            continue
        covered = np.zeros(len(record.signal),dtype=bool)
        for center in by_record[ri]:
            covered[max(0,center-WINDOW_HALF):min(len(covered),center+WINDOW_HALF)] = True
        times = np.array([t for t,c in record.events],dtype=np.int64)
        classes = np.array([c for t,c in record.events],dtype=np.int64)
        normal_times = times[classes==1]
        pools = {"AFIB":[],"AFL":[]}
        last_end = -1
        for start,end,rhythm in episodes:
            for desired in range(start+WINDOW_HALF,end-WINDOW_HALF+1,WINDOW_SAMPLES):
                if not len(normal_times):
                    continue
                j = int(np.searchsorted(normal_times,desired))
                choices = normal_times[max(0,j-1):min(len(normal_times),j+1)]
                center = int(min(choices,key=lambda t:abs(int(t)-desired)))
                left,right = center-WINDOW_HALF,center+WINDOW_HALF
                if left < start or right > end or left < last_end or covered[left:right].any():
                    continue
                lo,hi = np.searchsorted(times,[left,right])
                if hi<=lo or np.any(classes[lo:hi]!=1) or np.any(record.target[left:right]>1):
                    continue
                pools[rhythm].append((center,start,end))
                last_end = right
        for rhythm,candidates in pools.items():
            if not candidates:
                if any(item[2]==rhythm for item in episodes):
                    summary.append({"record":record.name,"rhythm":rhythm,"eligible":0,"selected":0})
                continue
            selected = [candidates[int(i)] for i in np.linspace(0,len(candidates)-1,min(cap,len(candidates)),dtype=int)]
            size = len(selected)
            order = rng.permutation(size)
            n_val = max(1,round(.1*size)) if size>=2 else 0
            n_test = max(1,round(.2*size)) if size>=3 else 0
            n_train = size-n_val-n_test
            assignments = {}
            for key,part in zip(("train","validation","test"),(order[:n_train],order[n_train:n_train+n_val],order[n_train+n_val:])):
                for index in part: assignments[int(index)] = key
            for i,(center,start,end) in enumerate(selected):
                split = assignments[i]
                event_index = int(np.searchsorted(times,center))
                original_sample = int(record.original_events[event_index][0])
                index = len(dataset.items)
                dataset.items.append((ri,center,1,original_sample))
                splits[split].append(index)
                audit.append({"record":record.name,"rhythm":rhythm,"split":split,"dataset_index":index,
                              "center_sample":center,"start_seconds":(center-WINDOW_HALF)/FS,
                              "end_seconds":(center+WINDOW_HALF)/FS,
                              "rhythm_start_seconds":start/FS,"rhythm_end_seconds":end/FS,
                              "selection":"uniform over eligible intervals; no original-window overlap"})
            summary.append({"record":record.name,"rhythm":rhythm,"eligible":len(candidates),"selected":size})
    return {"original_window_count":original_count,"supplemental_window_count":len(audit),
            "cap_per_record_per_rhythm":cap,"records":summary,"windows":audit}
