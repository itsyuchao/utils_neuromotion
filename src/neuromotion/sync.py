"""Anchor any stream to the rpi-fetch grounding clock: match a device's trigger
durations (magnet / electromagnet presses, Motive takes) to rpi-fetch edge
durations, check the match, and set raw meas_date from the anchor trigger."""
import mne
import numpy as np
import pandas as pd


def match_triggers_to_rpitime(triggers: list[float], rpitime: list[float], tol_s=0.008):
    """
    For each target DURATION in `a`, return ALL (start,end) non-nan index pairs in B
    such that sum(B[start:end+1]) ~= target within `tol`.

    - NaNs in duration_B are ignored (windows can cross them).
    - Indices returned are 0-based positional indices into duration_B.
    - end is inclusive.
    """
    a = np.asarray(triggers, dtype=float)
    B = np.asarray(rpitime, dtype=float)

    out = {i: [] for i in range(len(a))}

    print("Residual > 0 means rpi time is longer than target duration, < 0 means shorter.")

    for k, target in enumerate(a):
        if np.isnan(target):
            continue

        # brute force over all starts
        match_count = 0 
        for start in range(len(B)):
            s = 0.0
            # iteratively add until end
            for end in range(start, len(B)):
                s += B[end]
                if abs(s - target) <= tol_s:
                    start_idx = rpitime.index[start-1]  # guaranteed to be valid since start strictly >0 due to how durations are computed.
                    end_idx = rpitime.index[end]
                    out[k].append((start_idx, end_idx)) # return the actual indices in original dataframe 
                    print(f"Match found from rpi time index {start_idx} to {end_idx} for target duration {target:.3f} at index {k} with residual {s - target:.3f}.")
                    match_count += 1
        if match_count == 0:
            out[k].append((np.nan, np.nan))
        elif match_count > 1:
            print(f"Warning: Found {match_count} matches for target duration {target:.3f} at index {k}. Returning all matches.")

    return out


def filter_sequential_matches(matches: dict[int, list[tuple[float, float]]]):
    """
    Prune ambiguous (start, end) candidates for ONE train by sequential
    indexing: rpi indices must only increase with trigger-duration index.
    Consecutive durations share their middle trigger pulse, so a candidate for
    duration k+1 must START at the same rpi index where duration k's window
    ENDS; across an unmatched (NaN) duration the constraint relaxes to
    end <= start. Only candidates lying on at least one full consistent chain
    are kept. Returns the input unchanged when fewer than two durations are
    matched, or when no complete chain exists (check_train_matched flags it).
    """
    matched_keys = [k for k in sorted(matches)
                    if any(not np.isnan(c[0]) for c in matches[k])]
    if len(matched_keys) < 2:
        return matches

    def compatible(prev_key, prev_cand, next_key, next_cand):
        if next_key == prev_key + 1:
            return prev_cand[1] == next_cand[0]  # shared trigger pulse
        return prev_cand[1] <= next_cand[0]      # gap over unmatched duration(s)

    # Forward: candidate extends a chain from the first matched duration.
    fwd = {matched_keys[0]: [True] * len(matches[matched_keys[0]])}
    for pk, ck in zip(matched_keys[:-1], matched_keys[1:]):
        fwd[ck] = [any(f and compatible(pk, pc, ck, cc)
                       for f, pc in zip(fwd[pk], matches[pk]))
                   for cc in matches[ck]]
    # Backward: candidate reaches the last matched duration.
    bwd = {matched_keys[-1]: [True] * len(matches[matched_keys[-1]])}
    for pk, ck in reversed(list(zip(matched_keys[:-1], matched_keys[1:]))):
        bwd[pk] = [any(b and compatible(pk, pc, ck, cc)
                       for b, cc in zip(bwd[ck], matches[ck]))
                   for pc in matches[pk]]

    filtered = dict(matches)
    for k in matched_keys:
        kept = [c for c, f, b in zip(matches[k], fwd[k], bwd[k]) if f and b]
        if not kept:
            print(f"Warning: no sequentially consistent chain through duration "
                  f"index {k}. Keeping all matches.")
            return matches
        if len(kept) < len(matches[k]):
            print(f"Sequential filter at index {k}: "
                  f"{len(matches[k])} -> {len(kept)} match(es), kept {kept}.")
        filtered[k] = kept
    return filtered


def match_trains_to_rpitime(trains: dict[str: list[float]], rpitime: list[float], tol_s=0.008):
    """
    For each list of target durations in `trains`, return ALL (start,end) non-nan index pairs in B
    such that sum(B[start:end+1]) ~= target within `tol`, then pruned by
    filter_sequential_matches so only sequentially consistent matches remain
    (rpi indices increase over trigger durations, consecutive windows chained
    at their shared trigger pulse).
    """
    out = {}
    for key, triggers in trains.items():
        print(f"Matching train '{key}' with {len(triggers)} triggers to rpi time...")
        trig_dur = np.diff(triggers) # get the duration of each trigger train by taking the difference between consecutive triggers
        out[key] = filter_sequential_matches(
            match_triggers_to_rpitime(trig_dur, rpitime, tol_s=tol_s))
    return out


def check_train_matched(train_out: dict[int: list[tuple[float, float]]]):
    """
    Check for dict of trains match result 
    """
    bad_train = []
    for key, matches in train_out.items():
        # check consecutive matches are possible, even if multiple potential matches under one trigger duration 
        for i in range(len(matches)-1):
            consecutive = False
            for j in range(len(matches[i])):
                # check valid match exist
                if np.isnan(matches[i][j][0]) or np.isnan(matches[i][j][1]):
                    print(f"Key {key} has unmatched trigger at index {i}.")
                    bad_train.append(key)
                    break
                # check consecutive match possible 
                for k in range(len(matches[i+1])): 
                    if matches[i][j][1] <= matches[i+1][k][0]: 
                        consecutive = True
                        break
                if not consecutive:
                    print(f"Key {key} has non-consecutive matches between trigger {i} and {i+1}.") 
                    bad_train.append(key)
                    break
    good_train = [key for key in train_out.keys() if key not in bad_train]
    print(f"Good trains: {good_train}")
    return good_train


def set_meas_date_from_trigger(
    raw: mne.io.BaseRaw,
    trigger_rel_s: float,
    trigger_abs_iso,
):
    """
    Set raw.info['meas_date'] so that raw time trigger_rel_s corresponds to trigger_abs_iso.

    meas_date := absolute time of raw.times[0] (i.e., raw start).
    So raw_start_abs = trigger_abs - trigger_rel_s.
    """
    # Coerce TimeLike (str allowed per signature) to tz-aware UTC Timestamp;
    # no-op if already tz-aware.
    trigger_abs_iso = pd.to_datetime(trigger_abs_iso, utc=True)
    raw_start_abs_iso = trigger_abs_iso - pd.to_timedelta(float(trigger_rel_s), unit="s")

    # Convert to python datetime with tzinfo=UTC (MNE is happy with tz-aware datetimes)
    meas_date_dt = raw_start_abs_iso.to_pydatetime()
    raw.set_meas_date(meas_date_dt)
    return meas_date_dt


