from __future__ import annotations
import numpy as np
import warnings
import mne

def _reref_ieeg_ch(inst, reref_ch):
    """
    Re-reference Percept iEEG channels to derive sequential bipolar montages.

    The RNS device records three bipolar pairs per hemisphere from a 4-contact
    strip (contacts 0-3):
        ZERO_THREE  = V0 - V3
        ONE_THREE   = V1 - V3
        ZERO_TWO    = V0 - V2

    This function derives the requested sequential bipolar channel:
        ZERO_ONE    = ZERO_THREE - ONE_THREE       (V0 - V1)
        ONE_TWO     = ONE_THREE - ZERO_THREE + ZERO_TWO  (V1 - V2)
        TWO_THREE   = ZERO_THREE - ZERO_TWO        (V2 - V3)

    Parameters
    ----------
    inst : mne.io.Raw | mne.Epochs
        MNE object whose channel names follow the pattern
        ``{CONTACT}_{CONTACT}_{SIDE}`` (e.g. ``ZERO_THREE_LEFT``).
    reref_ch : str
        Desired output channel, e.g. ``"ZERO_ONE_LEFT"``.
        Valid contact pairs (order matters): ZERO_ONE, ONE_TWO, TWO_THREE.
        Reversed pairs (e.g. ONE_ZERO) will raise a warning and return None.

    Returns
    -------
    inst_out : mne.io.Raw | mne.Epochs
        A copy containing only the single re-referenced channel, or None if
        the requested pair is invalid.
    """
    CONTACTS = ["ZERO", "ONE", "TWO", "THREE"]
    VALID_PAIRS = {"ZERO_ONE", "ONE_TWO", "TWO_THREE"}
    REVERSED_PAIRS = {"ONE_ZERO", "TWO_ONE", "THREE_TWO",
                      "THREE_ZERO", "TWO_ZERO", "THREE_ONE"}

    # ---- parse the requested channel string ----
    parts = reref_ch.upper().split("_")
    if len(parts) != 3 or parts[0] not in CONTACTS or parts[1] not in CONTACTS:
        raise ValueError(
            f"reref_ch must be in the form CONTACT_CONTACT_SIDE "
            f"(e.g. ZERO_ONE_LEFT), got '{reref_ch}'"
        )
    pair = f"{parts[0]}_{parts[1]}"
    side = parts[2]  # e.g. "LEFT" or "RIGHT"

    if pair in REVERSED_PAIRS:
        warnings.warn(
            f"Reversed pair '{pair}' requested — order matters for bipolar "
            f"re-referencing. Valid sequential pairs are: {sorted(VALID_PAIRS)}. "
            f"Returning None.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if pair not in VALID_PAIRS:
        raise ValueError(
            f"Unsupported pair '{pair}'. Valid pairs: {sorted(VALID_PAIRS)}"
        )

    # ---- build source channel names for this side ----
    src = {
        "ZERO_THREE": f"ZERO_THREE_{side}",
        "ONE_THREE":  f"ONE_THREE_{side}",
        "ZERO_TWO":   f"ZERO_TWO_{side}",
    }

    # ---- formulas: each maps target -> [(coeff, source_ch), ...] ----
    formulas = {
        "ZERO_ONE":  [(1, src["ZERO_THREE"]), (-1, src["ONE_THREE"])],
        "ONE_TWO":   [(1, src["ONE_THREE"]),  (-1, src["ZERO_THREE"]),
                      (1, src["ZERO_TWO"])],
        "TWO_THREE": [(1, src["ZERO_THREE"]), (-1, src["ZERO_TWO"])],
    }

    formula = formulas[pair]

    # ---- verify all required source channels exist ----
    needed = [ch for _, ch in formula]
    missing = [ch for ch in needed if ch not in inst.ch_names]
    if missing:
        raise ValueError(
            f"Source channels {missing} not found in inst.ch_names: "
            f"{inst.ch_names}"
        )

    # ---- compute the re-referenced data ----
    inst_copy = inst.copy().pick(needed)
    data = inst_copy.get_data()  # (n_channels, n_samples) or (n_epochs, n_channels, n_samples)
    ch_idx = {ch: i for i, ch in enumerate(inst_copy.ch_names)}

    if isinstance(inst, mne.io.BaseRaw):
        result = sum(coeff * data[ch_idx[ch]] for coeff, ch in formula)
        result = result[np.newaxis, :]  # (1, n_samples)
        new_info = mne.create_info([reref_ch.upper()], inst.info["sfreq"],
                                   ch_types="seeg")
        return mne.io.RawArray(result, new_info)

    elif isinstance(inst, mne.BaseEpochs):
        result = sum(coeff * data[:, ch_idx[ch], :] for coeff, ch in formula)
        result = result[:, np.newaxis, :]  # (n_epochs, 1, n_samples)
        new_info = mne.create_info([reref_ch.upper()], inst.info["sfreq"],
                                   ch_types="seeg")
        return mne.EpochsArray(result, new_info, events=inst.events,
                               tmin=inst.tmin, metadata=inst.metadata)
    else:
        raise TypeError(
            f"inst must be mne.io.Raw or mne.Epochs, got {type(inst)}"
        )


def pick_or_reref(inst: mne.io.BaseRaw | mne.BaseEpochs, ieeg_picks: list[str] | str):
    """Pick channels from inst, re-referencing any that don't exist as-is."""
    picks_list = ieeg_picks if isinstance(ieeg_picks, list) else [ieeg_picks]
    existing = [ch for ch in picks_list if ch in inst.ch_names]
    to_reref = [ch for ch in picks_list if ch not in inst.ch_names]
    parts = []
    if existing:
        parts.append(inst.copy().pick(existing))
    for ch in to_reref:
        rerefed = _reref_ieeg_ch(inst, ch)
        if rerefed is not None:
            parts.append(rerefed)
    if not parts:
        raise ValueError(f"No valid channels from {picks_list} in {inst.ch_names}")
    inst = parts[0]
    if len(parts) > 1:
        inst.add_channels(parts[1:])
    return inst