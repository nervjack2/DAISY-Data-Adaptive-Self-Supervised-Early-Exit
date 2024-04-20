# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/hubconf.py ]
#   Synopsis     [ the HuBERT torch hubconf ]
#   Author       [ S3PRL / Kushal Lakhotia ]
"""*********************************************************************************************"""


import logging
import os
import time
from pathlib import Path

from filelock import FileLock

from s3prl.util.download import _urls_to_filepaths

from .convert import load_and_convert_fairseq_ckpt
from .expert import LegacyUpstreamExpert as _LegacyUpstreamExpert
from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)

NEW_ENOUGH_SECS = 2.0

def ee_hubert_custom(
    f_type: str,
    ckpt: str,
    legacy: bool = False,
    fairseq: bool = False,
    refresh: bool = False,
    **kwargs,
):
    assert not (legacy and fairseq), (
        "The option 'legacy' will directly load a fairseq checkpoint, "
        "while the option 'fairseq' will first convert the fairseq checkpoint to "
        "be fairseq indenpendent and then load the checkpoint. "
        "These two options cannot be used jointly."
    )

    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    if fairseq:
        ckpt: Path = Path(ckpt)
        converted_ckpt = ckpt.parent / f"{ckpt.stem}.converted.pt"
        lock_file = Path(str(converted_ckpt) + ".lock")

        logger.info(f"Converting a fairseq checkpoint: {ckpt}")
        logger.info(f"To: {converted_ckpt}")

        with FileLock(str(lock_file)):
            if not converted_ckpt.is_file() or (
                refresh and (time.time() - os.path.getmtime(ckpt)) > NEW_ENOUGH_SECS
            ):
                load_and_convert_fairseq_ckpt(ckpt, converted_ckpt)

        ckpt = converted_ckpt

    assert os.path.isfile(ckpt)
    if legacy:
        return _LegacyUpstreamExpert(f_type, ckpt, **kwargs)
    else:
        return _UpstreamExpert(f_type, ckpt, **kwargs)


def ee_hubert_local_cluster(*args, **kwargs):
    feature_type = 'cluster'
    return ee_hubert_custom(f_type=feature_type, *args, **kwargs)

def ee_hubert_local(*args, **kwargs):
    feature_type = 'normal'
    return ee_hubert_custom(f_type=feature_type, *args, **kwargs)
