"""Atomic JSON output shared by training, calibration and simulation."""

import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_npz(path, compression='deflate', **arrays):
    """Atomic, standard NPZ with a selectable lossless ZIP codec.

    All original arrays remain directly readable by numpy.load. No reduced
    precision, dropped fields, reconstruction rules or external dependencies.
    """
    import zipfile
    import numpy as np
    codecs={'deflate':zipfile.ZIP_DEFLATED,'bzip2':zipfile.ZIP_BZIP2,'lzma':zipfile.ZIP_LZMA}
    if compression not in codecs:raise ValueError('Unknown lossless NPZ codec')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with zipfile.ZipFile(temporary,'w',compression=codecs[compression],allowZip64=True) as archive:
        for key,value in arrays.items():
            with archive.open(key+'.npy','w',force_zip64=True) as member:
                np.lib.format.write_array(member,np.asarray(value),allow_pickle=False)
    temporary.replace(path)
