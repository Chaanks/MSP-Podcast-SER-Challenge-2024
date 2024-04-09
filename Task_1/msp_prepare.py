"""
MSP-PODCAST data preparation.

Authors
 * Jarod Duret 2024
"""

import logging
import json
import pathlib as pl
from tqdm import tqdm
from speechbrain.dataio.dataio import load_pkl, save_pkl


logger = logging.getLogger(__name__)
OPT_FILE = "opt_msp_prepare.pkl"
METADATA = "Labels/labels_consensus.json"
TRAIN_JSON = "train.json"
VALID_JSON = "valid.json"
TEST_JSON = "test.json"
WAVS = "Audios"
TRANSCRIPTS = "Transcripts"
SPLITS = {
    "train": "Train",
    "valid": "Development",
}


def prepare_msp(
    data_folder,
    save_folder,
    splits=["train", "valid"],
    filter_keys=[],
    upsampling=False,
    upsampling_distribution=None,
    skip_prep=False,
):
    """
    Prepares the csv files for the MSP-PODCAST datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LJspeech dataset is stored
    save_folder : str
        The directory where to store the csv/json files
    splits : list
        List of dataset splits to prepare
    filter_keys: list
        List of keys to filter
    skip_prep : bool
        If True, skip preparation

    Returns
    -------
    None

    Example
    -------
    >>> from recipes.MSP-PODCAST.msp_prepare import prepare_msp
    >>> data_folder = 'data/msp/'
    >>> save_folder = 'save/'
    >>> splits = ['train', 'valid']
    >>> prepare_msp(data_folder, save_folder, splits)
    """

    if skip_prep:
        return

    # Creating configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "save_folder": save_folder,
        "filter_keys": filter_keys,
        "upsampling": upsampling,
        "upsampling_distribution": upsampling_distribution,
    }

    save_folder = pl.Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)

    data_folder = pl.Path(data_folder)
    meta_json = data_folder / METADATA
    wavs_folder = data_folder / WAVS
    transcripts_folder = data_folder / TRANSCRIPTS
    save_opt = save_folder / OPT_FILE

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional check to make sure labels_consensus.json and wavs folder exists
    assert meta_json.exists(), f"{METADATA} does not exist"
    assert wavs_folder.exists(), f"{WAVS} folder does not exist"
    assert transcripts_folder.exists(), f"{TRANSCRIPTS} folder does not exist"

    logger.info("Creating json file for msp-podcast Dataset..")

    with open(meta_json, "r") as file:
        metadata = json.load(file)

    for split in splits:
        logger.info(f"Preparing {split}..")
        split_name = SPLITS[split]
        set = {}
        for uttid, items in tqdm(metadata.items()):
            if items["Split_Set"] != split_name:
                continue
            if items["EmoClass"] in filter_keys:
                continue
            uttid = uttid.split(".")[0]
            wav = wavs_folder / f"{uttid}.wav"
            transcript = open(transcripts_folder / f"{uttid}.txt").readline()

            set[uttid] = {
                "wav": wav.as_posix(),
                "text": transcript,
                "emo": items["EmoClass"],
                "act": items["EmoAct"],
                "dom": items["EmoDom"],
                "val": items["EmoVal"],
                "spk": items["SpkrID"],
                "sex": items["Gender"],
            }

            if (
                split == "train"
                and upsampling
                and upsampling_distribution[items["EmoClass"]] > 1
            ):
                for i in range(0, upsampling_distribution[items["EmoClass"]]):
                    set[f"{uttid}-{i}"] = set[uttid]

        save_json = save_folder / f"{split}.json"
        with open(save_json, "w") as train_file:
            json.dump(set, train_file, indent=2)

    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf):
    """
    Detects if the msp-podcast data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking json files
    skip = True

    split_files = {
        "train": TRAIN_JSON,
        "valid": VALID_JSON,
        "test": TEST_JSON,
    }

    for split in splits:
        if not (save_folder / split_files[split]).exists():
            skip = False

    #  Checking saved options
    save_opt = save_folder / OPT_FILE
    if skip is True:
        if save_opt.is_file():
            opts_old = load_pkl(save_opt.as_posix())
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False
    return skip
