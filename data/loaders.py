"""
This code is a modified version of `AudioLoader` and `AudioDataset` 
from the `audiotools.data.datasets` module.
Original source: https://github.com/descriptinc/audiotools/blob/master/audiotools/data/datasets.py
"""

import os ; opj=os.path.join
import re
from audiotools import AudioSignal
from audiotools.core import util
import torch
from typing import List, Callable, Union, Dict
# from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import default_matcher, align_lists


class AudioLoader:
    def __init__(
        self,
        sources: List[str] = None,
        weights: List[float] = None,
        transform: Callable = None,
        relative_path: str = "",
        ext: List[str] = util.AUDIO_EXTENSIONS,
        shuffle: bool = True,
        shuffle_state: int = 0,
    ):
        self.audio_lists = util.read_sources(
            sources, relative_path=relative_path, ext=ext
        )

        self.audio_indices = [
            (src_idx, item_idx)
            for src_idx, src in enumerate(self.audio_lists)
            for item_idx in range(len(src))
        ]
        if shuffle:
            state = util.random_state(shuffle_state)
            state.shuffle(self.audio_indices)

        self.sources = sources
        self.weights = weights
        self.transform = transform
        
    def __call__(
        self,
        state,
        sample_rate,
        duration,
        loudness_cutoff=-40,
        num_channels=1,
        offset = None,
        source_idx = None,
        item_idx=None,
        global_idx=None
    ):
        if source_idx is not None and item_idx is not None:
            try:
                audio_info = self.audio_lists[source_idx][item_idx]
            except:
                audio_info = {"path": "none"}
                
        elif global_idx is not None:
            source_idx, item_idx = self.audio_indices[
                global_idx % len(self.audio_indices)
            ]
            audio_info = self.audio_lists[source_idx][item_idx]
        else:
            audio_info, source_idx, item_idx = util.choose_from_list_of_lists(
                state, self.audio_lists, p=self.weights
            ) 
            
        path = audio_info["path"]
        signal = AudioSignal.zeros(duration, sample_rate, num_channels)

        if path != "none":
            if offset is None:
                if duration is not None:
                    try:
                        signal = AudioSignal.salient_excerpt(
                            path,
                            duration=duration,
                            state=state,
                            loudness_cutoff=loudness_cutoff,
                        )
                    except Exception as e:
                        ### AudioSignal has 6 channels
                        ###  /data2/yoongi/dataset/audioset_wav44/wavs/balanced_train/000003/id_s-HejPHC-Hk.wav
                        signal = AudioSignal(path, 
                                            offset=0,
                                            duration=duration)
                else:
                    signal = AudioSignal(path, offset=offset, duration=duration)
                    
        if num_channels == 1:
            signal = signal.to_mono()
        signal = signal.resample(sample_rate)
        
        if duration is not None:
            if signal.duration < duration:
                signal = signal.zero_pad_to(int(duration * sample_rate))
                
        for k, v in audio_info.items():
            signal.metadata[k] = v
            
        item = {
            "signal": signal,
            "source_idx": source_idx,
            "item_idx": item_idx,
            "source": str(self.sources[source_idx]),
            "path": str(path),
        }
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(state, signal=signal)
        return item
    


class AudioDataset:
    def __init__(
        self,
        loaders: Union[AudioLoader, List[AudioLoader], Dict[str, AudioLoader]],
        sample_rate: int,
        n_examples: int = 1000,
        duration: float = 0.5,
        offset: float = None,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        transform: Callable = None,
        aligned: bool = False,
        shuffle_loaders: bool = False,
        matcher: Callable = default_matcher,
        without_replacement: bool = True,
    ):
        # Internally we convert loaders to a dictionary
        if isinstance(loaders, list):
            loaders = {i: l for i, l in enumerate(loaders)}
        elif isinstance(loaders, AudioLoader):
            loaders = {0: loaders}

        self.loaders = loaders
        self.loudness_cutoff = loudness_cutoff
        self.num_channels = num_channels

        self.length = n_examples
        self.transform = transform
        self.sample_rate = sample_rate
        self.duration = duration
        self.offset = offset
        self.aligned = aligned
        self.shuffle_loaders = shuffle_loaders
        self.without_replacement = without_replacement

        if aligned:
            loaders_list = list(loaders.values())
            for i in range(len(loaders_list[0].audio_lists)):
                input_lists = [l.audio_lists[i] for l in loaders_list]
                # Alignment happens in-place
                align_lists(input_lists, matcher)

    def __getitem__(self, idx):
        state = util.random_state(idx)
        offset = None if self.offset is None else self.offset
        item = {}
        keys = list(self.loaders.keys())
        if self.shuffle_loaders:
            state.shuffle(keys)

        loader_kwargs = {
            "state": state,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "loudness_cutoff": self.loudness_cutoff,
            "num_channels": self.num_channels,
            "global_idx": idx if self.without_replacement else None,
        }

        # Draw item from first loader
        loader = self.loaders[keys[0]]
        item[keys[0]] = loader(**loader_kwargs)

        for key in keys[1:]:
            loader = self.loaders[key]
            if self.aligned:
                # Path mapper takes the current loader + everything
                # returned by the first loader.
                offset = item[keys[0]]["signal"].metadata["offset"]
                loader_kwargs.update(
                    {
                        "offset": offset,
                        "source_idx": item[keys[0]]["source_idx"],
                        "item_idx": item[keys[0]]["item_idx"],
                    }
                )
            item[key] = loader(**loader_kwargs)

        # Sort dictionary back into original order
        keys = list(self.loaders.keys())
        item = {k: item[k] for k in keys}

        item["idx"] = idx
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(
                state=state, signal=item[keys[0]]["signal"]
            )

        # If there's only one loader, pop it up
        # to the main dictionary, instead of keeping it
        # nested.
        if len(keys) == 1:
            item.update(item.pop(keys[0]))

        return item

    def __len__(self):
        return self.length

    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        """Collates items drawn from this dataset. Uses
        :py:func:`audiotools.core.util.collate`.

        Parameters
        ----------
        list_of_dicts : typing.Union[list, dict]
            Data drawn from each item.
        n_splits : int
            Number of splits to make when creating the batches (split into
            sub-batches). Useful for things like gradient accumulation.

        Returns
        -------
        dict
            Dictionary of batched data.
        """
        return util.collate(list_of_dicts, n_splits=n_splits)


class ConcatDataset(AudioDataset):
    def __init__(self, datasets: list):
        self.datasets = datasets

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        dataset = self.datasets[idx % len(self.datasets)]
        return dataset[idx // len(self.datasets)]