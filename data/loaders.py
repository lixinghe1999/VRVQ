import os ; opj=os.path.join
import re
from audiotools import AudioSignal
from audiotools.core import util
import torch
from typing import List, Callable


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
        
        items = {
            "signal": signal,
            "source_idx": source_idx,
            "item_idx": item_idx,
            "source": str(self.sources[source_idx]),
            "path": str(path),
        }
        
        return items
    

# class AudioDataset:
#     def __init__(
#         self, 
#         loader: AudioLoader,
#         sample_rate: int,
#         n_examples: int,
#         duration: float,
#         loudness_cutoff: float = -40,
#         num_channels: int = 1,
#         without_replacement: bool = True
#     ):
#         self.loader = loader
#         self.sample_rate = sample_rate
#         self.n_examples = n_examples
#         self.duration = duration
#         self.loudness_cutoff = loudness_cutoff
#         self.num_channels = num_channels
#         self.without_replacement = without_replacement
    
#     def __len__(self):
#         return self.n_examples
    
#     def __getitem__(self, idx):
#         state = util.random_state(idx)
#         loader_kwargs = {
#             "state": state,
#             "sample_rate": self.sample_rate,
#             "duration": self.duration,
#             "loudness_cutoff": self.loudness_cutoff,
#             "num_channels": self.num_channels,
#             "item_idx": idx if self.without_replacement else None
#         }
#         item = self.loader(**loader_kwargs)
#         item["idx"] = idx
#         return item
    
#     @staticmethod
#     def collate(list_of_dicts, n_splits=None):
#         return util.collate(list_of_dicts, n_splits=n_splits)