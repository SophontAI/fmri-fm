import gzip
import json
import os
import sys
import random
import re
from fnmatch import fnmatch
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import io
import tarfile
from tqdm import tqdm
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, Dataset
import torchdata.nodes as tn
from torchdata.nodes.base_node import BaseNode
from torchdata.nodes.types import Stateful

def to_tensor(img, mask, gsr=False):
    img = torch.from_numpy(img)
    if gsr:
        img = (img - 0.5) / 0.2
    if mask is not None:
        img = unmask(img, mask)
    img = img.unsqueeze(0).float()  # (C, T, H, W)
    return img

def unmask(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    unmasked = torch.zeros(
        (img.shape[0], *mask.shape), dtype=img.dtype, device=img.device
    )
    unmasked[:, mask] = img
    return unmasked

def batch_unmask(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # img: shape [B, C, D] -> [32, 16, 30191]
    # mask: shape [M, N] -> [144, 320]

    B, C, D = img.shape  # Batch size, channels, last dimension size
    M, N = mask.shape  # Mask dimensions

    # Ensure the mask is flattened to apply along the last dimension (D) of img
    flat_mask = mask.view(-1)  # shape: [M * N]
    num_unmasked_elements = flat_mask.sum()  # The number of true elements in the mask

    # Initialize an empty tensor for the unmasked output
    unmasked = torch.zeros((B, C, M * N), dtype=img.dtype, device=img.device)

    # Use broadcasting and advanced indexing to unmask
    idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # Indices where mask is True
    unmasked[:, :, idx] = img[:, :, :num_unmasked_elements]

    # Reshape the unmasked tensor to the original shape
    unmasked = unmasked.view(B, C, M, N)

    return unmasked

##### NSD #####
shared1000 = np.where(np.load("/scratch/gpfs/KNORMAN/mindeyev2_dataset/shared1000.npy"))[0]
NSD_NUM_SHARDS = 300

def load_nsd_flat_mask(folder="/weka/proj-medarc/shared/NSD-Flat/") -> torch.Tensor:
    mask = np.load(os.path.join(folder, "nsd-flat_mask.npy"))
    mask = torch.as_tensor(mask)
    return mask

def load_nsd_flat_mask_visual(folder="/weka/proj-medarc/shared/NSD-Flat/") -> torch.Tensor:
    mask = np.load(os.path.join(folder, "nsd-flat_mask_visual.npy"))
    mask = torch.as_tensor(mask)
    return mask

class NSDDataset:
    def __init__(
        self,
        split: str = "all",
        root: str = "/weka/proj-medarc/shared/NSD-Flat",
        frames: int = 16,
        same_run_samples: int = 16,
        gsr: Optional[bool] = True,
        sub: Optional[str] = None,
        ses: Optional[str] = None,
        num_sessions: Optional[int] = 40,
        run: Optional[str] = None,
        mindeye: Optional[bool] = False,
        mindeye_TR_delay: Optional[int] = 3,
        fix_start: Optional[int] = -1,
        resampled: Optional[bool] = True,
    ):
        self.split = split
        self.root = root
        self.frames = frames
        self.same_run_samples = same_run_samples
        self.gsr = gsr
        self.sub = sub
        self.num_sessions = num_sessions
        self.run = run
        self.mindeye = mindeye
        self.mindeye_TR_delay = mindeye_TR_delay
        self.mask = load_nsd_flat_mask(root)
        self.fix_start = fix_start
        self.resampled = resampled
        self.tar_files = self._build_tar_list() 


    def _build_tar_list(self):
        tar_files = []
        sample_count = 0

        for tar_file_path in tqdm(
            sorted(Path(self.root).rglob("*.tar")), desc="Building Tar File List and Counting Samples"
        ):
            tar_file_path_str = str(tar_file_path)
            tar_contains_matching_file = False

            with tarfile.open(tar_file_path_str, "r") as tar:
                for member in tar.getmembers():
                    base_name = os.path.basename(member.name)
                    if self.sub is not None:
                        if not (f"{self.sub}" in base_name):
                            continue
                    if self.num_sessions is not None:
                        session_str = base_name.split('_')[1] # Gets "ses-03" from base_name
                        session_num = int(session_str.split('-')[1]) # Gets 3 as integer
                        if session_num > self.num_sessions:
                            continue
                    if self.run is not None:
                        if not (f"{self.run}" in base_name):
                            continue

                    if not self.mindeye:
                        if self.split=="train":
                            if (("ses-01" in base_name) and not ("sub-01" in base_name)):
                                continue
                        elif self.split=="test":
                            if not (("ses-01" in base_name) and not ("sub-01" in base_name)):
                                continue

                    if member.name.endswith("bold.npy"): 
                        sample_count += 1
                        tar_contains_matching_file = True 

                if tar_contains_matching_file:
                    tar_files.append(tar_file_path_str) 

        self.len_unique_samples = sample_count 
        return tar_files

    def _pull_data(self, sample):
        bold = sample["bold.npy"]
        meta = sample["meta.json"]
        misc = sample["misc.npz"] 
        offset = misc["offset"]
        global_signal = misc["global_signal"]
        beta = misc["beta"]
        mean = misc["mean"]
        std = misc["std"]

        if self.mindeye:
            events = sample["events.json"]
            return {
                "bold": bold,
                "meta": meta,
                "offset": offset,
                "global_signal": global_signal,
                "beta": beta,
                "mean": mean,
                "std": std,
                "events": events,
            }
        else:
            return {
                "bold": bold,
                "meta": meta,
                "offset": offset,
                "global_signal": global_signal,
                "beta": beta,
                "mean": mean,
                "std": std,
            }

    def _process_sample(self, sample_dict):
        if sample_dict is None: return None # Skip samples that failed to load

        # Filter samples according to train/test split
        # this is similar to build_tar_list() but that function only excludes
        # tars that have 0 samples for what we want; here we exclude samples within-tars
        meta = sample_dict["meta"]
        if not self.mindeye:
            if self.split=="train":
                if meta['ses']==1 and not meta['sub']==1:
                    return None
            elif self.split=="test":
                if not (meta['ses']==1 and not meta['sub']==1):
                    return None
        else:
            if self.sub is not None:
                if meta['sub'] != int(self.sub[-1]):
                    return None
            
            events = sample_dict["events"]
            nsd_onset_indices, nsd_ids = [], []
            for event in events:
                nsd_onset_indices.append(event['index'])
                nsd_ids.append(event['nsd_id'])
            nsd_onset_indices = np.array(nsd_onset_indices).astype(int)
            nsd_ids = np.array(nsd_ids).astype(int) - 1 # because it's originally 1-indexed
            
            if len(nsd_onset_indices)==0:
                return None
            elif self.split=="test":
                if not np.any(np.isin(nsd_ids, shared1000)):
                    return None
            
        img = sample_dict["bold"]
        offset = sample_dict["offset"]
        global_signal = sample_dict["global_signal"]
        beta = sample_dict["beta"]
        mean = sample_dict["mean"]
        std = sample_dict["std"]

        img = img / 255.0
        if not self.gsr:
            img = (img - 0.5) / 0.2
            img = mean + std * img
            img = img + global_signal[:, None] * beta + offset

            session_mean = img.mean(axis=0)
            session_std = img.std(axis=0)
            img = (img - session_mean[None]) / session_std[None]

        if not self.mindeye:
            if self.fix_start!=-1:
                starts = np.arange(self.fix_start, self.same_run_samples)
            else:
                offset_start = np.random.choice(np.arange(self.frames - 1))
                starts = np.random.choice(np.arange(offset_start, len(img) - self.frames, self.frames), size=self.same_run_samples, replace=False)
    
            clips_list = []
            for start in starts:
                clip = to_tensor(img[start:start + self.frames], mask=self.mask, gsr=self.gsr)
                clips_list.append(clip)
            clips = torch.stack(clips_list, dim=0)

            return clips, meta

        elif self.mindeye:
            if self.split=="train":
                nsd_onset_indices_split = nsd_onset_indices[~np.isin(nsd_ids, shared1000)]
                nsd_ids_split = nsd_ids[~np.isin(nsd_ids, shared1000)]
            elif self.split=="test":
                nsd_onset_indices_split = nsd_onset_indices[np.isin(nsd_ids, shared1000)]
                nsd_ids_split = nsd_ids[np.isin(nsd_ids, shared1000)]
            else:
                nsd_onset_indices_split = nsd_onset_indices
                nsd_ids_split = nsd_ids

            if self.resampled:
                random_pick = np.random.randint(len(nsd_ids_split))
            else: # return ALL samples; your batch_size should be 1 because now the amount of samples returned is variable which otherwise messes with webloader concatenation
                random_pick = np.arange(len(nsd_ids_split))
            
            nsd_id = nsd_ids_split[random_pick]
            chosen_onset = nsd_onset_indices_split[random_pick]
            starts = chosen_onset + self.mindeye_TR_delay

            if self.resampled:
                start = starts
                clips = to_tensor(img[start:start + self.frames], mask=self.mask, gsr=self.gsr).unsqueeze(0)
            else:
                clips_list = []
                for start in starts:
                    clip = to_tensor(img[start:start + self.frames], mask=self.mask, gsr=self.gsr)
                    clips_list.append(clip)
                clips = torch.stack(clips_list, dim=0)
            
            return clips, meta, nsd_id #, misc            


    def get_dataset(self):
        if self.resampled:
            dataset = wds.WebDataset(self.tar_files,
                                resampled=NSD_NUM_SHARDS, # enables shard shuffling and infinite iterator, NOT sample shuffling
                                nodesplitter=wds.split_by_node,
                                workersplitter=wds.split_by_worker)
        else:
            dataset = wds.WebDataset(self.tar_files,
                                resampled=False,
                                nodesplitter=wds.split_by_node,
                                workersplitter=wds.split_by_worker)
        # note that resampled=False doesnt mean all unique samples will be provided and then iterator stops; it instead means every tar will be sampled exactly once

        dataset = dataset.decode() # Use simple .decode() for auto-decoding

        # Pull data from the decoded sample into a dictionary
        dataset = dataset.map(self._pull_data)
        dataset = dataset.select(lambda x: x is not None)

        # Process each sample (normalization, frame selection)
        dataset = dataset.map(self._process_sample)
        dataset = dataset.select(lambda x: x is not None)

        return dataset


##### HCP #####
# HCP-specific constants
HCP_FLAT_ROOT = "https://huggingface.co/datasets/bold-ai/HCP-Flat/resolve/main"
HCP_NUM_SHARDS = 1803

# Tasks and conditions used in prior works (Zhang, 2021; Rastegarnia, 2023)
INCLUDE_TASKS = ["EMOTION", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"]
# full list: ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'MOVIE1', 'MOVIE2', 'MOVIE3', 'MOVIE4', 'RELATIONAL', 'REST', 'RETBAR1', 'RETBAR2', 'RETCCW', 'RETCON', 'RETCW', 'RETEXP', 'SOCIAL', 'WM']
INCLUDE_CONDS = [
    "fear", "neut", "math", "story", "lf", "lh", "rf", "rh", "t", "match", "relation",
    "mental", "rnd", "0bk_body", "2bk_body", "0bk_faces", "2bk_faces", "0bk_places",
    "2bk_places", "0bk_tools", "2bk_tools",
]

HCP_TR = {"3T": 0.72, "7T": 1.0}
DEFAULT_DELAY_SECS = 4 * 0.72

def load_hcp_flat_mask(folder="/weka/proj-medarc/shared/HCP-Flat/") -> torch.Tensor:
    mask = np.load(os.path.join(folder, "hcp-flat_mask.npy"))
    mask = torch.as_tensor(mask)
    return mask

class HCPDataset:
    def __init__(
        self,
        split: str = "all",
        root: str = None,
        frames: int = 16,
        same_run_samples: int = 16,
        gsr: Optional[bool] = True,
        clip_mode: Literal["seq", "event"] = "seq",
        fix_start: Optional[int] = -1,
        cond_subset: Optional[bool] = False,
        train_split: float = .9
    ):
        self.split = split
        self.root = root
        self.frames = frames
        self.same_run_samples = same_run_samples
        self.gsr = gsr
        self.clip_mode = clip_mode
        self.mask = load_hcp_flat_mask(root)
        self.fix_start = fix_start
        self.len_unique_samples = 21633 # approx. HCP_NUM_SHARDS * 12
        self.sub_list = []
        self.cond_subset = cond_subset
        self.tar_files = self._build_tar_list(train_split=train_split)

    def _build_tar_list(self,train_split=.9):
        tar_files = []
        sample_count = 0

        # If root is a local path, use Path.rglob; if URL, assume pre-sharded structure
        if self.root.startswith("http"):
            shard_range = range(HCP_NUM_SHARDS)
            tar_files = [f"{self.root}/tars/hcp-flat_{shard:06d}.tar" for shard in shard_range]
            return tar_files
        else:
            # train/test split is based no tar files to be most computationally efficient, 
            # this is ok because the tars were created already shuffling the samples
            tar_files = sorted([str(path) for path in Path(self.root).rglob("*.tar")])
            if self.split=="train":
                tar_files = tar_files[:int(len(tar_files)*train_split)]
                if train_split != .9:
                    self.len_unique_samples = int(19464 * (train_split/.9))
                else:
                    self.len_unique_samples = 19464
            elif self.split=="test":
                tar_files = tar_files[int(len(tar_files)*.9):]
                self.len_unique_samples = 2169
            return tar_files
        
        return tar_files

    def _pull_data(self, sample):
        bold = sample["bold.npy"]
        meta = sample["meta.json"]
        events = sample["events.json"]
        misc = sample["misc.npz"]
        offset = misc["offset"]
        global_signal = misc["global_signal"]
        beta = misc["beta"]
        mean = misc["mean"]
        std = misc["std"]

        return {
            "bold": bold,
            "meta": meta,
            "events": events,
            "offset": offset,
            "global_signal": global_signal,
            "beta": beta,
            "mean": mean,
            "std": std,
        }

    def _process_sample(self, sample_dict):
        if sample_dict is None: return None  # Skip samples that failed to load

        if self.cond_subset:
            events = sample_dict["events"]
            if events == []:
                return None
            else:
                if events[-5]['trial_type'] != events[-1]['trial_type']:
                    return None
                elif not events[0]['trial_type'] in INCLUDE_CONDS:
                    return None
                else: 
                    trial_type = events[0]['trial_type']
            

        # if not meta in INCLUDE_TASKS:
        #     return None
        
        img = sample_dict["bold"]
        meta = sample_dict["meta"]
        offset = sample_dict["offset"]
        global_signal = sample_dict["global_signal"]
        beta = sample_dict["beta"]
        mean = sample_dict["mean"]
        std = sample_dict["std"]

        # Apply GSR or reverse normalization
        img = img / 255.0
        if not self.gsr:
            img = (img - 0.5) / 0.2
            img = mean + std * img
            img = img + global_signal[:, None] * beta + offset
            session_mean = img.mean(axis=0)
            session_std = img.std(axis=0)
            img = (img - session_mean[None]) / session_std[None]

        # Determine start indices for clips
        if self.fix_start != -1:
            starts = np.arange(self.fix_start, self.fix_start + self.same_run_samples)
        else:
            offset_start = np.random.choice(np.arange(self.frames - 1))
            TRs = np.arange(offset_start, len(img) - self.frames, self.frames)

            while len(TRs) < self.same_run_samples: # sometimes the samples have too few total TRs
                new_offset = (offset_start + len(TRs)) % self.frames
                TRs = np.concatenate([TRs, np.arange(new_offset, len(img) - self.frames, self.frames)])
                TRs = np.unique(TRs) # remove possible duplicates and sort it
            
            starts = np.random.choice(
                TRs,
                size=self.same_run_samples,
                replace=False,
            )

        clips_list = []
        for start in starts:
            clip = to_tensor(img[start:start + self.frames], mask=self.mask, gsr=self.gsr)
            clips_list.append(clip)
        clips = torch.stack(clips_list, dim=0)

        if self.cond_subset:
            return clips, meta, trial_type
        else: 
            return clips, meta

    def get_dataset(self):
        dataset = wds.WebDataset(
            self.tar_files,
            resampled=HCP_NUM_SHARDS,  # Enables shard shuffling, not sample shuffling
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )

        dataset = dataset.decode()  # Auto-decode all sample types
        dataset = dataset.map(self._pull_data)
        dataset = dataset.select(lambda x: x is not None)

        # Process each sample (normalization, frame selection)
        dataset = dataset.map(self._process_sample)
        dataset = dataset.select(lambda x: x is not None)

        return dataset