"""
real_coffeepod_image_dataset.py
────────────────────────────────────────────────────────────────────────
Forked from real_pusht_image_dataset.py with minimal edits for xArm‑7 coffee‑pod:
  • Class renamed RealCoffeepodImageDataset
  • Accept only 7‑D action vectors (assert patched)
  • All other logic (caching, sampling, normalization) unchanged
"""

from typing import Dict, List
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats)

class RealCoffeepodImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            delta_action=False,
        ):
        assert os.path.isdir(dataset_path)
        
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore()
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception:
                        shutil.rmtree(cache_zarr_path)
                        raise
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        if replay_buffer is None:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore()
            )
        
        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = episode_ends[i-1] if i>0 else 0
                end = episode_ends[i]
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            t = attr.get('type','low_dim')
            if t=='rgb':
                rgb_keys.append(key)
            else:
                lowdim_keys.append(key)
        
        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask)
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32)/255.
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]
        
        action = data['action'].astype(np.float32)
        # only 7‑D actions: no slicing needed

        return {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }


def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[..., idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr


def _get_replay_buffer(dataset_path, shape_meta, store):
    # parse shape meta
    rgb_keys, lowdim_keys = [], []
    out_resolutions, lowdim_shapes = {}, {}
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        t = attr.get('type','low_dim')
        shape = tuple(attr.get('shape'))
        if t=='rgb':
            rgb_keys.append(key)
            _,h,w=shape
            out_resolutions[key] = (w,h)
        else:
            lowdim_keys.append(key)
            lowdim_shapes[key] = shape
            if 'pose' in key:
                assert shape in [(2,),(6,)]
    
    action_shape = tuple(shape_meta['action']['shape'])
    # only xArm‑7 7‑DOF action
    assert action_shape==(7,), f"unsupported action dim {action_shape}"

    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys+['action'],
            image_keys=rgb_keys)

    # no slicing for 7‑D
    for key, shape in lowdim_shapes.items():
        if 'pose' in key and shape==(2,):
            zarr_resize_index_last_dim(replay_buffer[key], idxs=[0,1])

    return replay_buffer


def test():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize('../diffusion_policy/config'):
        cfg = hydra.compose('train_diffusion_transformer_real_hybrid_workspace')
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'][:])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(diff, axis=-1)
    _ = plt.hist(dists, bins=100); plt.title('real action velocity')

