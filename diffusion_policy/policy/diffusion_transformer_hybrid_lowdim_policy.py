# diffusion_policy/policy/diffusion_transformer_hybrid_lowdim_policy.py

from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionTransformerHybridLowdimPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # architecture
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            # passthrough
            **kwargs):
        super().__init__()

        # --- parse shape_meta, only low_dim supported ---
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1, "Action must be 1D"
        action_dim = action_shape[0]

        # compute total dimension of low-dim observations
        obs_feature_dim = 0
        for key, attr in shape_meta['obs'].items():
            t = attr.get('type', 'low_dim')
            assert t == 'low_dim', f"Only low_dim supported, got '{t}' for obs key '{key}'"
            d = attr['shape'][0]
            obs_feature_dim += d

        # a trivial encoder for low-dim: identity
        obs_encoder = nn.Identity()

        # build the diffusion transformer
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        # mask generator for low-dim
        mask_gen = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        # normalizer for low-dim + action
        normalizer = LinearNormalizer()

        # save
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = mask_gen
        self.normalizer = normalizer
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        # inference steps
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            **kwargs):
        """
        Standard DDIM sampling loop, applied to low-dim trajectories.
        """
        # start from pure noise
        trajectory = torch.randn_like(condition_data, generator=generator)
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # enforce conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            # predict residual
            model_out = self.model(trajectory, t, cond)
            # step backwards
            trajectory = scheduler.step(
                model_out, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        # final enforce
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Given a dict of low-dim observations, return predicted actions.
        """
        # normalize obs & actions (we won't use past_action here)
        nobs = self.normalizer.normalize(obs_dict)
        # prepare action placeholder for conditioning
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps
        Da = self.action_dim
        T = self.horizon

        # flatten and encode obs if needed
        if self.obs_as_cond:
            # concatenate all low-dim keys along the last dim
            # shape: (B, To, obs_feature_dim)
            concat = torch.cat([nobs[k][:,:To] for k in nobs], dim=-1)
            cond = concat
            # prepare trajectory slots
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(shape, device=cond.device, dtype=cond.dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # not typical for lowdim; skip
            cond = None
            cond_data = None
            cond_mask = None

        # sample
        sample = self.conditional_sample(cond_data, cond_mask, cond=cond, **self.kwargs)

        # unnormalize
        naction = sample[..., :Da]
        action = self.normalizer['action'].unnormalize(naction)

        if not self.pred_action_steps_only:
            start = To - 1
            end = start + self.n_action_steps
            action = action[:, start:end]

        return {'action': action, 'action_pred': action}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        """
        The same optimizer grouping as the hybrid image version:
        - Transformer params
        - (trivial) obs_encoder params
        """
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def compute_loss(self, batch):
        """
        MSE loss on the diffused trajectory, identical to image version.
        """
        # normalize
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B, H = nactions.shape[0], nactions.shape[1]
        To = self.n_obs_steps
        Da = self.action_dim

        # build conditioning and trajectory
        if self.obs_as_cond:
            concat = torch.cat([nobs[k][:,:To] for k in nobs], dim=-1)
            cond = concat
            trajectory = nactions[:, To-1:To-1 + self.n_action_steps] \
                         if self.pred_action_steps_only \
                         else nactions
        else:
            cond = None
            trajectory = torch.cat([nactions, concat], dim=-1)

        # mask for conditioning
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # forward diffusion
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=trajectory.device).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # apply conditioning
        noisy[condition_mask] = trajectory[condition_mask]

        # predict
        pred = self.model(noisy, timesteps, cond)
        pred_type = self.noise_scheduler.config.prediction_type
        target = noise if pred_type == 'epsilon' else trajectory

        # compute masked loss
        loss_mask = ~condition_mask
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
        return loss
