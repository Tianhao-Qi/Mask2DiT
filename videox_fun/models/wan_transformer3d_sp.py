import math
import inspect
from typing import Any, Dict

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.distributed as dist
from diffusers.utils import is_torch_version

from ..dist import get_sp_group as dist_get_sp_group
from .sp_comm import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group, set_pad, get_pad, split_sequence, gather_sequence, all_to_all_comm
from .wan_transformer3d import attention, sinusoidal_embedding_1d, WanRMSNorm, WanTransformer3DModel
from diffusers.configuration_utils import register_to_config


class SequenceParallelWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    @staticmethod
    def pad_freqs(original_tensor, target_len):
        seq_len, s1, s2 = original_tensor.shape
        pad_size = target_len - seq_len
        padding_tensor = torch.ones(
            pad_size,
            s1,
            s2,
            dtype=torch.int64,
            device=original_tensor.device)
        padding_tensor = padding_tensor.to(torch.complex64)
        padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
        return padded_tensor

    def rope_apply(self, x, grid_sizes, freqs):
        """
        x:          [B, L, N, C].
        grid_sizes: [B, 3].
        freqs:      [M, C // 2].
        """
        s, n, c = x.size(1), x.size(2), x.size(3) // 2
        # split freqs
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # loop over samples
        # TODO: remove loop
        output = []
        sp_size = get_sequence_parallel_world_size()
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            
            # precompute multipliers
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                                dim=-1).reshape(seq_len, 1, -1)

            # apply rotary embedding
            if sp_size > 1:
                x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
                    s, n, -1, 2))
                sp_rank = get_sequence_parallel_rank()
                freqs_i = self.pad_freqs(freqs_i, s * sp_size)
                s_per_rank = s
                freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                            s_per_rank), :, :]
                x_i = torch.view_as_real(x_i * freqs_i_rank.to(torch.complex64)).flatten(2)
                x_i = torch.cat([x_i, x[i, s:]])
            else:
                x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
                    seq_len, n, -1, 2))
                x_i = torch.view_as_real(x_i * freqs_i.to(torch.complex64)).flatten(2)
                x_i = torch.cat([x_i, x[i, seq_len:]])

            # append to collection
            output.append(x_i)
        return torch.stack(output).to(x.dtype)

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype):
        """
        x:         [B, L, D]
        seq_lens:  [B]
        freqs:     [1024, D / (2 * num_heads)]
        """
        B, L, D = x.shape
        N, d = self.num_heads, self.head_dim
        assert D == N * d

        sp_group = get_sp_group()
        sp_size = get_sequence_parallel_world_size()

        q = self.norm_q(self.q(x.to(dtype))).view(B, L, N, d)
        k = self.norm_k(self.k(x.to(dtype))).view(B, L, N, d)
        v = self.v(x.to(dtype)).view(B, L, N, d)

        q = self.rope_apply(q, grid_sizes, freqs).to(dtype)
        k = self.rope_apply(k, grid_sizes, freqs).to(dtype)

        if sp_size > 1:
            q, k, v = map(
                lambda x: all_to_all_comm(x, sp_group, scatter_dim=2, gather_dim=1),
                [q, k, v]
            )
        out = attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size
        ).to(dtype)  # shape: [B, L, heads_local, d]

        if sp_size > 1:
            out = all_to_all_comm(out, sp_group, scatter_dim=1, gather_dim=2)

        out = out.flatten(2)  # [B, L, D]
        return self.o(out)


class SequenceParallelWanI2VCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, dtype):
        """
        x:        [B, L1, C]        (sequence tokens, already split along seq dim)
        context:  [B, L2 + 257, C]  (CLIP image token + text token)
        context_lens: [B]
        """
        B, _, D = x.shape
        N, d = self.num_heads, self.head_dim
        sp_group = get_sp_group()
        sp_size = get_sequence_parallel_world_size()

        # split context
        context_img = context[:, :257]          # CLIP token
        context_text = context[:, 257:]         # prompt token

        # q from input token
        q = self.norm_q(self.q(x.to(dtype))).view(B, -1, N, d)

        # k/v from text
        k = self.norm_k(self.k(context_text.to(dtype))).view(B, -1, N, d)
        v = self.v(context_text.to(dtype)).view(B, -1, N, d)

        # k/v from image
        k_img = self.norm_k_img(self.k_img(context_img.to(dtype))).view(B, -1, N, d)
        v_img = self.v_img(context_img.to(dtype)).view(B, -1, N, d)

        # head parallel all_to_all
        if sp_size > 1:
            q, k, v, k_img, v_img = map(
                lambda x: all_to_all_comm(x, sp_group, scatter_dim=2, gather_dim=1),
                [q, k, v, k_img, v_img]
            )

        # image attention
        img_x = attention(q, k_img, v_img, k_lens=None)

        # text attention
        txt_x = attention(q, k, v, k_lens=context_lens)

        # total
        out = (img_x + txt_x).to(dtype)

        # gather heads back
        if sp_size > 1:
            out = all_to_all_comm(out, sp_group, scatter_dim=1, gather_dim=2)
        out = out.flatten(2)
        return self.o(out)


class WanTransformer3DSPModel(WanTransformer3DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 覆盖 self.blocks 中的 attention
        for i, block in enumerate(self.blocks):
            # 替换 self attention
            block.self_attn = SequenceParallelWanSelfAttention(
                dim=self.dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                qk_norm=self.qk_norm,
                eps=self.eps
            )

            # 替换 cross attention
            cross_attn_type = self.model_type
            if cross_attn_type == "i2v":
                block.cross_attn = SequenceParallelWanI2VCrossAttention(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    qk_norm=self.qk_norm,
                    eps=self.eps
                )


class WanTransformer3DSPModel(WanTransformer3DModel):
    @register_to_config
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 覆盖 self.blocks 中的 attention
        for i, block in enumerate(self.blocks):
            # 替换 self attention
            block.self_attn = SequenceParallelWanSelfAttention(
                dim=block.dim,
                num_heads=block.num_heads,
                window_size=block.window_size,
                qk_norm=block.qk_norm,
                eps=block.eps
            )

            # 替换 cross attention
            cross_attn_type = self.model_type
            if cross_attn_type == "i2v":
                block.cross_attn = SequenceParallelWanI2VCrossAttention(
                    dim=block.dim,
                    num_heads=block.num_heads,
                    window_size=block.window_size,
                    qk_norm=block.qk_norm,
                    eps=block.eps
                )

    # Set the __signature__ attribute to match Parent.__init__
    __init__.__signature__ = inspect.signature(WanTransformer3DModel.__init__)

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        cond_flag=True,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # add control adapter
        if self.control_adapter is not None and y_camera is not None:
            y_camera = self.control_adapter(y_camera)
            x = [u + v for u, v in zip(x, y_camera)]

        sp_group = get_sp_group()
        sp_size = get_sequence_parallel_world_size()

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        if self.ref_conv is not None and full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += full_ref.size(1)
            x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref, x)]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        if sp_size > 1:
            set_pad("pad", x.shape[1], sp_group)
            x = split_sequence(x, sp_group, dim=1, pad=get_pad("pad"))

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
        
        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = should_calc
            else:
                should_calc = self.teacache.should_calc
        
        # TeaCache
        if self.teacache is not None:
            if not should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype
                        )
                        x = block(x, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        dtype,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype
                    )
                    x = block(x, **kwargs)

        if self.sp_world_size > 1:
            x = dist_get_sp_group().all_gather(x, dim=1)

        if sp_size > 1:
            x = gather_sequence(x, sp_group, dim=1, pad=get_pad("pad"))

        if self.ref_conv is not None and full_ref is not None:
            full_ref_length = full_ref.size(1)
            x = x[:, full_ref_length:]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x