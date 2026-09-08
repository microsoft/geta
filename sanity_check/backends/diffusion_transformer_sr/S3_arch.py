import math
import numbers
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def exists(x):
    return x is not None


def to_3d(x):
    b, c, h, w = x.shape
    x_permuted = x.permute(0, 2, 3, 1)
    return x_permuted.contiguous().view(b, h * w, c)


def to_4d(x, h, w):
    b, _, c = x.shape
    x_permuted = x.permute(0, 2, 1)
    return x_permuted.contiguous().view(b, c, h, w)


def custom_normalize(tensor, dim=-1, eps=1e-12):
    norm = tensor.pow(2).sum(dim=dim, keepdim=True).sqrt() + eps
    return tensor / norm


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.LeakyReLU(0.1, inplace=True),
        res_scale=1,
    ):

        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):
        m = []
        if (int(scale) & (int(scale) - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super().__init__(*m)


class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        denoise,
        condition,
        timesteps=1000,
        beta_schedule="linear",
        image_size=256,
        n_feats=128,
        clip_denoised=False,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        parameterization="x0",  # all assuming fixed variance schedules
    ):
        super().__init__()
        assert parameterization in ["eps", "x0"], (
            'currently only supporting "eps" and "x0"'
        )
        self.parameterization = parameterization
        print(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )
        # self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.image_size = image_size  # try conv?
        self.channels = n_feats
        self.model = denoise
        self.condition = condition

        self.v_posterior = v_posterior
        self.l_simple_weight = l_simple_weight

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, (
            "alphas have to be defined for each timestep"
        )

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, c, clip_denoised: bool):
        model_out = self.model(x, t, c)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, model_out

    def p_sample(self, x, t, c, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, predicted_noise = self.p_mean_variance(
            x=x, t=t, c=c, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean, predicted_noise

    def forward(self, img, x=None):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        device = self.betas.device
        b = img.shape[0]

        shape = (img.shape[0], self.channels * 4)
        # x_noisy = torch.randn(shape, device=device)
        x_noisy = torch.zeros(shape, device=device)
        c = self.condition(img)
        IPR = x_noisy
        for i in reversed(range(self.num_timesteps)):
            IPR, _ = self.p_sample(
                IPR,
                torch.full((b,), i, device=device, dtype=torch.long),
                c,
                clip_denoised=self.clip_denoised,
            )
        return IPR


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class BiasFree_CIP_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        # if isinstance(normalized_shape, numbers.Integral):
        #     normalized_shape = (normalized_shape,)
        # normalized_shape = torch.Size(normalized_shape)
        # assert len(normalized_shape) == 1
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.num_samples = normalized_shape[0]
        self.weight = nn.Parameter(torch.ones(self.num_samples))
        # self.weight.data.copy_(base_layer_norm.weight.data)
        # self.weight = nn.Parameter(torch.ones(normalized_shape))
        # self.normalized_shape = normalized_shape
        # self.num_samples = self.normalized_shape[0]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean_input = torch.sum(input, dim=-1) / self.num_samples
        rstd_val = torch.rsqrt(
            torch.sum(input.pow(2), dim=-1) / self.num_samples - mean_input.pow(2)
        )
        # return (input * rstd_val.unsqueeze(-1) - mean_input.unsqueeze(-1) * rstd_val.unsqueeze(-1)) * self.weight + self.bias
        return (input * rstd_val.unsqueeze(-1)) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            # self.body = BiasFree_LayerNorm(dim)
            self.body = BiasFree_CIP_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(256, dim * 2, bias=False),
        )

    def forward(self, x, k_v):
        b, c, h, w = x.shape
        k_v = self.kernel(k_v).view(-1, c * 2, 1, 1)
        k_v1, k_v2 = k_v.chunk(2, dim=1)
        x = x * k_v1 + k_v2
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class ConvMHA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.c_fix = self.head_dim
        # self.c_out = dim

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        q = custom_normalize(q, dim=-1)
        k = custom_normalize(k, dim=-1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        # out = out.view(b, self.c_out, h, w)
        out = out.view(b, self.num_heads * self.head_dim, h, w)

        out = self.project_out(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kernel = nn.Sequential(
            nn.Linear(256, dim * 2, bias=False),
        )
        self.conv_mha = ConvMHA(dim, num_heads, bias)

    def forward(self, x, k_v):
        b, c, h, w = x.shape
        k_v = self.kernel(k_v).view(-1, c * 2, 1, 1)
        k_v1, k_v2 = k_v.chunk(2, dim=1)
        x = x * k_v1 + k_v2

        return self.conv_mha(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        k_v = y[1]
        x = x + self.attn(self.norm1(x), k_v)
        x = x + self.ffn(self.norm2(x), k_v)

        return [x, k_v]


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )
        self.dep_breaker = nn.Conv2d(
            n_feat * 2, n_feat * 2, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.dep_breaker.weight.data.zero_()
        for t in range(2 * n_feat):
            self.dep_breaker.weight.data[t, t, 0, 0] = 1.0

    def forward(self, x):
        return self.dep_breaker(self.body(x))
        # return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )
        self.dep_breaker = nn.Conv2d(
            n_feat // 2, n_feat // 2, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.dep_breaker.weight.data.zero_()
        for t in range(n_feat // 2):
            self.dep_breaker.weight.data[t, t, 0, 0] = 1.0

    def forward(self, x):
        return self.dep_breaker(self.body(x))


class DIRformerDownPath(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        scale=4,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
    ):
        super().__init__()
        self.scale = scale
        if self.scale == 2:
            inp_channels = 12
            self.pixel_unshuffle = nn.PixelUnshuffle(2)
        elif self.scale == 1:
            inp_channels = 48
            self.pixel_unshuffle = nn.PixelUnshuffle(4)
        else:
            inp_channels = 3

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )

    def forward(self, feat, k_v):
        inp_enc_level1 = self.patch_embed(feat)

        out_enc_level1, _ = self.encoder_level1([inp_enc_level1, k_v])

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, _ = self.encoder_level2([inp_enc_level2, k_v])

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, _ = self.encoder_level3([inp_enc_level3, k_v])

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent, _ = self.latent([inp_enc_level4, k_v])

        return out_enc_level1, out_enc_level2, out_enc_level3, latent


class DIRformerUpPath(nn.Module):
    def __init__(
        self,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
        num_refinement_blocks=4,
        out_channels=3,
    ):

        super().__init__()

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.dep_breaker = nn.Conv2d(
            int(dim * 2), int(dim * 2), kernel_size=1, padding=0, stride=1, bias=False
        )  # TC added
        self.dep_breaker.weight.data.zero_()
        for t in range(dim * 2):
            self.dep_breaker.weight.data[t, t, 0, 0] = 1.0

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        modules_tail = [
            Upsampler(default_conv, 4, int(dim * 2**1), act=False),
            default_conv(int(dim * 2**1), out_channels, 3),
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, out_enc_level1, out_enc_level2, out_enc_level3, latent, k_v):
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, _ = self.decoder_level3([inp_dec_level3, k_v])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, _ = self.decoder_level2([inp_dec_level2, k_v])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.dep_breaker(inp_dec_level1)
        out_dec_level1, _ = self.decoder_level1([inp_dec_level1, k_v])

        out_dec_level1, _ = self.refinement([out_dec_level1, k_v])
        out_dec_level1 = self.tail(out_dec_level1)
        return out_dec_level1


class DIRformer(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        scale=4,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
    ):

        super().__init__()
        self.scale = scale
        if self.scale == 2:
            inp_channels = 12
            self.pixel_unshuffle = nn.PixelUnshuffle(2)
        elif self.scale == 1:
            inp_channels = 48
            self.pixel_unshuffle = nn.PixelUnshuffle(4)
        else:
            inp_channels = 3
        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.down_path = DIRformerDownPath(
            inp_channels=inp_channels,
            scale=scale,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        self.up_path = DIRformerUpPath(
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,  ## Other option 'BiasFree'
            num_refinement_blocks=num_refinement_blocks,
            out_channels=out_channels,
        )

        # self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        # modules_tail = [Upsampler(default_conv, 4, int(dim*2**1), act=False),
        #                 default_conv(int(dim*2**1), out_channels, 3)]
        # self.tail = nn.Sequential(*modules_tail)

    def forward(self, inp_img, k_v):
        if self.scale == 2 or self.scale == 1:
            feat = self.pixel_unshuffle(inp_img)
        else:
            feat = inp_img

        # inp_enc_level1 = self.patch_embed(feat)
        out_enc_level1, out_enc_level2, out_enc_level3, latent = self.down_path(
            feat, k_v
        )
        # out_enc_level1, out_enc_level2, out_enc_level3, latent = self.down_path(inp_enc_level1, k_v)
        out_dec_level1 = self.up_path(
            out_enc_level1, out_enc_level2, out_enc_level3, latent, k_v
        )

        # out_dec_level1 = self.tail(out_dec_level1)
        out_dec_level1 = out_dec_level1 + F.interpolate(
            inp_img, scale_factor=self.scale, mode="nearest"
        )

        # out_dec_level1,_ = self.refinement([out_dec_level1,k_v])
        # out_dec_level1 = self.tail(out_dec_level1) + F.interpolate(inp_img, scale_factor=self.scale, mode='nearest')

        return out_dec_level1


class CPEN(nn.Module):
    def __init__(self, n_feats=64, n_encoder_res=6, scale=4):
        super().__init__()
        self.scale = scale
        if scale == 2:
            E1 = [
                nn.Conv2d(12, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
            ]
        elif scale == 1:
            E1 = [
                nn.Conv2d(48, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
            ]
        else:
            E1 = [
                nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
            ]

        E2 = [
            ResBlock(default_conv, n_feats, kernel_size=3) for _ in range(n_encoder_res)
        ]
        E3 = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E = E1 + E2 + E3
        self.E = nn.Sequential(*E)
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.pixel_unshufflev2 = nn.PixelUnshuffle(2)

    def forward(self, x):
        if self.scale == 2:
            feat = self.pixel_unshufflev2(x)
        elif self.scale == 1:
            feat = self.pixel_unshuffle(x)
        else:
            feat = x
        fea = self.E(feat).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)

        return fea1


class ResMLP(nn.Module):
    def __init__(self, n_feats=512):
        super().__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats, n_feats),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        res = self.resmlp(x)
        return res


class denoise(nn.Module):
    def __init__(self, n_feats=64, n_denoise_res=5, timesteps=5):
        super().__init__()
        self.max_period = timesteps * 10
        n_featsx4 = 4 * n_feats
        resmlp = [
            nn.Linear(n_featsx4 * 2 + 1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp = nn.Sequential(*resmlp)

    def forward(self, x, t, c):
        t = t.float()
        t = t / self.max_period
        t = t.view(-1, 1)
        c = torch.cat([c, t, x], dim=1)

        fea = self.resmlp(c)

        return fea


class DiffIRS3SNPE(nn.Module):
    def __init__(
        self,
        n_encoder_res=9,
        inp_channels=3,
        out_channels=3,
        scale=1,
        dim=64,
        num_blocks=[13, 1, 1, 1],
        num_refinement_blocks=13,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.2,
        bias=False,
        LayerNorm_type="BiasFree",  ## Other option 'BiasFree'
        n_denoise_res=1,
        linear_start=0.1,
        linear_end=0.99,
        timesteps=4,
    ):
        super().__init__()

        # Generator
        self.G = DIRformer(
            inp_channels=inp_channels,
            out_channels=out_channels,
            scale=scale,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,  ## Other option 'BiasFree'
        )
        self.condition = CPEN(n_feats=64, n_encoder_res=n_encoder_res, scale=scale)

        self.denoise = denoise(
            n_feats=64, n_denoise_res=n_denoise_res, timesteps=timesteps
        )

        self.diffusion = DDPM(
            denoise=self.denoise,
            condition=self.condition,
            n_feats=64,
            linear_start=linear_start,
            linear_end=linear_end,
            timesteps=timesteps,
        )

    def forward(self, img, IPRS1=None):
        if self.training:
            # IPRS2, pred_IPR_list=self.diffusion(img,IPRS1)
            # sr = self.G(img, IPRS2)

            # return sr, pred_IPR_list

            IPRS2 = self.diffusion(img)
            sr = self.G(img, IPRS2)
        else:
            IPRS2 = self.diffusion(img)
            sr = self.G(img, IPRS2)

        return sr
