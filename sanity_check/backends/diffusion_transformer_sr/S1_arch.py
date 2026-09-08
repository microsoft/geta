import math
import numbers

import torch
import torch.nn.functional as F
from torch import nn


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
            self.body = BiasFree_LayerNorm(dim)
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

        self.c_fix = dim // self.num_heads
        self.c_out = dim

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        c_fix = c // self.num_heads
        c_fix = self.c_fix
        q = q.view(b, self.num_heads, c_fix, h * w)
        k = k.view(b, self.num_heads, c_fix, h * w)
        v = v.view(b, self.num_heads, c_fix, h * w)

        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)
        q = custom_normalize(q, dim=-1)
        k = custom_normalize(k, dim=-1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        # out = out.view(b, c, h, w)
        out = out.view(b, self.c_out, h, w)

        out = self.project_out(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kernel = nn.Sequential(
            nn.Linear(256, dim * 2, bias=False),
        )

        self.conv_mha = ConvMHA(dim, num_heads, bias)
        # self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, k_v):
        b, c, h, w = x.shape
        k_v = self.kernel(k_v).view(-1, c * 2, 1, 1)
        k_v1, k_v2 = k_v.chunk(2, dim=1)
        x = x * k_v1 + k_v2

        out = self.conv_mha(x)
        return out


class DiffTransformerBlock(nn.Module):
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
            # TC added to break dependency for PixelShuffle
            nn.Conv2d(
                n_feat * 2, n_feat * 2, kernel_size=1, stride=1, padding=0, bias=False
            ),
        )

    def forward(self, x):
        return self.body(x)


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
            # inp_channels =12
            self.pixel_unshuffle = nn.PixelUnshuffle(2)
        elif self.scale == 1:
            # inp_channels =48
            self.pixel_unshuffle = nn.PixelUnshuffle(4)
        # else:
        #     inp_channels =3
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                DiffTransformerBlock(
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
                DiffTransformerBlock(
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
                DiffTransformerBlock(
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
                DiffTransformerBlock(
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
        LayerNorm_type="WithBias",  ## Other option 'BiasFree',
        num_refinement_blocks=4,
    ):

        super().__init__()

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decoder_level3 = nn.Sequential(
            *[
                DiffTransformerBlock(
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
                DiffTransformerBlock(
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
                DiffTransformerBlock(
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
            int(dim * 2), int(dim * 2), kernel_size=1, padding=0, stride=1
        )
        # self.refinement = nn.Sequential(*[DiffTransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

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

        out_dec_level1 = self.dep_breaker(out_dec_level1)
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
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

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
        )

        modules_tail = [
            Upsampler(default_conv, 4, int(dim * 2**1), act=False),
            default_conv(int(dim * 2**1), out_channels, 3),
        ]

        self.refinement = nn.Sequential(
            *[
                DiffTransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        self.tail = nn.Sequential(*modules_tail)

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

        out_dec_level1, _ = self.refinement([out_dec_level1, k_v])
        out_dec_level1 = self.tail(out_dec_level1) + F.interpolate(
            inp_img, scale_factor=self.scale, mode="nearest"
        )

        return out_dec_level1


class CPEN(nn.Module):
    def __init__(self, n_feats=64, n_encoder_res=6, scale=4):
        super().__init__()
        self.scale = scale
        if scale == 2:
            E1 = [
                nn.Conv2d(60, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
            ]
        elif scale == 1:
            E1 = [
                nn.Conv2d(96, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
            ]
        else:
            E1 = [
                nn.Conv2d(51, n_feats, kernel_size=3, padding=1),
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

    def forward(self, x, gt):
        gt0 = self.pixel_unshuffle(gt)
        if self.scale == 2:
            feat = self.pixel_unshufflev2(x)
        elif self.scale == 1:
            feat = self.pixel_unshuffle(x)
        else:
            feat = x
        x = torch.cat([feat, gt0], dim=1)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        S1_IPR = []
        fea1 = self.mlp(fea)
        S1_IPR.append(fea1)
        return fea1, S1_IPR


class DiffIRS1(nn.Module):
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

        self.E = CPEN(n_feats=64, n_encoder_res=n_encoder_res, scale=scale)

    def forward(self, x, gt):

        if self.training:
            IPRS1, S1_IPR = self.E(x, gt)

            sr = self.G(x, IPRS1)

            return sr, S1_IPR
        else:
            IPRS1, _ = self.E(x, gt)

            sr = self.G(x, IPRS1)

            return sr
