"""Parse config files and load appropriate model."""
import torch

from contextlib import nullcontext
import os

from .resnets import ResNet, resnet_depths_to_config
from .densenets import DenseNet, densenet_depths_to_config
from .nfnets import NFNet
from .vgg import VGG
from .pyramidnets import PyramidNet
from .convmixer import ConvMixer
from .mobilenet import MobileNetV2
from .vit_small import ViT


def construct_model(cfg_model, channels, classes):
    """cfg_model templates can be found under config/model."""
    report_invariance = False
    if "flipinvariantresnet" in cfg_model.name.lower():
        report_invariance = True
        block, layers = resnet_depths_to_config(cfg_model.depth)
        model = OutputAveragedResNet(
            block,
            layers,
            channels,
            classes,
            stem=cfg_model.stem,
            convolution_type=cfg_model.convolution,
            nonlin=cfg_model.nonlin_fn,
            norm=cfg_model.normalization,
            downsample=cfg_model.downsample,
            width_per_group=cfg_model.width,
            zero_init_residual=True if "skip_residual" in cfg_model.initialization else False,
        )
    elif "orbitresnet" in cfg_model.name.lower():
        report_invariance = True
        block, layers = resnet_depths_to_config(cfg_model.depth)
        model = OrbitMappedResNet(
            block,
            layers,
            channels,
            classes,
            stem=cfg_model.stem,
            convolution_type=cfg_model.convolution,
            nonlin=cfg_model.nonlin_fn,
            norm=cfg_model.normalization,
            downsample=cfg_model.downsample,
            width_per_group=cfg_model.width,
            zero_init_residual=True if "skip_residual" in cfg_model.initialization else False,
        )
    elif "resnet" in cfg_model.name.lower():
        block, layers = resnet_depths_to_config(cfg_model.depth)
        model = ResNet(
            block,
            layers,
            channels,
            classes,
            stem=cfg_model.stem,
            convolution_type=cfg_model.convolution,
            nonlin=cfg_model.nonlin_fn,
            norm=cfg_model.normalization,
            downsample=cfg_model.downsample,
            width_per_group=cfg_model.width,
            zero_init_residual=True if "skip_residual" in cfg_model.initialization else False,
        )
    elif "densenet" in cfg_model.name.lower():
        growth_rate, block_config, num_init_features = densenet_depths_to_config(cfg_model.depth)
        model = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bn_size=cfg_model.bn_size,
            drop_rate=cfg_model.drop_rate,
            channels=channels,
            num_classes=classes,
            memory_efficient=cfg_model.memory_efficient,
            norm=cfg_model.normalization,
            nonlin=cfg_model.nonlin_fn,
            stem=cfg_model.stem,
            convolution_type=cfg_model.convolution,
        )
    elif "vgg" in cfg_model.name.lower():
        model = VGG(
            cfg_model.name,
            in_channels=channels,
            num_classes=classes,
            norm=cfg_model.normalization,
            nonlin=cfg_model.nonlin_fn,
            head=cfg_model.head,
            convolution_type=cfg_model.convolution,
            drop_rate=cfg_model.drop_rate,
            classical_weight_init=cfg_model.classical_weight_init,
        )
    elif "linear" in cfg_model.name.lower():
        model = torch.nn.Sequential(torch.nn.Flatten(), _Select(100), torch.nn.Linear(100, classes))  # for debugging only
    elif "mlp" in cfg_model.name.lower():
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3072, cfg_model.width),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg_model.width, cfg_model.width),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg_model.width, cfg_model.width),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg_model.width, classes),
        )
    elif "convmixer" in cfg_model.name.lower():
        model = ConvMixer(
            cfg_model.dim,
            cfg_model.depth,
            channels=channels,
            kernel_size=cfg_model.kernel_size,
            patch_size=1,
            n_classes=classes,
        )
    elif "mobilenetv2" in cfg_model.name.lower():
        model = MobileNetV2(num_classes=classes, num_channels=channels)

    elif "nfnet" in cfg_model.name.lower():
        model = NFNet(
            channels,
            classes,
            variant=cfg_model.variant,
            stochdepth_rate=cfg_model.stochdepth_rate,
            alpha=cfg_model.alpha,
            se_ratio=cfg_model.se_ratio,
            activation=cfg_model.nonlin,
            stem=cfg_model.stem,
            use_dropout=cfg_model.use_dropout,
        )
    elif "pyramidnet" in cfg_model.name:
        model = PyramidNet(cfg_model.depth, cfg_model.alpha, channels, classes, bottleneck=cfg_model.bottleneck)
    elif "e2cnn-base" in cfg_model.name:
        from .e2cnnmodels import Wide_ResNet as e2cnn_WideResnet

        report_invariance = True
        model = e2cnn_WideResnet(cfg_model.depth, cfg_model.width_modifier, 0.0, initial_stride=1, N=2, f=True, r=0, num_classes=classes)
    elif "e2cnn-comp" in cfg_model.name:
        from .e2cnnmodels import ComparableResNet as e2cnn_ComaparableResnet

        report_invariance = True
        model = e2cnn_ComaparableResnet(cfg_model.depth, cfg_model.width, 0.0, initial_stride=1, N=2, f=True, r=0, num_classes=classes)
    elif "lieconv" in cfg_model.name:
        report_invariance = False  # breaks LieConv due to change in cached batch_size. Turn on only for testing
        from lie_conv.lieConv import ImgLieResnet
        from lie_conv.lieGroups import SO2

        model = ImgLieResnet(
            num_targets=classes,
            chin=channels,
            k=cfg_model.k,  # channel width for the network. Can be int (same for all) or array to specify individually.
            total_ds=cfg_model.total_ds,  # ds_frac = (total_ds)**(1/num_layers), ds_frac: total downsampling to perform throughout the layers of the net.
            fill=cfg_model.fill,  # specifies the fraction of the input which is included in local neighborhood.
            nbhd=cfg_model.nbhd,  # number of samples to use for Monte Carlo estimation (p)
            num_layers=cfg_model.num_layers,
            group=SO2(0.2),  # Chosen group to be equivariant to.
            increase_channels=False,  # handled via list of k
            act="relu",
        )
        ds_frac = (cfg_model.total_ds) ** (1 / cfg_model.num_layers)
        fill = [cfg_model.fill / ds_frac**i for i in range(cfg_model.num_layers)]
        print(f"LieResnet instantiated with ds_frac={ds_frac}, fill={fill}")
    elif "vit" in cfg_model.name.lower():
        model = ViT(
            image_size=32,
            patch_size=cfg_model.patch_size,
            num_classes=classes,
            channels=channels,
            dim=cfg_model.head_dim,
            depth=cfg_model.depth,
            heads=cfg_model.heads,
            mlp_dim=cfg_model.mlp_dim,
            dropout=cfg_model.dropout,
            emb_dropout=cfg_model.emb_dropout,
        )
    elif "swin" in cfg_model.name.lower():
        from .swin_transformers import SwinTransformer

        model = SwinTransformer(
            img_size=32,
            patch_size=cfg_model.patch_size,
            in_chans=channels,
            num_classes=classes,
            embed_dim=cfg_model.embed_dim,
            depths=cfg_model.depths,
            num_heads=cfg_model.num_heads,
            window_size=cfg_model.window_size,
            mlp_ratio=cfg_model.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=cfg_model.drop_path_rate,
            norm_layer=torch.nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            fused_window_process=False,
        )
    elif "swinv2" in cfg_model.name.lower():
        from .swin_transformers import SwinTransformerV2

        model = SwinTransformerV2(
            img_size=32,
            patch_size=cfg_model.patch_size,
            in_chans=channels,
            num_classes=classes,
            embed_dim=cfg_model.embed_dim,
            depths=cfg_model.depths,
            num_heads=cfg_model.num_heads,
            window_size=cfg_model.window_size,
            mlp_ratio=cfg_model.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=cfg_model.drop_path_rate,
            norm_layer=torch.nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=[0, 0, 0, 0],
        )
    elif "efficientnet" in cfg_model.name.lower():

        from efficientnet_pytorch import EfficientNet, efficientnet

        block_args, global_params = efficientnet(
            width_coefficient=2,
            depth_coefficient=1,
            image_size=32,
            dropout_rate=0.0,
            drop_connect_rate=0.0,
            num_classes=classes,
            include_top=True,
        )
        model = EfficientNet(block_args, global_params)
    else:
        raise ValueError(f"Could not find model with name {cfg_model.name}.")

    num_params, num_buffers = (
        sum([p.numel() for p in model.parameters()]),
        sum([b.numel() for b in model.buffers()]),
    )
    print(f"Model architecture {cfg_model.name} loaded with {num_params:,} parameters and {num_buffers:,} buffers.")

    if report_invariance:
        _invariance_report(model, channels)
    return model


def _invariance_report(model, channels):
    # random 32x32 RGB data [this will only test correctly for CIFAR models]
    model.eval()
    x = torch.randn(4, channels, 32, 32)

    # the images flipped along the vertical axis
    x_fv = x.flip(dims=[3])
    # the images flipped along the horizontal axis
    x_fh = x.flip(dims=[2])
    # the images rotated by 90 degrees
    x90 = x.rot90(1, (2, 3))
    # the images flipped along the horizontal axis and rotated by 90 degrees
    x90_fh = x.flip(dims=[2]).rot90(1, (2, 3))

    # feed all inputs to the model
    y = model(x)
    y_fv = model(x_fv)
    y_fh = model(x_fh)
    y90 = model(x90)
    y90_fh = model(x90_fh)

    # the outputs should be (about) the same for all transformations the model is invariant to
    print()
    print("TESTING INVARIANCE:                                    ")
    print("REFLECTIONS along the VERTICAL axis  [='horiz. flip']: " + ("YES" if torch.allclose(y, y_fv, atol=1e-6) else "NO"))
    print("REFLECTIONS along the HORIZONTAL axis [='vert. flip']: " + ("YES" if torch.allclose(y, y_fh, atol=1e-6) else "NO"))
    print("90 degrees ROTATIONS:                                  " + ("YES" if torch.allclose(y, y90, atol=1e-6) else "NO"))
    print("REFLECTIONS along the 45 degrees axis:                 " + ("YES" if torch.allclose(y, y90_fh, atol=1e-6) else "NO"))
    model.train()


def prepare_model(model, cfg, process_idx, setup):
    model.to(**setup)
    if cfg.impl.JIT == "trace":  # only rarely possible
        with torch.cuda.amp.autocast(enabled=cfg.impl.mixed_precision):
            template = torch.zeros([cfg.data.batch_size, cfg.data.channels, cfg.data.pixels, cfg.data.pixels]).to(**setup)
            model = torch.jit.trace(model, template)
    elif cfg.impl.JIT == "script":
        torch._C._jit_set_nvfuser_enabled(True)  # fuser2
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_texpr_fuser_enabled(False)  # fuser1
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_fusion_strategy([("STATIC", 20), ("DYNAMIC", 20)])  # maybe this is overkill
        model = torch.jit.script(model)
    if cfg.impl.setup.dist:
        if cfg.hyp.train_stochastic:
            # Use DDP only in stochastic mode
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[process_idx],
                output_device=process_idx,
                broadcast_buffers=False,  # ghostnorm
                static_graph=False,
            )
        else:
            for param in model.parameters():
                torch.distributed.broadcast(param.data, 0, async_op=True)
            torch.distributed.barrier()
    else:
        model.no_sync = nullcontext

    os.makedirs(os.path.join(cfg.original_cwd, "checkpoints"), exist_ok=True)

    return model


class _Select(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, : self.n]


class OutputAveragedResNet(ResNet):
    """wrap usual resnet with output averaging against horizontal flips."""

    # def __init__(self, *args, mode="sequential", raise_bn_guard=False, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.mode = mode
    #     self.raise_bn_guard = raise_bn_guard

    def forward(self, inputs, *args, **kwargs):
        # if self.mode == "sequential":
        # Sequential:
        return (super().forward(inputs, *args, **kwargs) + super().forward(torch.flip(inputs, [3]), *args, **kwargs)) / 2
        # else:
        #     # Parallel:
        #     B = inputs.shape[0]
        #     inputs = torch.cat([inputs, torch.flip(inputs, [3])], dim=0)
        #     return super().forward(inputs, *args, **kargs).view(2, B, -1).mean(dim=0)


class OrbitMappedResNet(ResNet):
    """Choose unique orbit mapping as in Gandikota et al..
    This is variation only implements flip groups, not full continuous rotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        grad_weight = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float).unsqueeze(0).unsqueeze(1)
        grad_weight = torch.cat((torch.transpose(grad_weight, 2, 3), grad_weight), 0)
        self.groups = 3
        grad_weight = torch.cat([grad_weight] * self.groups, 0)
        self.register_buffer("weight", grad_weight)

    def forward(self, inputs, *args, **kwargs):
        # if self.mode == "sequential":
        # Sequential:
        batch_left = inputs
        batch_right = torch.flip(inputs, [3])

        scores_left = self._eval_orbit_map(batch_left)
        scores_right = self._eval_orbit_map(batch_right)

        final_inputs = inputs
        final_inputs[scores_left < scores_right] = batch_right[scores_left < scores_right]
        return super().forward(final_inputs, *args, **kwargs)

    def _eval_orbit_map(self, inputs):
        """Mapping based on average gradient magnitude.

        score = <(0, 1) , integrate_hw grad(inputs)_bchw dhw
        """
        diffs = torch.nn.functional.conv2d(inputs, self.weight, None, stride=1, padding=1, dilation=1, groups=self.groups)
        score = diffs[:, 1::2].mean(dim=[1, 2, 3])
        return score
