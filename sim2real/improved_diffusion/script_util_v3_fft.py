import argparse
import inspect

from . import gaussian_diffusion_fft as gd
from .respace_fft import SpacedDiffusion, space_timesteps
from .unet_v3 import MARResModel, UNetModel

NUM_CLASSES = 1000


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,          # 1000
        learn_sigma=learn_sigma,        # 是否学习方差 
        noise_schedule=noise_schedule,  # 'linear'
        use_kl=use_kl,                  # false 
        predict_xstart=predict_xstart,  # False
        rescale_timesteps=rescale_timesteps, # True
        rescale_learned_sigmas=rescale_learned_sigmas, # True   使用RESCALED_MSE loss 
        timestep_respacing=timestep_respacing, # ''
    )
    return model, diffusion


def create_model(
    image_size,       # 512
    num_channels,     # 128
    num_res_blocks,   # 2
    learn_sigma,      # False
    class_cond,       # False
    use_checkpoint,   # False
    attention_resolutions, # '16,8',
    num_heads,        # 4
    num_heads_upsample, # -1 
    use_scale_shift_norm, # True
    dropout,          # 0.0
):

    if image_size == 512:
        channel_mult = (1, 2, 2, 4, 4, 8)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))   # [512 // 16 8] = [32, 64]

    return MARResModel(
        in_channels=1,                                # 网络输入通道，1,超分辨率 X 2                         1
        model_channels=num_channels,                  # 网络基础通道， 默认128                               128
        out_channels=(1 if not learn_sigma else 2),   # 是否学习方差，如果学习，输出2通道                     1
        num_res_blocks=num_res_blocks,                # U-Net 每一层有多少个残差快                           2
        attention_resolutions=tuple(attention_ds),    # 使用注意力机制位置                                   [32, 64]
        dropout=dropout,                              # dropout 概率                                        0.0
        channel_mult=channel_mult,                    # Unet 每层数及通道变化倍数                            [2, 2, 2, 2, 2]
        num_classes=(NUM_CLASSES if class_cond else None), # 是否基于条件  False                            None
        use_checkpoint=use_checkpoint,                # use gradient checkpointing to reduce memory usage.  False
        num_heads=num_heads,                          #  the number of attention heads in each attention layer. 4
        num_heads_upsample=num_heads_upsample,        # -1
        use_scale_shift_norm=use_scale_shift_norm,         # True
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,    # ** 未传入
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),  #  ** 确定使用的steps
        betas=betas,
        model_mean_type=(                                          # **
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
