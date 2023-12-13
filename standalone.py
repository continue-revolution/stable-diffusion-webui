import sys, os
from typing import Optional
sys.argv.append('--opt-sdp-attention')
sys.argv.append('--no-half-vae')
from contextlib import closing

import numpy as np
import torch
torch.set_float32_matmul_precision('medium')
from fastapi import FastAPI

from modules import initialize_util
from modules import initialize


def create_api(app):
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api


t2i_script = None
args = None
api = None
initialize.imports()
initialize.check_versions()


def init():
    global api, t2i_script, args
    from modules import script_callbacks, scripts, ui
    initialize.initialize()

    app = FastAPI()
    initialize_util.setup_middleware(app)
    api = create_api(app)

    from modules import script_callbacks, shared
    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)
    scripts.scripts_txt2img.initialize_scripts(False)
    assert shared.opts is not None

    ui.create_ui()
    t2i_script = scripts.scripts_txt2img.alwayson_scripts
    args = api.init_default_script_args(scripts.scripts_txt2img)

    return t2i_script, args


def load_model(filename):
    from modules import shared
    from modules.sd_models import CheckpointInfo, reload_model_weights
    if filename == shared.sd_model.sd_model_checkpoint:
        return
    
    reload_model_weights(shared.sd_model, CheckpointInfo(filename))


def enable_animatediff(
    alwayson_scripts,
    args,
    model="mm_sd_v15_v2.ckpt",
    video_length=0,
    fps=8,
    loop_number=0,
    closed_loop='R-P',
    batch_size=16,
    stride=1,
    overlap=-1,
    format=["GIF", "PNG"],
    interp='Off',
    interp_x=10,
    video_source=None,
    video_path='',
    latent_power=1,
    latent_scale=32,
    last_frame=None,
    latent_power_last=1,
    latent_scale_last=32,
):
    for script in alwayson_scripts:
        if script.name == 'animatediff':
            target = script
            break
    else:
        return args

    assert model
    # Thx to animatediff extension, we only have 1 arg here
    animatediff_arg = args[target.args_from]
    animatediff_arg.model = model
    animatediff_arg.enable = True
    animatediff_arg.video_length = video_length
    animatediff_arg.fps = fps
    animatediff_arg.loop_number = loop_number
    animatediff_arg.closed_loop = closed_loop
    animatediff_arg.batch_size = batch_size
    animatediff_arg.stride = stride
    animatediff_arg.overlap = overlap
    animatediff_arg.format = format
    animatediff_arg.interp = interp
    animatediff_arg.interp_x = interp_x
    animatediff_arg.video_source = video_source
    animatediff_arg.video_path = video_path
    animatediff_arg.latent_power = latent_power
    animatediff_arg.latent_scale = latent_scale
    animatediff_arg.last_frame = last_frame
    animatediff_arg.latent_power_last = latent_power_last
    animatediff_arg.latent_scale_last = latent_scale_last
    
    args[script.args_from] = animatediff_arg

    return args


def enable_controlnet(
    alwayson_scripts,
    args,
    input_mode: str = "batch",
    batch_images: str = "",
    module: Optional[str] = None,
    model: Optional[str] = None,
    weight: float = 1.0,
    image: Optional[np.ndarray] = None,
    resize_mode: str = "Crop and Resize",
    processor_res: int = 512,
    threshold_a: float = -1,
    threshold_b: float = -1,
    guidance_start: float = 0.0,
    guidance_end: float = 1.0,
    pixel_perfect: bool = True,
    control_mode: str = 'Balanced',
):
    assert model and module
    for script in alwayson_scripts:
        if script.name == 'controlnet':
            target = script
            break
    else:
        return args

    if target.args_to - target.args_from:
        remove = target.args_to - target.args_from
        for cnet_arg in args[target.args_from:target.args_to]:
            if cnet_arg.enabled:
                remove -= 1
        if remove:
            for _ in range(remove):
                args.pop(target.args_to-1)
                target.args_to -= 1
            shift = False
            for script in alwayson_scripts:
                if shift:
                    script.args_from -= remove
                    script.args_to -= remove
                if script.name == 'controlnet':
                    shift = True
    target.args_to += 1
    shift = False
    for script in alwayson_scripts:
        if shift:
            script.args_from += 1
            script.args_to += 1
        if script.name == 'controlnet':
            shift = True

    from scripts.controlnet_ui.controlnet_ui_group import UiControlNetUnit
    from scripts.batch_hijact import InputMode
    unit = UiControlNetUnit()
    if image is not None:
        unit.image = {'image': np.array(image), 'mask': np.zeros_like(np.array(image))}
    unit.input_mode = InputMode(input_mode)
    unit.batch_images = batch_images
    unit.module = module
    unit.model = model
    unit.weight = weight
    unit.image = image
    unit.resize_mode = resize_mode
    unit.processor_res = processor_res
    unit.threshold_a = threshold_a
    unit.threshold_b = threshold_b
    unit.guidance_start = guidance_start
    unit.guidance_end = guidance_end
    unit.pixel_perfect = pixel_perfect
    unit.control_mode = control_mode
    args.insert(target.args_to-1, unit)
    return args


def disable_all(
    alwayson_scripts,
    args,
):
    for script in alwayson_scripts:
        if script.name == 'controlnet':
            for arg in args[script.args_from:script.args_to]:
                arg.enabled = False
        if script.name == 'animatediff':
            for arg in args[script.args_from:script.args_to]:
                arg.enable = False


def txt2img(
    prompt: str = '1girl',
    negative_prompt: str = '',
    steps: int = 30,
    sampler_name: str = 'DPM++ 2M SDE Heun Exponential',
    n_iter: int = 1,
    batch_size: int = 1,
    cfg_scale: float = 5,
    height: int = 960,
    width: int = 576,
    enable_hr = False,
    denoising_strength: float = 0.6,
    hr_scale: float = 0,
    hr_upscaler: str = 'Lanczos',
    hr_second_pass_steps: int = 10,
    hr_resize_x: int = 1600,
    hr_resize_y: int = 960,
    hr_checkpoint_name: str = '',
    hr_sampler_name: str = '',
    hr_prompt: str = '',
    hr_negative_prompt: str = '',
    t2i_script = None,
    args: tuple = tuple()
):
    from modules import processing, shared, scripts
    from modules.shared import opts
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        prompt=prompt,
        negative_prompt=negative_prompt,
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        firstphase_width=width,
        firstphase_height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength if enable_hr else None,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
        hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        do_not_save_grid=True,
        do_not_save_samples=True
    )

    scripts.scripts_txt2img.alwayson_scripts = t2i_script
    p.scripts = scripts.scripts_txt2img
    p.script_args = args

    with closing(p):
        processed = scripts.scripts_txt2img.run(p, *args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    return processed.images


def img2img_batch(
    prompt: str = '1girl',
    negative_prompt: str = '',
    steps: int = 30,
    sampler_name: str = 'DPM++ 2M SDE Heun Exponential',
    mask_blur: int = 4,
    inpainting_fill: int = 1,
    n_iter: int = 1,
    batch_size: int = 1,
    cfg_scale: float = 5,
    image_cfg_scale: float = 5,
    denoising_strength: float = 0.6,
    height: int = 960,
    width: int = 576,
    scale_by: float = 0,
    resize_mode: int = 0,
    inpaint_full_res: bool = 1,
    inpaint_full_res_padding: int = 32,
    inpainting_mask_invert: int = 0,
    input_dir: str = '',
    inpaint_mask_dir = '',
    args: tuple = tuple(),
    use_png_info=False,
    png_info_props=None,
    png_info_dir=None
):

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    from modules import processing, shared, scripts, img2img
    from modules.shared import opts
    p = processing.StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        prompt=prompt,
        negative_prompt=negative_prompt,
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[],
        mask=None,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        do_not_save_grid=True,
        do_not_save_samples=True
    )

    scripts.scripts_txt2img.alwayson_scripts = t2i_script
    p.scripts = scripts.scripts_img2img
    p.script_args = args

    if inpaint_mask_dir:
        p.extra_generation_params["Mask blur"] = mask_blur

    with closing(p):
        processed = img2img.process_batch(p, input_dir, '', inpaint_mask_dir, args, to_scale=scale_by > 0, scale_by=scale_by, use_png_info=use_png_info, png_info_props=png_info_props, png_info_dir=png_info_dir)

    shared.total_tqdm.clear()

    return processed.images


def img2img_single(
    prompt: str = '1girl',
    negative_prompt: str = '',
    image = None,
    mask = None,
    steps: int = 30,
    sampler_name: str = 'DPM++ 2M SDE Heun Exponential', 
    mask_blur: int = 4,
    inpainting_fill: int = 1,
    n_iter: int = 1,
    batch_size: int = 1,
    cfg_scale: float = 5,
    image_cfg_scale: float = 5,
    denoising_strength: float = 0.6,
    height: int = 960,
    width: int = 576,
    scale_by: float = 0,
    resize_mode: int = 0,
    inpaint_full_res: bool = 1,
    inpaint_full_res_padding: int = 32,
    inpainting_mask_invert: int = 0,
    args: tuple = tuple()
):
    from PIL import ImageOps

    # Use the EXIF orientation of photos taken by smartphones.
    if image is not None:
        image = ImageOps.exif_transpose(image)

    if scale_by > 0:
        assert image, "Can't scale by because no image is selected"

        width = int(image.width * scale_by)
        height = int(image.height * scale_by)

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    from modules import processing, shared, scripts
    from modules.shared import opts
    p = processing.StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        prompt=prompt,
        negative_prompt=negative_prompt,
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        do_not_save_grid=True,
        do_not_save_samples=True
    )

    scripts.scripts_txt2img.alwayson_scripts = t2i_script
    p.scripts = scripts.scripts_img2img
    p.script_args = args

    if mask:
        p.extra_generation_params["Mask blur"] = mask_blur

    with closing(p):
        processed = scripts.scripts_img2img.run(p, *args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    return processed.images


mode_to_func = {
    'img2img_batch': img2img_batch,
    'img2img_single': img2img_single,
    'txt2img': txt2img,
}


@torch.no_grad()
def process(
    config = {
        'mode': 'txt2img',
        "model": "",
        "webui": {    
            'prompt': '1girl',
            'negative_prompt': 'easynegative',
            'steps': 20,
            'width': 512,
            'height': 512,
        },
        'cn': [
            {
                'model': 'control_v11p_sd15_canny',
                'module': 'canny',
                'weight': 0.8,
                'extra_kwargs': {
                    'threshold_a': 50,
                    'threshold_b': 150,
                }
            },
            {
                'model': 'control_v11f1p_sd15_depth',
                'module': 'depth_midas',
                'weight': 0.7
            }
        ],
        "ad": {
            "video_length": 24,
        }
    }
):
    load_model(config['model'])
    global t2i_script, args

    args = enable_animatediff(
        alwayson_scripts = t2i_script,
        args = args,
        **config['ad'])

    for cnet in config['cn']:
        args = enable_controlnet(
            alwayson_scripts = t2i_script,
            args = args,
            **cnet
        )

    from modules import devices
    with devices.autocast():
        test_imgs = mode_to_func[config["mode"]](
            **config['webui'],
            t2i_script = t2i_script,
            args = args,
        )
    devices.torch_gc()
    disable_all(t2i_script, args)


if __name__ == '__main__':
    t2i_script, arg = init()

    import time
    t0 = time.time()
    process(config={
        "mode": "txt2img",
        "model": "/home/conrevo/SD/stable-diffusion-webui/models/Stable-diffusion/AnythingV5Ink_ink.safetensors",
        "webui": {
            "prompt": "1girl, yoimiya (genshin impact), origen, line, comet, wink, Masterpiece, BestQuality. UltraDetailed, <lora:LineLine2D:0.7>, <lora:yoimiya:0.8>\n0: closed mouth\n10: open mouth",
            "negative_prompt": "sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt_v2, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, ((2girl)), (deformed fingers:1.2), (long fingers:1.2),succubus wings,horn,succubus horn,succubus hairstyle, (bad-artist-anime), bad-artist, bad hand, grayscale, skin spots, acnes, skin blemishes",
            "steps": 25,
            "width": 512,
            "height": 512
        },
        "cn": [],
        "ad": {
            "video_length": 24,
        }
    })
    t1 = time.time()
    print(t1-t0)
