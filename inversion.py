from functools import partial
from typing import Callable, List, Optional, Union, Tuple
from diffusers import StableDiffusionPipeline
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from diffusers import (
    DDIMScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler,
    PNDMScheduler, DPMSolverSinglestepScheduler, DPMSolverSDEScheduler,
    DPMSolverMultistepScheduler, KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler, UniPCMultistepScheduler
)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel

from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler,PNDMScheduler, LMSDiscreteScheduler


### credit to: https://github.com/cccntu/efficient-prompt-to-prompt

def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """ from noise to image"""
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )

def forward_ddim(x_t, alpha_t, alpha_tp1, eps_xt):
    """ from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)


class InversableStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder=None,
        requires_safety_checker: bool = True,
    ):
        # Use keyword arguments for compatibility with newer diffusers releases.
        # Newer StableDiffusionPipeline.__init__ versions insert optional modules
        # such as ``image_encoder`` before ``requires_safety_checker``. Passing
        # the latter positionally therefore registers a bool as a module and
        # fails in ``register_modules`` with ``bool has no __module__``.
        super(InversableStableDiffusionPipeline, self).__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

        self.forward_diffusion = partial(self.backward_diffusion2, reverse_process=True)

    def set_scheduler(self, sampler: str):
        """ 根据采样器类型动态设置 scheduler """
        schedulers = {
            "ddim": DDIMScheduler,
            "euler": EulerDiscreteScheduler,
            "heun": HeunDiscreteScheduler,
            "lms": LMSDiscreteScheduler,
            "pndm": PNDMScheduler,
            "dpms_s": DPMSolverSinglestepScheduler,
            "dpms_sde": DPMSolverSDEScheduler,
            "dpms_m": DPMSolverMultistepScheduler,
            "kdpm2": KDPM2DiscreteScheduler,
            "kdpm2a": KDPM2AncestralDiscreteScheduler,
            "unipc": UniPCMultistepScheduler,
        }
        if sampler in schedulers:
            self.scheduler = schedulers[sampler].from_config(self.scheduler.config)
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")

    def freeze_params(self):
        # Freeze the diffusion backbone while preserving gradients with respect
        # to latent inputs used by the watermark encoder.
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()

    def enable_revision_memory_optimizations(self):
        """Enable memory-safe differentiable diffusion for N/loss ablations."""
        self._revision_gradient_checkpointing = True
        try:
            self.enable_attention_slicing("max")
        except Exception:
            pass
        try:
            self.vae.enable_slicing()
        except Exception:
            pass
        try:
            self.vae.enable_tiling()
        except Exception:
            pass

    def _unet_sample_revision(self, latent_model_input, timestep, text_embeddings):
        """Run the frozen UNet with activation checkpointing when gradients are needed."""
        use_ckpt = bool(getattr(self, "_revision_gradient_checkpointing", False)) \
            and torch.is_grad_enabled() and latent_model_input.requires_grad
        if not use_ckpt:
            return self.unet(
                latent_model_input, timestep, encoder_hidden_states=text_embeddings
            ).sample

        def _forward(x):
            return self.unet(
                x, timestep, encoder_hidden_states=text_embeddings, return_dict=False
            )[0]

        return torch_checkpoint(_forward, latent_model_input, use_reentrant=False)

    def get_random_latents(self, latents=None, height=512, width=512, generator=None, batch_size=1):
        # 兼容 vae_scale_factor
        vae_scale_factor = getattr(self, "vae_scale_factor", 8)  # 默认 8
        height = height or self.unet.config.sample_size * vae_scale_factor
        width = width or self.unet.config.sample_size * vae_scale_factor

        device = self.device  # 使用 self.device 而不是 _execution_device
        num_channels_latents = self.unet.in_channels

        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
            latents,
        )

        return latents

    @torch.no_grad()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings
    
    # @torch.no_grad()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        vae_param = next(self.vae.parameters())
        image = image.to(device=vae_param.device, dtype=vae_param.dtype)
        use_ckpt = bool(getattr(self, "_revision_gradient_checkpointing", False)) \
            and torch.is_grad_enabled() and image.requires_grad and not sample
        if use_ckpt:
            def _encode_mode(x):
                return self.vae.encode(x).latent_dist.mode()
            encoding = torch_checkpoint(_encode_mode, image, use_reentrant=False)
        else:
            encoding_dist = self.vae.encode(image).latent_dist
            encoding = encoding_dist.sample(generator=rng_generator) if sample else encoding_dist.mode()

        # 兼容不同 VAE 版本
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        latents = encoding * scaling_factor

        return latents


    @torch.no_grad()
    def backward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: bool = True,
        **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False


        for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            prev_timestep = (
                t
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps
            )
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            
            # ddim 
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
        return latents

    def backward_diffusion2(
            self,
            use_old_emb_i=25,
            text_embeddings=None,
            old_text_embeddings=None,
            new_text_embeddings=None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            reverse_process: bool = True,
            num_steps: int = 3,  # 控制逆过程步数
            **kwargs,
    ):
        """Perform the reverse process for a specified number of steps and return latents."""
        device = self.device  # 确保所有计算都在同一设备上
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # 确保 `timesteps_tensor` 在 `device` 上
        timesteps_tensor = self.scheduler.timesteps.to(device)
        latents = latents.to(device)

        # 兼容不同 scheduler，确保 init_noise_sigma 存在
        init_noise_sigma = getattr(self.scheduler, "init_noise_sigma", 1.0)
        latents = latents * init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False

        # 获取 `alphas_cumprod`，如果 scheduler 没有这个属性，就给一个合理的默认值
        alphas_cumprod = getattr(self.scheduler, "alphas_cumprod", None)
        if alphas_cumprod is None:
            print(
                f"Warning: Scheduler {type(self.scheduler).__name__} does not have 'alphas_cumprod'. Using default values.")
            alphas_cumprod = torch.linspace(1.0, 0.0, steps=num_inference_steps, device=device)  # 线性递减模拟

        # 确保 `alphas_cumprod` 在 `device` 上
        alphas_cumprod = alphas_cumprod.to(device)

        for i, t in enumerate(reversed(timesteps_tensor) if reverse_process else timesteps_tensor):
            if i >= num_steps:  # 控制步数
                break
            t = t.long()
            if prompt_to_prompt:
                text_embeddings = old_text_embeddings if i < use_old_emb_i else new_text_embeddings

            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self._unet_sample_revision(latent_model_input, t, text_embeddings)

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 计算 prev_timestep
            num_train_timesteps = getattr(self.scheduler.config, "num_train_timesteps",
                                          num_inference_steps * 20)  # 设默认值
            prev_timestep = max(t - (num_train_timesteps // num_inference_steps), 0)  # 确保不小于 0
            prev_timestep = min(prev_timestep, len(alphas_cumprod) - 1)  # 确保不超出 `alphas_cumprod` 范围

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

            # 确保索引时 `t` 在正确设备
            t = t.to(device)

            alpha_prod_t = alphas_cumprod[t]
            alpha_prod_t_prev = alphas_cumprod[prev_timestep]

            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t

            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )

        return latents

    # 第一部分：在第13步返回噪点
    @torch.no_grad()
    def step_13_return_noise(self, latents, prompt, num_inference_steps=50,
                             guidance_scale=7.5, negative_prompt=None, generator=None,
                             start_step=13, **kwargs):
        """ 运行到第 start_step 步时返回噪声 """
        # self.set_scheduler(sampler)  # 设置所选采样器
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt=latents.shape[0],
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps[:start_step]):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self._unet_sample_revision(latent_model_input, t, text_embeddings)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    # 第二部分：继续处理剩余的时间步
    def continue_from_step_13(self, latents, prompt, num_inference_steps=50,
                              guidance_scale=7.5, negative_prompt=None, start_step=13, **kwargs):
        """ 从 step_13 继续运行到最终 """
        # self.set_scheduler(sampler)  # 设置所选采样器
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt=latents.shape[0],
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt
        )

        timesteps = self.scheduler.timesteps[start_step:]

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self._unet_sample_revision(latent_model_input, t, text_embeddings)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def generate_latents_with_guidance(self, latents, prompt, num_inference_steps=50, guidance_scale=7.5,
                                       negative_prompt=None, last_grad_steps=0, generator=None, **kwargs):
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. 编码输入提示词，不需要梯度计算
        with torch.no_grad():
            text_embeddings = self._encode_prompt(prompt, device, num_images_per_prompt=latents.shape[0],
                                                  do_classifier_free_guidance=do_classifier_free_guidance,
                                                  negative_prompt=negative_prompt)

        # 2. 设置调度器的步数和时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. 执行生成过程
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 4. 控制梯度计算范围
            if i < num_inference_steps - last_grad_steps:
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings, return_dict=False
                    )[0]
            else:
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings, return_dict=False
                )[0]

            # 5. 分类器自由引导调整
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 6. 根据 `noise_pred` 更新 `latents`
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    #输出类型: PyTorch 张量
    # @torch.no_grad()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        vae_param = next(self.vae.parameters())
        latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
        scaled_latents = 1 / 0.18215 * latents
        image = []
        for i in range(len(latents)):
            latent_i = scaled_latents[i : i + 1]
            use_ckpt = bool(getattr(self, "_revision_gradient_checkpointing", False)) \
                and torch.is_grad_enabled() and latent_i.requires_grad
            if use_ckpt:
                def _decode(x):
                    return self.vae.decode(x, return_dict=False)[0]
                decoded = torch_checkpoint(_decode, latent_i, use_reentrant=False)
            else:
                decoded = self.vae.decode(latent_i).sample
            image.append(decoded)
        return torch.cat(image, dim=0)

    #输出类型: NumPy 数组
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        # latents = 1 / self.vae.config.scaling_factor * latents
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @torch.no_grad()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

    @torch.no_grad()
    def encode_image(self, decoded_image: torch.FloatTensor, **kwargs):
        # 使用 VAE 的编码器将解码后的图像编码为潜在空间表示
        encoded_output = self.vae.encode(decoded_image)
        encoded_latents = encoded_output.mean
        return encoded_latents

    @torch.no_grad()
    def reverse_transform(self, decoded_image: torch.FloatTensor, **kwargs):
        # 1. 将 decoded_image 编码回潜在空间
        encoded_latents = self.encode_image(decoded_image, **kwargs)

        # 2. 还原缩放
        inal_latents = encoded_latents * 0.18215
        return inal_latents

    def vae_decode(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents  # 缩放潜在表示
        decoded_output = self.vae.decode(scaled_latents)  # 解码，返回的是图像
        decoded_images = decoded_output.sample
        decoded_images = decoded_images.clamp(0, 1)  # 确保像素值在有效范围内
        return decoded_images

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",  # 图片格式可以是 "pil" 或 "tensor"
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            sampler: str = "dpms_m",  # 添加采样器参数
    ):
        """ 进行推理，并支持不同的采样器 """

        # 1. 设置采样器
        self.set_scheduler(sampler)

        # 2. 计算默认 height 和 width，兼容 `vae_scale_factor`
        vae_scale_factor = getattr(self, "vae_scale_factor", 8)  # 默认 8
        height = height or self.unet.config.sample_size * vae_scale_factor
        width = width or self.unet.config.sample_size * vae_scale_factor

        # 3. 检查输入是否有效
        self.check_inputs(prompt, height, width, callback_steps)

        # 4. 定义调用参数
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self.device  # 兼容性更强
        do_classifier_free_guidance = guidance_scale > 1.0

        # 5. 编码文本提示
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 6. 设定时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. 预处理潜变量
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 8. 计算 `extra_step_kwargs`
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. 计算 `num_warmup_steps`
        scheduler_order = getattr(self.scheduler, "order", 1)  # 兼容 `scheduler` 没有 `order` 的情况
        num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler_order, 0)

        # 10. 采样去噪循环
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 预测噪声残差
            noise_pred = self._unet_sample_revision(latent_model_input, t, text_embeddings)

            # 执行自由引导（guidance）
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 计算前一步的潜变量
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # 执行回调（如果提供）
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 11. 解码最终潜变量
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents / getattr(self.vae.config, "scaling_factor", 0.18215), return_dict=False)[0]

        # 12. 仅当输出格式为 "pil" 或 "tensor" 时进行后处理
        if output_type in ["pil", "tensor"]:
            image = self.image_processor.postprocess(image, output_type=output_type)

        # 13. 释放资源
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        # 14. 返回最终图像
        return image if not return_dict else image
