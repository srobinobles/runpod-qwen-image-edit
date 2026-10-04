# syntax=docker/dockerfile:1
#
# RunPod Serverless worker for Qwen-Image-Edit-2511.
#
# A thin derivative of the official worker-comfyui that just bakes the models
# into the image — no custom handler, no entrypoint. The official implementation
# runs as-is.
#
# The workflow is passed per request via the API's input.workflow, so output
# size, step count, and LoRA choice can change without rebuilding the image.
# Many prebuilt workers freeze the workflow inside the image and can't move the
# output off square (a given width/height is silently dropped).
#
# Models are baked into the image, not mounted from a network volume: a volume
# keeps billing for capacity as long as it exists, and reading weights from it
# makes cold starts slower. The base tag is pinned to a PyTorch built for CUDA
# 12.8 — the plain 5.8.6-base installs comfy-cli's default (newer) PyTorch, which
# fails to start ("no kernel image is available") on a host that declares 12.8.
FROM runpod/worker-comfyui:5.8.6-base-cuda12.8.1 AS base

# Models are fetched in separate stages. BuildKit runs independent stages in
# parallel, so a download that took ~22 min sequentially drops to about the time
# of the single largest file. RunPod Hub caps the build at 30 min and spends ~8
# of those on image export/transfer, so the parallelism is what lets it finish.
# Fetch with the wget that ships in base; download to absolute paths and COPY
# into place in the final stage (comfy model download's destination depends on
# comfy-cli's default workspace setting).

# The diffusion model. FP8-mixed quantized build (the bf16 full weights are
# 53.7 GB and need an 80 GB-class GPU; FP8 fits on a 24 GB card).
FROM base AS diffusion
RUN mkdir -p /models/diffusion_models \
 && wget -q --tries=3 -O /models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
      https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors

# Text encoder (Qwen2.5-VL 7B).
FROM base AS text-encoder
RUN mkdir -p /models/text_encoders \
 && wget -q --tries=3 -O /models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
      https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors

FROM base AS vae
RUN mkdir -p /models/vae \
 && wget -q --tries=3 -O /models/vae/qwen_image_vae.safetensors \
      https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors

# LoRAs. The Lightning LoRA gives 4-step generation (less time + GPU cost; the
# workflow decides whether to use it).
# ── ADDED (Ancientel §2.B Phase B): also pull fal's multi-angle LoRA into the
#    same stage. The existing COPY --from=lora carries all of /models/loras/ into
#    the image, so one extra wget here is all that's needed.
FROM base AS lora
RUN mkdir -p /models/loras \
 && wget -q --tries=3 -O /models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
      https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
 && wget -q --tries=3 -O /models/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors \
      https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/qwen-image-edit-2511-multiple-angles-lora.safetensors

FROM base

# Keep models in separate COPY layers (four layers). Merging them into one makes
# a single ~29 GB layer, which makes export/transfer even slower.
COPY --from=diffusion /models/diffusion_models/ /comfyui/models/diffusion_models/
COPY --from=text-encoder /models/text_encoders/ /comfyui/models/text_encoders/
COPY --from=vae /models/vae/ /comfyui/models/vae/
COPY --from=lora /models/loras/ /comfyui/models/loras/

# The base image already ships this handler, but the Hub listing requires a
# handler.py in the repo itself, so it's placed here explicitly. Contents are
# worker-comfyui's, unchanged (both are AGPL-3.0).
COPY handler.py /handler.py
