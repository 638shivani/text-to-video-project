import streamlit as st
import torch
import numpy as np
import imageio
import torch.optim as optim

from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler
)

from peft import (
    LoraConfig,
    get_peft_model
)

# ---------------------------------
# STREAMLIT PAGE
# ---------------------------------

st.set_page_config(
    page_title="AI Text To Video",
    layout="wide"
)

st.title("🎬 AI Text To Video Generator")

st.write("Generate AI videos using Fine-Tuned Text-to-Video Model")

prompt = st.text_area(
    "Enter Prompt",
    height=200
)

generate = st.button("Generate Video")

# ---------------------------------
# LOAD MODEL + FINETUNE
# ---------------------------------

@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    # -----------------------------
    # LOAD PRETRAINED MODEL
    # -----------------------------

    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=dtype
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    pipe = pipe.to(device)

    # -----------------------------
    # APPLY LORA FINETUNING
    # -----------------------------

    lora_config = LoraConfig(
        r=2,
        lora_alpha=8,
        target_modules=["to_q", "to_k"],
        lora_dropout=0.05,
        bias="none"
    )

    pipe.unet = get_peft_model(
        pipe.unet,
        lora_config
    )

    # -----------------------------
    # SMALL TRAINING DATASET
    # -----------------------------

    dataset = [
        "A dog running in a green park",
        "A car moving on road",
        "A person walking on beach"
    ]

    optimizer = optim.Adam(
        pipe.unet.parameters(),
        lr=1e-4
    )

    # -----------------------------
    # FINETUNING LOOP
    # -----------------------------

    st.write("⚡ Fine-tuning model...")

    for epoch in range(2):

        for text_prompt in dataset:

            latents = torch.randn(
                (1, 4, 4, 32, 32),
                device=device,
                dtype=dtype
            )

            timestep = torch.randint(
                0,
                1000,
                (1,),
                device=device
            ).long()

            noise = torch.randn_like(
                latents,
                dtype=dtype
            )

            noisy_latents = latents + noise

            text_input = pipe.tokenizer(
                text_prompt,
                return_tensors="pt"
            ).to(device)

            encoder_hidden_states = pipe.text_encoder(
                **text_input
            )[0]

            encoder_hidden_states = encoder_hidden_states.to(dtype)

            noise_pred = pipe.unet(
                noisy_latents,
                timestep,
                encoder_hidden_states=encoder_hidden_states
            ).sample

            loss = ((noise_pred - noise) ** 2).mean()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        st.write(f"Epoch {epoch} Loss: {loss.item()}")

    return pipe

# ---------------------------------
# GENERATE VIDEO
# ---------------------------------

if generate:

    if prompt == "":

        st.warning("Please enter prompt")

    else:

        with st.spinner("Generating Video..."):

            pipe = load_model()

            result = pipe(
                prompt,

                negative_prompt="""
                blurry,
                unclear,
                low quality,
                pixelated,
                unrealistic,
                distorted,
                bad frame composition
                """,

                num_inference_steps=50,

                num_frames=16,

                guidance_scale=10
            )

            video_frames = result.frames

            frames = video_frames[0]

            frames = [
                (frame * 255).astype(np.uint8)
                for frame in frames
            ]

            output_video = "generated_video.mp4"

            imageio.mimsave(
                output_video,
                frames,
                fps=4
            )

            st.success("✅ Video Generated Successfully!")

            st.video(output_video)

            # DOWNLOAD BUTTON

            with open(output_video, "rb") as file:

                st.download_button(
                    label="⬇ Download Video",
                    data=file,
                    file_name="generated_video.mp4",
                    mime="video/mp4"
                )
