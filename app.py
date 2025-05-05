
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion",
    torch_dtype=torch.float16,
    revision="fp16"
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate(prompt):
    image = pipe(prompt).images[0]
    return image

# Interface
demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Describe your scene (e.g., 'A peaceful Ghibli-style forest')"),
    outputs="image",
    title="Mohitama - Ghibli-Style AI Art",
    description="Create Studio Ghibli-style AI art using Mohitama, powered by open-source models."
)

demo.launch()
