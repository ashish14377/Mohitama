
# Mohitama - Ghibli Style Image Generator (Colab + Gradio)
!pip install diffusers transformers accelerate gradio --quiet

from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

# Load Ghibli-style model from Hugging Face
pipe = StableDiffusionPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion",
    torch_dtype=torch.float16,
    revision="fp16"
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate(prompt):
    image = pipe(prompt).images[0]
    return image

# Gradio Interface
demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Describe your scene (e.g. 'A fantasy forest with sunlight')"),
    outputs="image",
    title="Mohitama - Art with the Soul of Mohit",
    description="Generate Ghibli-style art from your imagination."
)

demo.launch()
