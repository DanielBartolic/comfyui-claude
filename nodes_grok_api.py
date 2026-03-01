import os
import io
import base64
import numpy as np
from PIL import Image

from openai import OpenAI


MODELS = [
    "grok-4-1-fast-non-reasoning",
    "grok-4-1-fast-reasoning",
]


def tensor_to_base64(tensor):
    """Convert a ComfyUI IMAGE tensor to a base64-encoded JPEG string."""
    arr = np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=95)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


class GrokAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODELS, {"default": "grok-4-1-fast-non-reasoning"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 32000}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "run"
    CATEGORY = "Realstagram/Grok"
    DESCRIPTION = "Call the xAI Grok API with text and optional images."

    def run(self, model, api_key, prompt, max_tokens, image_1=None, image_2=None):
        key = api_key.strip() if api_key.strip() else os.environ.get("XAI_API_KEY", "")
        if not key:
            raise ValueError("No API key provided and XAI_API_KEY env var not set.")

        client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")

        content = []

        for img in (image_1, image_2):
            if img is not None:
                b64 = tensor_to_base64(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                    },
                })

        content.append({"type": "text", "text": prompt})

        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )

        return (response.choices[0].message.content,)


NODE_CLASS_MAPPINGS = {
    "GrokAPI": GrokAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokAPI": "Grok API",
}
