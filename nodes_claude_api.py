import os
import io
import base64
import numpy as np
from PIL import Image

import anthropic


MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


def tensor_to_base64(tensor):
    """Convert a ComfyUI IMAGE tensor to a base64-encoded JPEG string."""
    arr = np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=95)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


class ClaudeAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODELS, {"default": "claude-sonnet-4-6"}),
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
    CATEGORY = "Realstagram/Claude"
    DESCRIPTION = "Call the Anthropic Claude API with text and optional images."

    def run(self, model, api_key, prompt, max_tokens, image_1=None, image_2=None):
        key = api_key.strip() if api_key.strip() else os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("No API key provided and ANTHROPIC_API_KEY env var not set.")

        client = anthropic.Anthropic(api_key=key)

        content = []

        for img in (image_1, image_2):
            if img is not None:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": tensor_to_base64(img),
                    },
                })

        content.append({"type": "text", "text": prompt})

        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )

        return (message.content[0].text,)


NODE_CLASS_MAPPINGS = {
    "ClaudeAPI": ClaudeAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClaudeAPI": "Claude API",
}
