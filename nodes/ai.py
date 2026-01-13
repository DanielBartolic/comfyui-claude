"""AI functions."""

import base64
import io
import logging

import anthropic
from PIL import Image
import numpy as np

# Updated model list including latest Claude models
models = [
    # Latest versions (recommended)
    'claude-3-5-haiku-latest',
    'claude-3-5-sonnet-latest',
    'claude-3-opus-latest',
    # New Claude 4.5 models
    'claude-sonnet-4-5-20250929',
    'claude-haiku-4-5-20251001',
    # Claude 4 models
    'claude-opus-4-1-20250805',
    'claude-sonnet-4-20250514',
    # Specific dated versions
    'claude-3-5-haiku-20241022',
    'claude-3-5-sonnet-20241022',
    'claude-3-opus-20240229',
    'claude-3-haiku-20240307',
]


def run_prompt(
    prompt: str, system_prompt: str, model: str, api_key: str
) -> str:
    """Execute a simple prompt with Claude.

    Args:
        prompt (str): The prompt to execute.
        system_prompt (str): The system prompt to use.
        model (str): The model to use.
        api_key (str): The API key to use.

    Returns:
        str: The result of the prompt.
    """
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {'role': 'user', 'content': prompt},
            ],
        )

        if message and message.content and len(message.content) > 0:
            return message.content[0].text  # type: ignore  # noqa: PGH003

    except anthropic.AuthenticationError:
        error_msg = "Authentication failed. Please check your API key."
        logging.error(error_msg)
        return f"ERROR: {error_msg}"
    except Exception as e:
        error_msg = f'Error: {str(e)}'
        logging.error(error_msg)
        return f"ERROR: {error_msg}"

    return ''


def describe_image(
    image: 'torch.Tensor',  # type: ignore[name-defined]  # noqa: F821
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
) -> str:
    """Send an image to Claude's vision API.

    Args:
        image (torch.Tensor): The image to describe.
        prompt (str): The prompt to use.
        system_prompt (str): The system prompt to use.
        model (str): The model to use.
        api_key (str): The API key to use.

    Returns:
        str: The result of the prompt.
    """
    try:
        # Validate image exists
        if image is None:
            return "ERROR: No image provided"
            
        # Convert tensor to image
        if len(image.shape) == 4:
            image = image.squeeze(0)  # Remove batch dimension
            
        # Handle different tensor ranges
        if image.max() <= 1.0:
            image_tensor = image * 255
        else:
            image_tensor = image
            
        image_array = image_tensor.byte().cpu().numpy()
        
        # Handle different channel arrangements
        if len(image_array.shape) == 3 and image_array.shape[0] == 3:
            image_array = np.transpose(image_array, (1, 2, 0))
            
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array, mode='RGB')
        
        # Save to bytes
        buffered = io.BytesIO()
        pil_image.save(buffered, format='JPEG', quality=95)
        img_data = buffered.getvalue()

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/jpeg',
                                'data': base64.b64encode(img_data).decode(
                                    'utf-8'
                                ),
                            },
                        },
                        {
                            'type': 'text',
                            'text': prompt,
                        },
                    ],
                }
            ],
        )

        if message and message.content and len(message.content) > 0:
            return message.content[0].text  # type: ignore  # noqa: PGH003

    except anthropic.AuthenticationError:
        error_msg = "Authentication failed. Please check your API key."
        logging.error(error_msg)
        return f"ERROR: {error_msg}"
    except Exception as e:
        error_msg = f'Error processing image: {str(e)}'
        logging.error(error_msg)
        return f"ERROR: {error_msg}"

    return ''


def describe_image_cached(
    image: 'torch.Tensor',  # type: ignore[name-defined]  # noqa: F821
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
) -> str:
    """Send an image to Claude's vision API with prompt caching.

    This function uses prompt caching to save costs when processing multiple
    images with the same system prompt. The system prompt is cached and reused
    across API calls, reducing token costs by 90% for cached content.

    Args:
        image (torch.Tensor): The image to describe.
        prompt (str): The prompt to use.
        system_prompt (str): The system prompt to use (will be cached).
        model (str): The model to use.
        api_key (str): The API key to use.

    Returns:
        str: The result of the prompt.
    """
    try:
        # Validate image exists
        if image is None:
            return "ERROR: No image provided"
            
        # Convert tensor to image
        if len(image.shape) == 4:
            image = image.squeeze(0)  # Remove batch dimension
            
        # Handle different tensor ranges
        if image.max() <= 1.0:
            image_tensor = image * 255
        else:
            image_tensor = image
            
        image_array = image_tensor.byte().cpu().numpy()
        
        # Handle different channel arrangements
        if len(image_array.shape) == 3 and image_array.shape[0] == 3:
            image_array = np.transpose(image_array, (1, 2, 0))
            
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array, mode='RGB')
        
        # Save to bytes
        buffered = io.BytesIO()
        pil_image.save(buffered, format='JPEG', quality=95)
        img_data = buffered.getvalue()

        client = anthropic.Anthropic(api_key=api_key)
        
        # Use prompt caching by structuring system as an array with cache_control
        system_with_cache = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ] if system_prompt else []
        
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_with_cache,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/jpeg',
                                'data': base64.b64encode(img_data).decode(
                                    'utf-8'
                                ),
                            },
                        },
                        {
                            'type': 'text',
                            'text': prompt,
                        },
                    ],
                }
            ],
        )

        if message and message.content and len(message.content) > 0:
            return message.content[0].text  # type: ignore  # noqa: PGH003

    except anthropic.AuthenticationError:
        error_msg = "Authentication failed. Please check your API key."
        logging.error(error_msg)
        return f"ERROR: {error_msg}"
    except Exception as e:
        error_msg = f'Error processing image: {str(e)}'
        logging.error(error_msg)
        return f"ERROR: {error_msg}"

    return ''
