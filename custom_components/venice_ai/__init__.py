# custom_components/venice_ai/__init__.py

from __future__ import annotations

import logging
import base64
import mimetypes # For guessing MIME type from path

import httpx # For fetching image from URL
import voluptuous as vol

from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from homeassistant.components.camera import (
    async_get_image,
    DEFAULT_CONTENT_TYPE as CAMERA_DEFAULT_CONTENT_TYPE,
    DOMAIN as CAMERA_DOMAIN,
)

# Import standard Home Assistant constants directly
from homeassistant.const import CONF_API_KEY

# Constants from your custom_components/venice_ai/const.py
from .const import (
    DOMAIN,
    CONF_CHAT_MODEL,
    RECOMMENDED_CHAT_MODEL,
    CONF_MAX_TOKENS,
    RECOMMENDED_MAX_TOKENS,
    CONF_TEMPERATURE,
    RECOMMENDED_TEMPERATURE,
    CONF_TOP_P,
    RECOMMENDED_TOP_P,
    # Add any other constants from your const.py that are used in this file
)
from .client import AsyncVeniceAIClient, VeniceAIError, AuthenticationError

_LOGGER = logging.getLogger(__name__)

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_ANALYZE_IMAGE = "analyze_image"

PLATFORMS = ("conversation",) # From your manifest.json
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN) # From original __init__.py

class VeniceAIConfigEntry(ConfigEntry):
    """Venice AI config entry with runtime data."""
    runtime_data: AsyncVeniceAIClient

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Venice AI services."""

    # --- Handler for the existing generate_image service ---
    async def render_image(call: ServiceCall) -> ServiceResponse:
        entry_id = call.data["config_entry"]
        entry: VeniceAIConfigEntry | None = hass.config_entries.async_get_entry(entry_id) # type: ignore

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry", # Ensure this key exists in strings.json
                translation_placeholders={"config_entry": entry_id},
            )

        client: AsyncVeniceAIClient = entry.runtime_data
        
        # Parameters from your services.yaml for generate_image
        prompt = call.data["prompt"]
        model_override = call.data.get("model") # Optional model override
        size = call.data.get("size", "1024x1024")
        quality = call.data.get("quality", "standard") # Maps to SimpleGenerateImageRequest
        style = call.data.get("style", "natural")     # Maps to SimpleGenerateImageRequest
        
        # Determine if using the more detailed /image/generate or OpenAI-compatible /images/generations
        # The swagger file suggests /images/generations uses 'size', 'quality', 'style'
        # and /image/generate uses 'height', 'width', 'steps', 'cfg_scale' etc.
        # Let's assume the client.images.generate method in your client.py
        # is designed to handle the simpler OpenAI-compatible parameters or adapt them.
        # If it targets /image/generate directly, you might need to parse 'size' into height/width.

        try:
            # This call needs to align with how your client.py's `images.generate` is implemented
            # and which Venice AI endpoint it targets (/image/generate or /images/generations).
            # The SimpleGenerateImageRequest uses 'response_format' (url or b64_json).
            response = await client.images.generate( # This assumes client.images.generate exists
                prompt=prompt,
                model=model_override or entry.options.get("image_model", "default"), # Or a dedicated image_model option
                size=size,
                quality=quality,
                style=style,
                response_format="b64_json", # As per SimpleGenerateImageRequest, or "url"
                n=1, # SimpleGenerateImageRequest only supports n=1
                # If targeting /image/generate, parameters would be like:
                # height=parsed_height, width=parsed_width, steps=20, cfg_scale=7.5 etc.
            )

            # Response processing based on SimpleGenerateImageRequest (OpenAI compatible)
            if response.data and response.data[0]:
                if response.data[0].b64_json:
                    return {"b64_json": response.data[0].b64_json}
                if response.data[0].url: # If you requested URL
                    return {"url": response.data[0].url}
            
            # Fallback for /image/generate structure if response is different
            if hasattr(response, 'images') and response.images and isinstance(response.images, list) and response.images[0]:
                 return {"b64_json": response.images[0]}

            _LOGGER.error(f"Failed to get image data from Venice AI response: {response}")
            raise HomeAssistantError("Failed to get image data from Venice AI response.")

        except VeniceAIError as err:
            _LOGGER.error(f"Error generating image with Venice AI: {err}")
            raise HomeAssistantError(f"Error generating image: {err}") from err
        except Exception as err:
            _LOGGER.exception(f"Unexpected error generating image: {err}")
            raise HomeAssistantError(f"Unexpected error generating image: {err}") from err

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        render_image,
        schema=vol.Schema( # Schema from your modified services.yaml
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {"integration": DOMAIN}
                ),
                vol.Required("prompt"): cv.string,
                vol.Optional("model"): cv.string,
                vol.Optional("size", default="1024x1024"): vol.In(
                    ("auto", "256x256", "512x512", "1024x1024", "1024x1792", "1792x1024", "1536x1024", "1024x1536")
                ),
                vol.Optional("quality", default="standard"): vol.In(
                    ("auto", "standard", "hd", "high", "medium", "low")
                ),
                vol.Optional("style", default="natural"): vol.In(
                    ("vivid", "natural")
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    # --- New analyze_image service handler ---
    async def handle_analyze_image(call: ServiceCall) -> ServiceResponse:
        """Handle the analyze_image service call using Venice AI vision."""
        entry_id = call.data["config_entry"]
        entry: VeniceAIConfigEntry | None = hass.config_entries.async_get_entry(entry_id) # type: ignore

        if not entry or entry.domain != DOMAIN:
            raise ServiceValidationError(f"Invalid config_entry: {entry_id}")

        client: AsyncVeniceAIClient = entry.runtime_data
        if not client:
            _LOGGER.error("Venice AI client not available for config entry %s", entry_id)
            raise HomeAssistantError("Venice AI client not available.")

        prompt_text = call.data["prompt"]
        image_entity_id = call.data.get("image_entity")
        image_url = call.data.get("image_url")
        image_path = call.data.get("image_path")

        image_bytes = None
        image_mime_type = CAMERA_DEFAULT_CONTENT_TYPE

        try:
            if image_entity_id:
                _LOGGER.debug(f"Fetching image from entity: {image_entity_id}")
                image_data = await async_get_image(hass, image_entity_id, timeout=10)
                if image_data:
                    image_bytes = image_data.content
                    image_mime_type = image_data.content_type
                else:
                    raise HomeAssistantError(f"Could not retrieve image from entity: {image_entity_id}")
            elif image_url:
                _LOGGER.debug(f"Fetching image from URL: {image_url}")
                http_client_shared = get_async_client(hass)
                response = await http_client_shared.get(image_url, timeout=30)
                response.raise_for_status()
                image_bytes = response.content
                content_type_header = response.headers.get("content-type")
                if content_type_header:
                    image_mime_type = content_type_header.split(";")[0].strip()
            elif image_path:
                _LOGGER.debug(f"Fetching image from path: {image_path}")
                if not hass.config.is_allowed_path(image_path):
                    raise ServiceValidationError(f"Path not allowed for Home Assistant access: {image_path}")
                def read_file_bytes_sync(path_to_read):
                    with open(path_to_read, "rb") as f_img:
                        return f_img.read()
                image_bytes = await hass.async_add_executor_job(read_file_bytes_sync, image_path)
                
                mime_type_guess, _ = mimetypes.guess_type(image_path)
                if mime_type_guess:
                    image_mime_type = mime_type_guess
                else:
                    _LOGGER.warning(f"Could not determine MIME type for path {image_path}, defaulting to {image_mime_type}")

            if not image_bytes:
                raise HomeAssistantError("Failed to load image data from the provided source.")

            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:{image_mime_type};base64,{base64_image}"

            messages_payload = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": { "url": data_url }
                        },
                    ],
                }
            ]

            chat_model = entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
            _LOGGER.info(f"Using model '{chat_model}' for vision analysis. Ensure this model supports vision.")

            api_call_payload = {
                "model": chat_model,
                "messages": messages_payload,
                "max_tokens": entry.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                "temperature": entry.options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "top_p": entry.options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            }
            _LOGGER.debug(f"Sending vision analysis request to Venice AI. Model: {chat_model}. Prompt: '{prompt_text[:100]}...'")

            response_data = await client.chat.create_non_streaming(payload=api_call_payload)
            _LOGGER.debug(f"Received vision analysis response from Venice AI: {response_data}")

            if response_data and response_data.get("choices"):
                choice = response_data["choices"][0]
                message_data = choice.get("message", {})
                assistant_response_content = message_data.get("content")
                final_text_response = ""
                if isinstance(assistant_response_content, str):
                    final_text_response = assistant_response_content
                elif isinstance(assistant_response_content, list):
                    for part in assistant_response_content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            final_text_response += part.get("text", "") + " "
                    final_text_response = final_text_response.strip()
                else:
                     _LOGGER.warning(f"Received unexpected content type from assistant: {type(assistant_response_content)}")
                return {"text": final_text_response}
            else:
                _LOGGER.error(f"Unexpected or empty 'choices' in response from Venice AI vision call: {response_data}")
                raise HomeAssistantError("Venice AI returned an unexpected or empty response for image analysis.")

        except ServiceValidationError:
            raise
        except VeniceAIError as err:
            _LOGGER.error(f"Venice AI API error during image analysis: {err}")
            raise HomeAssistantError(f"Venice AI API error: {err}") from err
        except httpx.HTTPStatusError as err_http:
            _LOGGER.error(f"HTTP error processing image for analysis: URL={err_http.request.url}, Status={err_http.response.status_code}, Response='{err_http.response.text}'")
            raise HomeAssistantError(f"Failed to process image (HTTP {err_http.response.status_code})") from err_http
        except httpx.RequestError as err_req:
            _LOGGER.error(f"Request error processing image for analysis: URL={err_req.request.url}, Error={err_req}")
            raise HomeAssistantError(f"Failed to process image (Request error)") from err_req
        except Exception as err_ex:
            _LOGGER.exception("Unexpected error during Venice AI image analysis.")
            raise HomeAssistantError(f"An unexpected error occurred: {err_ex}") from err_ex

    hass.services.async_register(
        DOMAIN,
        SERVICE_ANALYZE_IMAGE,
        handle_analyze_image,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector({"integration": DOMAIN}),
                vol.Required("prompt"): cv.string,
                vol.Exclusive("image_entity", "image_source_group"): selector.EntitySelector(
                    {"domain": CAMERA_DOMAIN}
                ),
                vol.Exclusive("image_url", "image_source_group"): selector.TextSelector(
                    {"type": "url"}
                ),
                vol.Exclusive("image_path", "image_source_group"): selector.TextSelector(),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )
    return True

async def async_setup_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """Set up Venice AI from a config entry."""
    http_client = get_async_client(hass)
    
    api_key = entry.data.get(CONF_API_KEY) # Uses CONF_API_KEY from homeassistant.const
    if not api_key:
        _LOGGER.error("API Key not found in config entry data.")
        # According to HA dev docs, for auth issues during setup, return False.
        # Raising ConfigEntryAuthFailed is for re-authentication flows.
        return False

    venice_client = AsyncVeniceAIClient(
        api_key=api_key,
        http_client=http_client
        # base_url can be added if you make it configurable
    )

    try:
        await venice_client.models.list()
        _LOGGER.info("Successfully connected to Venice AI and listed models.")
    except AuthenticationError as err:
        _LOGGER.error("Venice AI API Key authentication failed during setup: %s", err)
        return False # Authentication failure
    except VeniceAIError as err:
        _LOGGER.error("Failed to connect or communicate with Venice AI API during setup: %s", err)
        raise ConfigEntryNotReady(f"Venice AI API error: {err}") from err
    except Exception as err:
        _LOGGER.exception("Unexpected error during Venice AI setup client test.")
        raise ConfigEntryNotReady(f"Unexpected error setting up Venice AI: {err}") from err

    entry.runtime_data = venice_client # type: ignore[assignment]

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # If your AsyncVeniceAIClient has an explicit close method that needs to be called:
    # client: AsyncVeniceAIClient | None = entry.runtime_data
    # if client and hasattr(client, 'close') and callable(client.close):
    #    await client.close()

    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle config entry reload."""
    await hass.config_entries.async_reload(entry.entry_id)