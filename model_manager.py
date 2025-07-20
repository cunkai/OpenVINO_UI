# src-tauri/backend/model_manager.py
import os
import sys
import openvino_genai as ov_genai
from pathlib import Path
import traceback
import queue
import threading
from PIL import Image


class ModelManager:
    """
    A class to manage loading and running inference on AI models.
    This encapsulates the logic for different model types and pipelines,
    making it easier to extend in the future.
    """
    def __init__(self):
        self.pipeline = None
        self.model_id = ""

    def Get_OV_Version(self):
        return ov_genai.get_version()

    def load_model(self, model_path_str: str, device: str) -> bool:

        """
        Loads a model from the given path onto the specified device.
        Determines whether to use a language or multimodal pipeline.
        """
        model_path = Path(model_path_str)
        if not model_path.is_dir():
            print(f"Error: Model path not found: {model_path}")
            return False

        if self.model_id == model_path.name and self.pipeline is not None:
            print(f"Model '{model_path.name}' is already loaded.")
            return True

        try:
            print(f"Loading model '{model_path.name}' on device '{device}'...")

            vision_model_file = model_path / 'openvino_vision_embeddings_model.bin'
            is_multimodal = vision_model_file.exists()

            os.environ["OV_GPU_THREADING_POLICY"] = "SINGLE"
            schedulerConfig = ov_genai.SchedulerConfig()
            # schedulerConfig.dynamic_split_fuse = False
            schedulerConfig.enable_prefix_caching = False
            # schedulerConfig.use_cache_eviction = False
            if is_multimodal:
                print(f"'{model_path.name}' is a multimodal model. Using VLMPipeline.")
                self.pipeline = ov_genai.VLMPipeline(models_path=str(model_path), device=device, config={"scheduler_config": schedulerConfig})
            else:
                print(f"'{model_path.name}' is a language model. Using LLMPipeline.")
                self.pipeline = ov_genai.LLMPipeline(models_path=str(model_path), device=device, config={"scheduler_config": schedulerConfig})

            self.model_id = model_path.name
            print(f"Model '{self.model_id}' loaded successfully on {device}.")
            return True
        except Exception as e:
            self.pipeline = None
            self.model_id = ""
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False

    def is_loaded(self) -> bool:
        """Checks if a model pipeline is currently loaded."""
        return self.pipeline is not None

    def get_model_id(self) -> str:
        """Returns the ID of the currently loaded model."""
        return self.model_id

    def get_tokenizer(self):
        """Returns the tokenizer of the currently loaded model."""
        if self.is_loaded():
            return self.pipeline.get_tokenizer()
        return None

    def stream_generate(self, prompt: str, images: list[Image.Image], queue: queue.Queue, stop_event: threading.Event, **kwargs):
        """
        Runs inference and streams results to a queue.
        This method encapsulates the backend-specific streamer logic.
        """
        if not self.is_loaded():
            raise RuntimeError("No model is loaded to run generation.")

        def streamer(token: str, *args):
            if stop_event.is_set():
                return ov_genai.StreamingStatus.STOP
            queue.put({"type": "token", "content": token})
            return ov_genai.StreamingStatus.RUNNING

        ai_text = ""
        try:
            # Check if the pipeline is multimodal and if there are images to process
            if images and isinstance(self.pipeline, ov_genai.VLMPipeline):
                ai_text = self.pipeline.generate(prompt, images=images, streamer=streamer, **kwargs)
            else:
                # If no images or not a VLM pipeline, call generate without the images argument
                ai_text = self.pipeline.generate(prompt, streamer=streamer, **kwargs)
            
            # print(f"--- AI Talk ---\n{ai_text}\n---------------------------")

        except Exception as e:
            # Put the exception on the queue so the route handler can see it.
            queue.put({"type": "error", "content": str(e)})
            traceback.print_exc()
        finally:
            # Signal the end of the generation.
            queue.put(None)
