# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, ConcatenateIterator
import os
import re
import time
import torch
import subprocess
from PIL import Image
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "vikhyatk/moondream2"
MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/vikhyatk/moondream2-24-03-06.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        print("Loading model weights...")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision="2024-03-06",
            cache_dir=MODEL_CACHE
        )
        self.moondream = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            revision="2024-03-06",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            cache_dir=MODEL_CACHE
        ).to('cuda')
        self.moondream.eval()
        print("Setup took: ", time.time() - start)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input prompt", default="Describe this image"),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        pil_image = Image.open(image)
        image_embeds = self.moondream.encode_image(pil_image)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        thread = Thread(
            target=self.moondream.answer_question,
            kwargs={
                "image_embeds": image_embeds,
                "question": prompt,
                "tokenizer": self.tokenizer,
                "streamer": streamer,
            },
        )
        thread.start()

        for new_text in streamer:
            clean_text = re.sub("<$|<END$", "", new_text)
            yield clean_text
