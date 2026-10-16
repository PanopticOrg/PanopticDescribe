from __future__ import annotations

from enum import Enum

from PIL import Image


class Engine(Enum):
    florence2 = 'florence2'
    vlm = 'vlm'


class Detail(Enum):
    short = 'short'
    detailed = 'detailed'
    very_detailed = 'very_detailed'


FLORENCE_TASKS = {
    Detail.short: '<CAPTION>',
    Detail.detailed: '<DETAILED_CAPTION>',
    Detail.very_detailed: '<MORE_DETAILED_CAPTION>',
}


def resolve_device(torch):
    if torch.cuda.is_available():
        return 'cuda', torch.bfloat16
    if torch.backends.mps.is_available():
        return 'mps', torch.float16
    return 'cpu', torch.float32


class DescribeEngine:
    # number of images handed to describe() at once
    batch_size = 1

    def describe(self, images: list[Image.Image]) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class HFEngine(DescribeEngine):
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.device, self.dtype = resolve_device(torch)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()

    def close(self) -> None:
        del self.model
        if self.device == 'cuda':
            self.torch.cuda.empty_cache()


class FlorenceEngine(HFEngine):
    """Florence-2: small captioning model, English only, fast even on CPU."""

    def __init__(self, model_id: str, detail: Detail):
        super().__init__(model_id)
        self.task = FLORENCE_TASKS[detail]
        self.batch_size = 8 if self.device != 'cpu' else 2

    def describe(self, images: list[Image.Image]) -> list[str]:
        inputs = self.processor(text=[self.task] * len(images), images=images, return_tensors='pt')
        inputs = inputs.to(self.device, self.dtype)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=256, num_beams=3, do_sample=False)
        return [text.strip() for text in self.processor.batch_decode(output, skip_special_tokens=True)]


class VlmEngine(HFEngine):
    """A chat vision-language model (Qwen3.5 by default) driven by a free prompt, in any language."""

    def __init__(self, model_id: str, prompt: str, max_new_tokens: int):
        super().__init__(model_id)
        self.processor.tokenizer.padding_side = 'left'
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.batch_size = 8 if self.device != 'cpu' else 1

    def describe(self, images: list[Image.Image]) -> list[str]:
        messages = [[{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': self.prompt},
        ]}] for image in images]
        # enable_thinking is read by the Qwen3.5 template and ignored by the others
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt',
            padding=True, enable_thinking=False,
        ).to(self.device)
        inputs.pop('token_type_ids', None)
        prompt_len = inputs['input_ids'].shape[1]
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                         repetition_penalty=1.05)
        return [text.strip() for text in self.processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True)]


def make_engine(params) -> DescribeEngine:
    if Engine(params.engine) == Engine.vlm:
        return VlmEngine(params.vlm_model, params.prompt, params.max_tokens)
    return FlorenceEngine(params.florence_model, Detail(params.detail))
