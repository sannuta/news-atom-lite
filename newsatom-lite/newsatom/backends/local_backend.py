"""Local model backend. Requires: pip install torch transformers. GPU recommended."""
from .base import ModelBackend


class LocalBackend(ModelBackend):

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError("--model-path is required for the local backend.")
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Local backend requires: pip install torch transformers")

        print(f"   Loading model from {self.model_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )
        self._model.eval()

    def generate(self, prompt: str) -> str:
        import torch
        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = outputs[0][prompt_length:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
