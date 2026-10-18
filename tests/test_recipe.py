import json
import tempfile
import unittest
from pathlib import Path

from prepare_data import inspect_corpus

ROOT = Path(__file__).resolve().parents[1]


class PicoRecipeTests(unittest.TestCase):
    def test_config_describes_exact_15m_parameter_count(self):
        recipe = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        architecture = recipe["architecture"]
        self.assertEqual(architecture["tokenizer"], "byte")
        hidden = architecture["hidden_size"]
        layers = architecture["layers"]
        vocab = architecture["vocab"]
        context = architecture["context"]
        feed_forward = architecture["intermediate_size"]
        layer_norms = 9 * hidden + feed_forward
        attention_and_mlp = 4 * hidden * hidden + 2 * hidden * feed_forward
        per_layer = layer_norms + attention_and_mlp
        parameters = vocab * hidden + context * hidden + 2 * hidden + layers * per_layer
        self.assertEqual(parameters, recipe["parameter_count"])
        self.assertEqual(parameters, 14_983_680)

    def test_default_budget_is_explicit_and_training_uses_cuda(self):
        recipe = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        training = recipe["training"]
        self.assertEqual(training["device"], "cuda")
        self.assertEqual(training["optimizer"], "adamw")
        tokens = training["steps"] * training["batch_size"] * training["seq_len"] * training["gradient_accumulation_steps"]
        self.assertEqual(tokens, 163_840_000)

    def test_corpus_validation_hashes_utf8_without_loading_full_file(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            content = ("Pico local training corpus. " * 40).encode("utf-8")
            corpus.write_bytes(content)
            size, digest = inspect_corpus(corpus, minimum_bytes=100)
            self.assertEqual(size, len(content))
            self.assertEqual(len(digest), 64)
            corpus.write_bytes(b"\xff")
            with self.assertRaises(UnicodeDecodeError):
                inspect_corpus(corpus, minimum_bytes=0)


if __name__ == "__main__":
    unittest.main()
