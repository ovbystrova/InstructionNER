import json
from pathlib import Path
from unittest import TestCase

from parameterized import parameterized

from instruction_ner.model import Model


class TestModel(TestCase):

    maxDiff = None
    test_data_dir = Path(__file__).parent / "data" / "model"

    with open(test_data_dir / "sample_1.json") as f:
        data = json.load(f)

    options = data["options"]
    instruction = data["instruction"]
    text = data["text"]
    model_name = data["model_name"]

    generation_kwargs = data["generation_kwargs"]
    prediction_true = data["prediction_true"]

    @parameterized.expand(
        [
            (text, options, instruction, prediction_true),
        ]
    )
    def test_model_predict(self, text, options, instruction, prediction_true):

        model = Model(
            model_path_or_name=self.model_name, tokenizer_path_or_name=self.model_name
        )

        pred_text, _ = model.predict(
            text=text,
            instruction=instruction,
            options=options,
            generation_kwargs=self.generation_kwargs,
        )

        self.assertEqual(pred_text, prediction_true)
