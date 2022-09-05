from typing import Dict, List, Tuple
from unittest import TestCase

from parameterized import parameterized

from instruction_ner.core.datatypes import Span
from instruction_ner.formatters import (
    EntityExtractTaskFormatter,
    EntityTypeTaskFormatter,
    NERTaskFormatter,
)
from instruction_ner.formatters.Answer import AnswerFormatter


class TestReader(TestCase):
    maxDiff = None

    @parameterized.expand(
        [
            (
                "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .",
                [(9, 14, "LOC"), (31, 36, "LOC")],
                "en",
                "JAPAN is a LOC, CHINA is a LOC.",
            ),
            (
                "Bitar pulled off fine saves whenever they did .",
                [(0, 5, "PER")],
                "en",
                "Bitar is a PER.",
            ),
        ]
    )
    def test_format_answer_from_spans(
        self,
        context: str,
        entity_spans: List[Tuple[int, int, str]],
        language: str,
        output_true: str,
    ):

        answer_pred = AnswerFormatter.from_spans(
            context=context, entity_spans=entity_spans, language=language
        )

        self.assertEqual(answer_pred, output_true)

    @parameterized.expand(
        [
            (
                "AL-AIN , United Arab Emirates 1996-12-06",
                [(0, 6, "LOC"), (9, 29, "LOC")],
                "ru",
            )
        ]
    )
    def test_raise_error_when_wrong_language(
        self, context: str, entity_spans: List[Tuple[int, int, str]], language
    ):

        self.assertRaises(
            ValueError, AnswerFormatter.from_spans, context, entity_spans, language
        )

    @parameterized.expand(
        [
            (
                {"LOC": ["JAPAN"], "PER": ["CHINA"]},
                "en",
                "JAPAN is a LOC, CHINA is a PER.",
            ),
            ({"PER": ["Bitar"]}, "en", "Bitar is a PER."),
        ]
    )
    def test_format_answer_from_values(
        self, entity_values: Dict[str, List[str]], language: str, output_true: str
    ):

        answer_pred = AnswerFormatter.from_values(
            entity_values=entity_values, language=language
        )

        self.assertEqual(answer_pred, output_true)

    @parameterized.expand(
        [
            (
                "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .",
                "Please do NER",
                "Sentence: SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . "
                "Instruction: Please do NER Options: LOC, PER, MISC",
            ),
            (
                "Bitar pulled off fine saves whenever they did .",
                "Please please do NER",
                "Sentence: Bitar pulled off fine saves whenever they did . "
                "Instruction: Please please do NER Options: LOC, PER, MISC",
            ),
        ]
    )
    def test_format_instance_ner(self, context: str, instruction: str, output_true):

        entity_values = None
        entity_spans = [Span.from_json({"start": 0, "end": 5, "label": "LOC"})]

        options = ["LOC", "PER", "MISC"]

        formatter = NERTaskFormatter()
        instance_pred = formatter.format_instance(
            context=context,
            instruction=instruction,
            entity_values=entity_values,
            entity_spans=entity_spans,
            options=options,
        )

        return self.assertEqual(str(instance_pred), output_true)

    @parameterized.expand(
        [
            (
                "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .",
                "Please do NER",
                "Sentence: SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . "
                "Instruction: Please do NER: EMPTY Options: LOC, PER, MISC EMPTY is a LOC.",
            ),
            (
                "Bitar pulled off fine saves whenever they did .",
                "Please please do NER",
                "Sentence: Bitar pulled off fine saves whenever they did . "
                "Instruction: Please please do NER: EMPTY Options: LOC, PER, MISC EMPTY is a LOC.",
            ),
        ]
    )
    def test_format_instance_type(
        self, context: str, instruction: str, output_true: str
    ):
        entity_values = {"LOC": ["EMPTY"]}
        entity_spans = [Span.from_json({"start": 0, "end": 5, "label": "LOC"})]

        options = ["LOC", "PER", "MISC"]

        formatter = EntityTypeTaskFormatter()
        instance_pred = formatter.format_instance(
            context=context,
            instruction=instruction,
            entity_values=entity_values,
            entity_spans=entity_spans,
            options=options,
        )

        return self.assertEqual(str(instance_pred), output_true)

    @parameterized.expand(
        [
            (
                "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .",
                "Please extract entities",
                "Sentence: SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . "
                "Instruction: Please extract entities EMPTY.",
            ),
            (
                "Bitar pulled off fine saves whenever they did .",
                "Please please extract entity words",
                "Sentence: Bitar pulled off fine saves whenever they did . "
                "Instruction: Please please extract entity words EMPTY.",
            ),
        ]
    )
    def test_format_instance_extract(
        self, context: str, instruction: str, output_true: str
    ):
        entity_values = {"LOC": ["EMPTY"]}
        entity_spans = [Span.from_json({"start": 0, "end": 5, "label": "LOC"})]

        options = ["LOC", "PER", "MISC"]

        formatter = EntityExtractTaskFormatter()
        instance_pred = formatter.format_instance(
            context=context,
            instruction=instruction,
            entity_values=entity_values,
            entity_spans=entity_spans,
            options=options,
        )

        return self.assertEqual(str(instance_pred), output_true)
