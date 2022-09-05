from pathlib import Path
from typing import Any, Dict, List, Union

from instruction_ner.core.datatypes import DatasetField, Span
from instruction_ner.core.reader import Reader


# TODO this code duplicates CONLL parser a lot. Think about moving to ABC class
class MITReader(Reader):

    supported_extensions = [".bio"]
    sentence_separator = ""
    token_separator = "	"
    label_prefix = "-"
    MIN_SENTENCE_LENGTH = 3

    def read(self, data: List[str]) -> List[Dict[str, Any]]:
        """
        Main function of MIT Reader. Takes lines of strings as input, splits them into sentences.
        And return data in unified format
        :param data: List of string from .bio file
        :return: List of sentences with spans and entity values
        """

        sentences = self._split_by_sentences(data)
        sentences = [
            sentence
            for sentence in sentences
            if len(sentence) > self.MIN_SENTENCE_LENGTH
        ]

        data_processed = []
        for sentence in sentences:
            text, entity_spans = self._get_text_and_spans_from_sentence(sentence)
            entity_values = self._get_entity_values_from_text(text, entity_spans)

            dataset_item = {
                DatasetField.CONTEXT.value: text,
                DatasetField.ENTITY_VALUES.value: entity_values,
                DatasetField.ENTITY_SPANS.value: entity_spans,
            }
            data_processed.append(dataset_item)

        return data_processed

    def read_from_file(self, path_to_file: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Wrapper around self.read(). Read "path_to_file" and run self.read()
        :param path_to_file: string or Path
        :return: List of Dicts where each element is a sentence with entities
        """
        if isinstance(path_to_file, str):
            path_to_file = Path(path_to_file)

        if path_to_file.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Expected file to be on of {self.supported_extensions}. Got {path_to_file.suffix}"
            )

        with open(path_to_file, "r") as f:
            file_lines = f.readlines()
            file_lines = [x.strip("\n") for x in file_lines]

        data = self.read(data=file_lines)

        return data

    def _split_by_sentences(self, document: List[str]) -> List[List[str]]:
        """
        Get list of sentence tokes from document
        :param document: initial lines
        :return: List of sentences. Each sentence is a list of tokens
        """
        sentences = []

        i = 0
        sentence_tokens: List[str] = []
        while i < len(document):
            line = document[i]

            if line == self.sentence_separator:
                sentences.append(sentence_tokens)
                sentence_tokens = []
                i += 1
                continue

            sentence_tokens.append(line)
            i += 1

        if len(sentence_tokens) != 0:
            sentences.append(sentence_tokens)

        return sentences

    def _get_text_and_spans_from_sentence(self, sentence: List[str]):

        text_tokens = []
        entity_spans = []

        current_start_idx = 0
        entity_length = 0
        entity_label = None
        for token_line in sentence:
            tokens = token_line.split(self.token_separator)

            if len(tokens) != 2:
                raise ValueError(
                    f"Expected 2 elements after split, got {len(tokens)}: {token_line}"
                )

            token, label = (
                tokens[-1],
                tokens[0],
            )  # this is the only difference from CONLL
            text_tokens.append(token)

            if label == "O":

                if entity_label is not None and entity_length != 0:
                    # It means that previous token was entity and we should create Span
                    entity_span = Span(
                        start=current_start_idx,
                        end=current_start_idx + entity_length - 1,
                        label=entity_label,
                    )
                    entity_spans.append(entity_span)

                    entity_label = None
                    entity_length = 0

                    current_start_idx = entity_span.end + 1

                current_start_idx += len(token)
                current_start_idx += (
                    1  # Because in the end we join them with space symbol
                )
                continue

            # If we have two entities one after another
            if label.startswith("B" + self.label_prefix) and entity_label is not None:
                entity_span = Span(
                    start=current_start_idx,
                    end=current_start_idx + entity_length - 1,
                    label=entity_label,
                )
                entity_spans.append(entity_span)

                entity_length = 0
                current_start_idx = entity_span.end + 1

            tag, entity_label = label.split(self.label_prefix, maxsplit=1)
            entity_length += len(token) + 1

        # if last tokens of sentence are entity
        if entity_length is not None and entity_label is not None:
            entity_span = Span(
                start=current_start_idx,
                end=current_start_idx + entity_length - 1,
                label=entity_label,
            )
            entity_spans.append(entity_span)

        text = " ".join(text_tokens)

        return text, entity_spans
