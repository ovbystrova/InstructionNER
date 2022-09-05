from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from instruction_ner.core.datatypes import DatasetField, Span
from instruction_ner.core.reader import Reader


class CONLLReader(Reader):

    token_doc_start = "-DOCSTART-"
    token_separator = " "
    sentence_separator = ""
    label_prefix = "-"
    MIN_SENTENCE_LENGTH = 3

    C_B = 0
    C_LAST = 0

    def read(self, data: List[str]) -> List[Dict[str, Any]]:
        """
        Main function of CONLLReader. Based on list of conll lines extract sentences with their entities
        :param data: List of conll strings (eg. ['-DOCSTART- -X- -X- O', '\n', '\n', 'JAPAN NNP B-NP B-LOC']
        :return: List of Dicts where each element is a sentence with entities
        """

        documents = self._split_lines_by_documents(data)
        sentences = []
        for document in documents:
            sentence_tokens = self._split_documents_by_sentences(document)
            sentences.extend(sentence_tokens)

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

    def read_from_file(self, path_to_file: Union[str, Path]):
        """
        Wrapper around self.read(). Read 'path_to_file' and run self.read()
        :param path_to_file: string or Path
        :return: List of Dicts where each element is a sentence with entities
        """

        if isinstance(path_to_file, str):
            path_to_file = Path(path_to_file)

        if not path_to_file.suffix.endswith(".txt"):
            raise ValueError(f"Expected .txt file, got {path_to_file}")

        with open(path_to_file, "r") as f:
            file_lines = f.readlines()
            file_lines = [x.strip("\n") for x in file_lines]

        data = self.read(data=file_lines)

        return data

    def _split_lines_by_documents(self, file_lines: List[str]) -> List[List[str]]:
        """
        Get List of Documents (list of tokens) from all lines in conll file
        :param file_lines: initial file lines
        :return: List of documents. Each document is a list of tokens
        """

        documents = []

        i = 0
        document_tokens: List[str] = []
        while i < len(file_lines):
            line = file_lines[i]
            if line.startswith(self.token_doc_start):
                i += 2

                if (
                    len(document_tokens) == 0
                ):  # When processing first line document_tokens is empty
                    continue

                documents.append(document_tokens)
                document_tokens = list()
                continue

            document_tokens.append(line)
            i += 1

        if len(document_tokens) != 0:
            documents.append(document_tokens)

        return documents

    def _split_documents_by_sentences(self, document: List[str]) -> List[List[str]]:
        """
        Get list of sentence tokes from document
        :param document: initial document lines
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

    def _get_text_and_spans_from_sentence(
        self, sentence: List[str]
    ) -> Tuple[str, List[Span]]:
        """
        Get raw text from sentence token lines. Along with raw text return list of Span entities
        :param sentence: List of tokens (eg. ['-DOCSTART- -X- -X- O', '\n', '\n', 'JAPAN NNP B-NP B-LOC'])
        :return: raw text and entity Spans
        """

        text_tokens = []
        entity_spans = []

        current_start_idx = 0
        entity_length = 0
        entity_label = None
        for token_line in sentence:
            tokens = token_line.split(self.token_separator)

            if len(tokens) != 4:
                raise ValueError(
                    f"Expected 4 elements after split, got {len(tokens)}: {token_line}"
                )

            token, label = tokens[0], tokens[-1]
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

                self.C_B += 1

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
            self.C_LAST += 1
            entity_span = Span(
                start=current_start_idx,
                end=current_start_idx + entity_length - 1,
                label=entity_label,
            )
            entity_spans.append(entity_span)

        text = " ".join(text_tokens)

        return text, entity_spans
