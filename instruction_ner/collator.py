from typing import Any, Dict, List

import transformers

from instruction_ner.core.datatypes import Instance


class Collator:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        tokenizer_kwargs: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(
        self,
        batch: List[Instance],
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        """
        Tokenizes context, questions, answer and puts them into batch
        along with original instances
        :param batch: List of Instances
        :return: Batch
        """

        context_list = []
        question_list = []
        answer_list = []

        for instance in batch:
            context_list.append(instance.context)
            question_list.append(instance.question)
            answer_list.append(instance.answer)

        tokenized_batch = self.tokenizer(
            context_list, question_list, **self.tokenizer_kwargs
        )

        with self.tokenizer.as_target_tokenizer():
            answers = self.tokenizer(answer_list, **self.tokenizer_kwargs)

        tokenized_batch["answers"] = answers
        tokenized_batch["instances"] = batch

        return tokenized_batch
