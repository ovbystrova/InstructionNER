from typing import Any, Dict, List

import transformers

from src.core.datatypes import Instance


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

        context_list = []
        question_list = []
        answer_list = []

        for instance in batch:
            context_list.append(instance.context)
            question_list.append(instance.question)
            answer_list.append(instance.answer)

        tokenized_batch = self.tokenizer(
            question_list, context_list, **self.tokenizer_kwargs
        )

        tokenized_batch["instances"] = batch

        return tokenized_batch
