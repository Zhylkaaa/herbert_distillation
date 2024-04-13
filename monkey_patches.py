import warnings
import random
from typing import List

from transformers import (
    DataCollatorForWholeWordMask,
    HerbertTokenizer,
    HerbertTokenizerFast
)


class HerbertDataCollatorForWholeWordMask(DataCollatorForWholeWordMask):
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
                Get 0/1 labels for masked tokens with whole word mask proxy
                """
        if not isinstance(self.tokenizer, (HerbertTokenizer, HerbertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for HerbertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        special_tokens = [val for key, val in self.tokenizer.special_tokens_map.items()
                          if key not in ['unk_token', 'mask_token']]
        last_token = '</w>'
        for i, token in enumerate(input_tokens):
            if token in special_tokens:
                continue

            if len(cand_indexes) >= 1 and not last_token.endswith("</w>"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
            last_token = token

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels
