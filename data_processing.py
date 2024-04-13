import os
import shutil
import glob
import json
from argparse import ArgumentParser
from functools import partial

from speakleash import Speakleash
from datasets import load_dataset
from lm_dataformat import Archive, Reader


def is_high_quality(x, quality_threshold=0.9):
    return x['meta']['quality_ai']['HIGH'] >= quality_threshold


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datasets_path', default='datasets')
    parser.add_argument('--output_path', default='distillation_corpus')
    parser.add_argument('--datasets_file', default='corpus_datasets.json')
    parser.add_argument('--quality_threshold', default=0.9)
    args = parser.parse_args()

    with open(args.datasets_file) as f:
        datasets = json.load(f)

    sl = Speakleash(args.datasets_path)
    # download datasets
    for d in datasets:
        sl.get(d).data

    allowed_fields = ['quality', 'quality_ai']

    # sanitize metadata
    for d in datasets:
        ar = Archive('tmp_data')
        dataset_path = os.path.join(args.datasets_path, f'{d}.jsonl.zst')
        for text, metadata in Reader(dataset_path).stream_data(get_meta=True):
            keys = list(metadata.keys())
            for k in keys:
                if k not in allowed_fields:
                    del metadata[k]
            ar.add_data(text, meta=metadata)
        ar.commit()
        file = glob.glob('tmp_data/*.jsonl.zst')[0]
        shutil.move(file, dataset_path)

    hf_dataset = dataset = load_dataset('json',
                                        data_files=[os.path.join(args.datasets_path, f'{d}.jsonl.zst')
                                                    for d in datasets])
    high_quality = partial(is_high_quality, quality_threshold=args.quality_threshold)
    hf_dataset = hf_dataset.filter(high_quality)
    hf_dataset = hf_dataset['train'].train_test_split(train_size=0.8)
    hf_dataset['validation'] = hf_dataset.pop('test')
    hf_dataset.save_to_disk(args.output_path)
