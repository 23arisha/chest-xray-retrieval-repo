"""
Text preprocessing for MIMIC-CXR findings sections: sentence segmentation,
frequency-capped deduplication, and tokenized train/val/test dict building.

Corresponds to manuscript Section 3.1.1 (training-corpus preprocessing).
Note this is the training-corpus dedup step (max 3 instances per unique
sentence), distinct from the retrieval candidate-pool construction used in
evaluation (see evaluation/01_build_candidate_pools.py), which applies no
such cap.
"""
import random
from collections import Counter

import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


def extract_subsentences(text):
    if not text or text.strip() == "":
        return []
    doc = nlp(text.strip())
    findings = []
    for sent in doc.sents:
        s = sent.text.strip()
        s = " ".join(s.split())
        if len(s.split()) >= 3:
            findings.append(s)
    return findings


def deduplicate_sentences(df, max_per_sentence=3, seed=42):
    df = df.copy()
    df['findings'] = df['findings'].fillna('')

    empty_mask = df['findings'] == ''
    df         = df[~empty_mask].reset_index(drop=True)

    all_rows = []
    for img_id, row in tqdm(df.iterrows(), total=len(df),
                             desc='Collecting sentences'):
        text = row['findings'].strip()
        sents = extract_subsentences(text)
        for sent in sents:
            all_rows.append({
                'img_id':     img_id,
                'sentence':   sent,
                'image_path': row['image_path']
            })

    print(f'Before dedup: {len(all_rows)} pairs')

    counts = Counter(r['sentence'] for r in all_rows)
    print('\nTop 10 most common:')
    for sent, cnt in counts.most_common(10):
        print(f'  {cnt:5d}x  {sent[:70]}')

    random.seed(seed)
    random.shuffle(all_rows)

    seen = Counter()
    kept = []
    for row in all_rows:
        sent = row['sentence']
        if seen[sent] < max_per_sentence:
            kept.append(row)
            seen[sent] += 1

    print(f'After  dedup: {len(kept)} pairs')
    print(f'Reduction:    {100*(1-len(kept)/len(all_rows)):.1f}%')

    return kept, df


def build_sentence_dict_from_rows(kept_rows, df, tokenizer, max_length=64):
    all_img_ids                 = df.index.tolist()
    train_img_ids, temp_img_ids = train_test_split(
        all_img_ids, test_size=0.10, random_state=42
    )
    val_img_ids, test_img_ids   = train_test_split(
        temp_img_ids, test_size=0.50, random_state=42
    )

    train_img_set = set(train_img_ids)
    val_img_set   = set(val_img_ids)
    test_img_set  = set(test_img_ids)

    data_dict  = {}
    train_uids = []
    val_uids   = []
    test_uids  = []

    for uid, row in enumerate(tqdm(kept_rows, desc='Building dict')):
        img_id   = row['img_id']
        img_path = row['image_path']

        encoded = tokenizer(
            row['sentence'],
            max_length            = max_length,
            padding               = 'max_length',
            truncation            = True,
            return_attention_mask = True
        )

        data_dict[uid] = {
            'token_ids':      encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'image_path':     img_path,
            'image_id':       img_id,
            'sentence':       row['sentence']
        }

        if img_id in train_img_set:
            train_uids.append(uid)
        elif img_id in val_img_set:
            val_uids.append(uid)
        else:
            test_uids.append(uid)

    print(f'Train: {len(train_uids)} | Val: {len(val_uids)} | Test: {len(test_uids)}')

    return {
        'data_dict':  data_dict,
        'data_split': {
            'train_uids': train_uids,
            'val_uids':   val_uids,
            'test_uids':  test_uids
        },
        'vocab_size': tokenizer.vocab_size
    }
