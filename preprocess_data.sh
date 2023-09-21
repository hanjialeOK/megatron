jsonfile="data/eight.files3.json"
vocabfile="data/bert-large-cased-vocab.txt"
prefix="data/data-small3"

python3 tools/preprocess_data.py \
        --input $jsonfile \
        --output-prefix $prefix \
        --vocab-file $vocabfile \
        --dataset-impl mmap \
        --tokenizer-type BertWordPieceLowerCase \
        --split-sentences \
        --workers 1

vocabfile="data/gpt2-vocab.json"
mergefile="data/gpt2-merges.txt"
prefix="data/my-gpt2"

python3 tools/preprocess_data.py \
        --input $jsonfile \
        --output-prefix $prefix \
        --vocab-file $vocabfile \
        --dataset-impl mmap \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file $mergefile \
        --append-eod \
        --workers 1
