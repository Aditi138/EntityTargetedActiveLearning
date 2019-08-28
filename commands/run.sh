MODEL_NAME="test"
DATA="/home/aschaudh/EntityTargetedActiveLearning/data/Spanish"
python -u ../main.py \
    --dynet-seed 3278657 \
    --word_emb_dim 100 \
    --batch_size 10 \
    --model_name ${MODEL_NAME} \
    --lang es \
    --fixedVocab \
    --test_conll \
    --tot_epochs 1000 \
    --aug_lang_train_path $DATA/vocab.conll \
    --init_lr 0.015 \
    --valid_freq 1300 \
    --pretrain_emb_path $DATA/esp.vec \
    --dev_path $DATA/esp.dev \
    --test_path $DATA/esp.test \
    --train_path $DATA/transferred_data.conll  2>&1 | tee ${MODEL_NAME}.log 
