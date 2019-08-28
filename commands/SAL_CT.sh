$DIR="../data/Spanish/SAL_CT"
$DATA="../data/Spanish"

for i in {1..20} ; do
    python2 ../helper_scripts/pickKTokensRev.py --input $DIR/to_annotate_v${i}.1_LC.conll --k 200 --output $DIR/to_annotate_v${i}.1_200.conll

    python2 ../helper_scripts/SimulateAnnotations.py --input $DIR/to_annotate_v${i}.1_200.conll --output $DIR/v${i}.1.conll

    PREV=`expr $i - 1`

    python2 ../helper_scripts/removeAnnotatedSents.py --input $DIR//unlabel_v${PREV}.1.conll --annotated $DIR/v${i}.1.conll --output $DIR/unlabel_v${i}.1.conll

	if [ "$i" -gt 1 ]
	then
	python2 ../helper_scripts/CombineAnnotatedFiles.py --files $DIR/Entropy_v${PREV}.1.conll   $DIR/v${i}.1.conll --output $DIR/Entropy_v${i}.1.conll
    else
    cp $DIR/v1.1.conll $DIR/Entropy_v1.1.conll
    fi

    #Train the NER Model Using FineTune
    MODEL_NAME="200_SAL_CT_spa_${i}.1_finetune"
    python -u ../main.py \
        --dynet-seed 3278657 \
        --word_emb_dim 100 \
        --batch_size 10 \
        --model_name ${MODEL_NAME} \
        --lang es \
        --fixedVocab \
        --fineTune \
        --test_conll \
        --tot_epochs 1000 \
	    --aug_lang_train_path $DATA/vocab.conll \
        --init_lr 0.015 \
        --load_from_path ../saved_models/spanish_full_transfer_baseline.model \
        --valid_freq 1300 \
        --pretrain_emb_path $DATA/esp.vec \
        --dev_path $DATA/esp.dev \
        --test_path $DATA/esp.test \
        --train_path $DIR/Entropy_v${i}.1.conll  2>&1 | tee ${MODEL_NAME}.log


    #Run the Active Learning Session
    NEW=`expr $i + 1`
    #!/usr/bin/env bash
    MODEL_NAME="200_SAL_spa_${i}.1_finetune_activelearning"
    python -u ../main.py \
        --dynet-seed 3278657 \
        --mode test_1 \
        --fixedVocab \
        --aug_lang_train_path $DATA/vocab.conll \
  	    --word_emb_dim 100 \
        --batch_size 10 \
        --model_name ${MODEL_NAME} \
        --lang es \
        --tot_epochs 200 \
        --load_from_path  ../saved_models/200_SAL_CT_spa_${i}.1_finetune.model \
        --valid_freq 1300 \
        --pretrain_emb_path   $DATA/esp.vec  \
        --dev_path $DATA/esp.dev  \
        --test_path $DIR/unlabel_v${i}.1.conll \
        --to_annotate $DIR/to_annotate_v${NEW}.1.conll \
        --test_conll \
        --k 200 \
        --SPAN_wise \
        --train_path $DIR/Entropy_v${i}.1.conll  2>&1 | tee ${MODEL_NAME}.log

done
