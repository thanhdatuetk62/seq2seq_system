CONFIG_FILE=config-main.yml
SRC_PATH=../data/iwslt_en_vi/tst2013.en
SAVE_PATH=output.txt
MODEL_DIR=run
CHECKPOINT=17


python main.py infer --save_dir $MODEL_DIR \
                     --config $CONFIG_FILE \
                     --src_path $SRC_PATH \
                     --save_path $SAVE_PATH \
                     --sos_token "<vi>" \
                    #  --checkpoint $CHECKPOINT