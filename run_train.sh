CONFIG_FILE=config-main.yml
MODEL_DIR=run
# CHECKPOINT=11


python main.py train --save_dir $MODEL_DIR \
                     --config $CONFIG_FILE \
                    #  --checkpoint $CHECKPOINT