PEAK_LR=0.0001
MAX_EPOCH=120
PATIENCE=40

MAX_SENTENCES=8
UPDATE_FREQ=1
MAX_POSITIONS=256

PREFIX_LEN=5
ITER_NUM=5

LOSS_ENC_S=1
LOSS_DEC_S=1
LOSS_DEC_X=2

SEED=128


MODEL_NAME=pflen${PREFIX_LEN}_iter${ITER_NUM}_loss${LOSS_ENC_S}_${LOSS_DEC_S}_${LOSS_DEC_X}_lr${PEAK_LR}_bsz${MAX_SENTENCES}_seed${SEED}
SABDAB_DATA_DIR=abgnn/finetune/exp2-hern/nanobody
PRETRAINED_FILE=checkpoints/pretrained/new_RoPE_decoder_abbert_warmup10000_lr0.0006_maxsen128_upfreq4_samplenone_clip0.0/checkpoint_best.pt
SAVE_DIR=checkpoints/exp2_nanobody/${MODEL_NAME}
FAIRSEQ_MODELS_DIR=fairseq_models

echo $(which fairseq-train) 
fairseq-train --sabdab-data $SABDAB_DATA_DIR \
  --user-dir $FAIRSEQ_MODELS_DIR --finetune \
  --task antibody_generation_task \
  --criterion antibody_generation_loss \
  --arch antibody_roberta_base \
  --finetune-bert-scheme prefix_tuning --pre-seq-len $PREFIX_LEN \
  --refine-iteration $ITER_NUM --block_size 8 \
  --loss-scale-enc $LOSS_ENC_S --loss-scale-dec-sloss $LOSS_DEC_S --loss-scale-dec-xloss $LOSS_DEC_X \
  --optimizer adam --clip-norm 1.0 \
  --lr-scheduler fixed --lr $PEAK_LR \
  --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ --max-positions $MAX_POSITIONS --max-epoch $MAX_EPOCH \
  --log-format simple --log-interval 5 \
  --valid-subset valid,test --skip-invalid-size-inputs-valid-test \
  --save-interval 1 --save-dir $SAVE_DIR \
  --finetune-from-model $PRETRAINED_FILE \
  --patience $PATIENCE --tensorboard-logdir $SAVE_DIR/tensorboard  \
  --seed $SEED --num-workers 0

#python inference.py --cktpath $SAVE_DIR/checkpoint_best.pt > $SAVE_DIR/test_best.txt
#for infer_epoch in {1..40}
#do
#    python inference.py --cktpath $SAVE_DIR/checkpoint${infer_epoch}.pt > $SAVE_DIR/test${infer_epoch}.txt
#done
