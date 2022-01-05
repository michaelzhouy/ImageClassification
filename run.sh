nohup python3 -u train.py \
--data_dir='../data/' \
--model_dir='./model/model_pth/' \
--epochs=20 \
--model='swin_base_patch4_window7_224' \
--batch_size=128 \
--input_size=224 \
--LR=1e-3 \
--num_workers=4 \
--cuda='1,2,3,4' >> log 2>&1 &