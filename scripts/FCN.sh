export CUDA_VISIBLE_DEVICES=3

model_name=FCN
date='20240124'
train_epochs=100
batch_size=64
#
declare -A way_mode
way_mode[truncate_pre]='min'
way_mode[truncate_outer]='min'
way_mode[truncate_post]='min'
way_mode[zeropad_pre]='max'
way_mode[zeropad_outer]='max'
way_mode[zeropad_mid]='max'
way_mode[zeropad_post]='max'
way_mode[noisepad_pre]='max'
way_mode[noisepad_outer]='max'
way_mode[noisepad_post]='max'
way_mode[edgepad_pre]='max'
way_mode[edgepad_outer]='max'
way_mode[edgepad_post]='max'
way_mode[interpolate]='max'
way_mode[strf_pad]='max'
way_mode[random_pad]='max'
way_mode[zoom_pad]='max'

#alpha=0
declare -A dataset_length
dataset_length[AllGestureWiimoteX]=11
dataset_length[AllGestureWiimoteY]=8
dataset_length[AllGestureWiimoteZ]=33
dataset_length[GestureMidAirD1]=80
dataset_length[GestureMidAirD2]=80
dataset_length[GestureMidAirD3]=80
dataset_length[GesturePebbleZ1]=115
dataset_length[GesturePebbleZ2]=100
dataset_length[PickupGestureWiimoteZ]=29
dataset_length[PLAID]=100
dataset_length[ShakeGestureWiimoteZ]=41
dataset_length[AsphaltObstacles]=111
dataset_length[AsphaltPavementType]=96
dataset_length[AsphaltRegularity]=95

for way in truncate_pre truncate_outer truncate_post zeropad_pre zeropad_outer zeropad_mid \
zeropad_post noisepad_pre noisepad_outer noisepad_post edgepad_pre edgepad_outer edgepad_post \
interpolate strf_pad random_pad zoom_pad
do
for file in AllGestureWiimoteX AllGestureWiimoteY AllGestureWiimoteZ GestureMidAirD1 GestureMidAirD2 \
GestureMidAirD3 GesturePebbleZ1 GesturePebbleZ2 PickupGestureWiimoteZ PLAID ShakeGestureWiimoteZ AsphaltObstacles \
AsphaltPavementType AsphaltRegularity
do
mode=${way_mode[$way]}
python -u ../run.py \
--is_training 1 \
--data_path ../data/$file/ \
--data_name $file \
--model $model_name \
--way $way \
--mode $mode \
--data VARY_UCR \
--batch_size $batch_size \
--pos -1 \
--des 'Exp' \
--itr 1 \
--train_epochs $train_epochs \
--date $date
done
done
for way in s_fourselect
do
for file in AllGestureWiimoteX AllGestureWiimoteY AllGestureWiimoteZ GestureMidAirD1 \
GestureMidAirD2 GestureMidAirD3 GesturePebbleZ1 GesturePebbleZ2 PickupGestureWiimoteZ PLAID ShakeGestureWiimoteZ \
AsphaltObstacles AsphaltPavementType AsphaltRegularity
do
length=100
python -u ../run.py \
--is_training 1 \
--data_path ../data/$file/ \
--data_name $file \
--model $model_name \
--way $way \
--seq_len $length \
--data VARY_UCR \
--batch_size $batch_size \
--pos -1 \
--des 'Exp' \
--itr 1 \
--train_epochs $train_epochs \
--date $date
done
done
#spectral_pooling
for way in spectral
do
for file in AllGestureWiimoteX AllGestureWiimoteY AllGestureWiimoteZ GestureMidAirD1 \
GestureMidAirD2 GestureMidAirD3 GesturePebbleZ1 GesturePebbleZ2 PickupGestureWiimoteZ PLAID ShakeGestureWiimoteZ \
AsphaltObstacles AsphaltPavementType AsphaltRegularity
do
length=${dataset_length[$file]}
python -u ../run.py \
--is_training 1 \
--data_path ../data/$file/ \
--data_name $file \
--model $model_name \
--way $way \
--seq_len $length \
--data VARY_UCR \
--batch_size $batch_size \
--pos -1 \
--des 'Exp' \
--itr 1 \
--train_epochs $train_epochs \
--date $date
done
done
for file in  AllGestureWiimoteX AllGestureWiimoteY AllGestureWiimoteZ GestureMidAirD1 \
GestureMidAirD2 GestureMidAirD3 GesturePebbleZ1 GesturePebbleZ2 PickupGestureWiimoteZ PLAID ShakeGestureWiimoteZ \
AsphaltObstacles AsphaltPavementType AsphaltRegularity
do
for pooling_mode in sp avg max
do
for pos in 1
do
length=${dataset_length[$file]}
python -u ../run.py \
--is_training 1 \
--data_path ../data/$file/ \
--data_name $file \
--model $model_name \
--way zeropad_post \
--pooling_out $length \
--data VARY_UCR \
--batch_size $batch_size \
--pooling_mode $pooling_mode \
--pos $pos \
--des 'Exp' \
--itr 1 \
--norm backward \
--train_epochs $train_epochs \
--date $date
done
done
done

for file in AllGestureWiimoteX AllGestureWiimoteY AllGestureWiimoteZ GestureMidAirD1 \
GestureMidAirD2 GestureMidAirD3 GesturePebbleZ1 GesturePebbleZ2 PickupGestureWiimoteZ PLAID ShakeGestureWiimoteZ \
AsphaltObstacles AsphaltPavementType AsphaltRegularity
do
python -u ../run.py \
--is_training 1 \
--data_path ../data/$file/ \
--data_name $file \
--model $model_name \
--way dwt_nearest_guided_warping_A \
--data DWT \
--batch_size $batch_size \
--pos -1 \
--des 'Exp' \
--itr 1 \
--train_epochs $train_epochs \
--alpha 0.4 \
--beta 1.0 \
--date $date
python -u ../run.py \
--is_training 1 \
--data_path ../data/$file/ \
--data_name $file \
--model $model_name \
--way dwt_nearest_guided_warping_AB \
--data DWT \
--batch_size $batch_size \
--pos -1 \
--des 'Exp' \
--itr 1 \
--train_epochs $train_epochs \
--alpha 0.4 \
--beta 0.7 \
--date $date
done





