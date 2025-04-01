python3 train_instantaneous.py -e G -s 789 -d G &

for i in {0..20}; do
    mkdir -p log && CUDA_VISIBLE_DEVICES=$(($i % 2 + 2)) nohup python3 train_instantaneous.py -e E -s $((770 + $i)) -d E > log/F_${i}.log 2>&1 &
done

# mkdir -p log && CUDA_VISIBLE_DEVICES=$(($i % 2 + 2)) nohup python3 train_instantaneous.py -e E -s $((770 + $i)) -d E > log/F_${i}.log 2>&1 &
# --
# 
# python3 train_instantaneous.py -e I -s 770 -d I_0
# CUDA_VISIBLE_DEVICES=2 python3 train_instantaneous.py -e H -s 771 -d H
# idx=3
# CUDA_VISIBLE_DEVICES=${idx} nohup python3 train_instantaneous.py -e E -s $((770 + idx)) -d E > log/E_${idx}.log 2>&1 &   


# # log/E_3.log: cannot overwrite existing file
# CUDA_VISIBLE_DEVICES=2 python3 train_instantaneous.py -e E -s 774 -d E &
# CUDA_VISIBLE_DEVICES=3 python3 train_instantaneous.py -e D -s 7
# CUDA_VISIBLE_DEVICES=3 python train_instantaneous.py -e E -s 770
# CUDA_VISIBLE_DEVICES=3 python train_instantaneous.py -e F -s 770