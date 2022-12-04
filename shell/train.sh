time=$(date "+%Y%m%d")
# time=2022041
work_folder=./work_dir/${time}

NVIDIA_VISIBLE=0,1 bash ./shell/dist_train.sh \
    ./configs/frcn.py \
    2 \
    --work-dir  ${work_folder}/frcn
# done

# python ./tools/train.py \
#     /home/sunchen/Projects/ECCV_OOD/configs/swin/swin_base_stronger_baseline.py \
#     --gpu-ids 0 \
#     --work-dir ${work_folder}/swinb_sbaseline