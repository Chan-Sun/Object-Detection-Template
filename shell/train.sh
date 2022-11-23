time=$(date "+%Y%m%d")
# time=2022041
work_folder=./work_dir/${time}

NVIDIA_VISIBLE=0,1 bash ./tools/dist_train.sh \
    /home/sunchen/Projects/ECCV_OOD/configs/swin/swin_base_stronger_baseline.py\
    2 \
    --work-dir  ${work_folder}/swinb_sbaseline
# done

python ./tools/train.py \
    /home/sunchen/Projects/ECCV_OOD/configs/swin/swin_base_stronger_baseline.py \
    --gpu-ids 0 \
    --work-dir ${work_folder}/swinb_sbaseline