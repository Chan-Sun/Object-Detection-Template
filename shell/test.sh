time=$(date "+%Y%m%d")
# time=2022041
work_folder=./work_dir/${time}

NVIDIA_VISIBLE=0,1 bash ./shell/dist_test.sh \
    //home/sunchen/Projects/XMU/configs/frcn.py \
    /home/sunchen/Projects/XMU/work_dir/20221204/frcn/epoch_12.pth \
    2 \
    --work-dir  ${work_folder}/frcn \
    --out test.pkl