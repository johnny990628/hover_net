nohup python run_infer.py \
--gpu=1 \
--model_mode=original \
--model_path=/data1/johnny99457/hover_net/models/pretrained/hovernet_original_consep.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=/data1/johnny99457/hover_net/dataset/TCGA-49-4505-01Z-00-DX4_patches_40x \
--output_dir=/data1/johnny99457/hover_net/dataset/TCGA-49-4505-01Z-00-DX4_result_40x/consep > consep.log 2>&1 &
