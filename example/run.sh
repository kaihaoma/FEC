#!/bin/bash

while getopts "s:d:m:b:w:h:" arg; do
  case $arg in
    s) sys=$OPTARG;;
    m) model=$OPTARG;;
    b) batch_size=$OPTARG;;
    d) dataset=$OPTARG;;
    w) num_workers=$OPTARG;;
    h) num_hots=$OPTARG;;
    
  esac
done

case $dataset in 
  criteo)
  echo "criteo"
  num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"
  in_memory_binary_criteo_path="./datasets/bas_dataset/sample_criteo"
  hot_entries_path="./datasets/fec_dataset/hot_entries/criteo_hot_entries.npy"
  ;;
  avazu) 
  echo "avazu"
  num_embeddings_per_feature="7,7,4737,7745,26,8552,559,36,2686408,6729486,8251,5,4,2626,8,9,435,4,68,172,60"
  in_memory_binary_criteo_path="./datasets/bas_dataset/sample_avazu"
  hot_entries_path="./datasets/fec_dataset/hot_entries/avazu_hot_entries.npy"
  ;;
  criteo-tb)
  echo "criteo-tb"
  num_embeddings_per_feature="45833189,36747,17246,7414,20244,3,7115,1442,63,29275262,1572177,345139,10,2209,11268,129,4,975,14,48937458,11316797,40094538,452105,12607,105,36"
  in_memory_binary_criteo_path="./datasets/bas_dataset/sample_criteo-tb"
  hot_entries_path="./datasets/fec_dataset/hot_entries/criteo-tb_hot_entries.npy"
  ;;
esac

echo "Using system:$sys, batch size: $batch_size num_workers: $num_workers num_hots:$num_hots num_embeddings_per_feature=$num_embeddings_per_feature"

cmd="torchrun --nnodes 1 --nproc_per_node $num_workers --rdzv_backend c10d --rdzv_endpoint localhost --rdzv_id 1234 --role trainer dlrm_main.py --pin_memory --embedding_dim 128 --system_name $sys --model $model --dataset_name $dataset --in_memory_binary_criteo_path $in_memory_binary_criteo_path --hot_entries_path $hot_entries_path --dense_arch_layer_sizes "512,256,128" --over_arch_layer_sizes "1024,1024,512,256,1" --batch_size $batch_size --n_hot_entries $num_hots --num_embeddings_per_feature $num_embeddings_per_feature"

$cmd