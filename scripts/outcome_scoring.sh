CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
python train.py \
--task outcome_scoring \
--trial 20 \
--model gat \
--keeper_aware \
--use_xg \
--return_type disc_0.9 \
--sparsify none \
--edge_in_dim 2 \
--node_emb_dim 128 \
--graph_emb_dim 128 \
--mlp_h1_dim 64 \
--mlp_h2_dim 16 \
--gnn_layers 2 \
--gnn_heads 4 \
--skip_conn \
--n_epochs 100 \
--batch_size 512 \
--lambda_l1 1e-6 \
--start_lr 0.0002 \
--min_lr 1e-5 \
--print_freq 50 \
--seed 100