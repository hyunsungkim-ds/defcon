CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
python train.py \
--task shot_blocking \
--trial 1 \
--model gat \
--possessor_aware \
--keeper_aware \
--ball_z_aware \
--poss_vel_aware \
--filter_blockers \
--sparsify delaunay \
--node_emb_dim 128 \
--graph_emb_dim 128 \
--mlp_h1_dim 64 \
--mlp_h2_dim 16 \
--gnn_layers 2 \
--gnn_heads 4 \
--skip_conn \
--n_epochs 100 \
--batch_size 512 \
--lambda_l1 0.0001 \
--start_lr 0.002 \
--min_lr 1e-6 \
--print_freq 50 \
--seed 100