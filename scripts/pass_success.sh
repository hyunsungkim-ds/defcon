CUDA_VISIBLE_DEVICES=1 \
CUDA_LAUNCH_BLOCKING=1 \
python train.py \
--task pass_success \
--trial 20 \
--model gat \
--ipw_model_id pass_intent/20 \
--min_pass_dur 0.5 \
--possessor_aware \
--keeper_aware \
--ball_z_aware \
--poss_vel_aware \
--extend_features \
--sparsify none \
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
--min_lr 1e-5 \
--print_freq 50 \
--seed 100