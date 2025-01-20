python -m attack_resilience_complex_networks.explain \
  --cfg synthetic_gene_sf_n_vec \
  --global_seed 111 \
  --root_dir /data/selinda \
  --agent rl-gnn \
  --model_path synthetic_gene_sf_n_vec-agent-rl-gnn-seed-111_6/best-models/best_model.zip \
  --explanation_type model \
  --num_instances 30 \
  --nosingle_step
