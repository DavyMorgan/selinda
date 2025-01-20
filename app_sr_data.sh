python -m attack_resilience_complex_networks.generate_symbolic_regression_data \
  --cfg synthetic_gene_sf_n_vec \
  --global_seed 111 \
  --root_dir /data/selinda \
  --model_path synthetic_gene_sf_n_vec-agent-rl-gnn-seed-111_6/best-models/best_model.zip \
  --num_instances 30 \
  --nosingle_step \
  --save_filename ./sr_data_sf.csv
