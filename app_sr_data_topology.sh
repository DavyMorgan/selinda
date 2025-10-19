python -m attack_resilience_complex_networks.generate_symbolic_regression_data \
  --cfg synthetic_ba_n_topology \
  --global_seed 111 \
  --root_dir /data/selinda \
  --nohas_dynamics \
  --model_path synthetic_ba_n_topology-agent-rl-gnn-seed-111_4/best-models/best_model.zip \
  --norandom_episode \
  --num_instances 30 \
  --single_step \
  --save_filename ./sr_data_topology_single_step.csv
