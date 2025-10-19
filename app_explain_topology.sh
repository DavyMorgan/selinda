python -m attack_resilience_complex_networks.explain \
  --cfg synthetic_ba_n_topology \
  --global_seed 111 \
  --root_dir /data/selinda \
  --nohas_dynamics \
  --model_path synthetic_ba_n_dismantle-agent-rl-gnn-seed-111_4/best-models/best_model.zip \
  --explanation_type model \
  --norandom_episode \
  --num_instances 30 \
  --nosingle_step
