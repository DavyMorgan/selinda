python -m attack_resilience_complex_networks.eval \
  --cfg synthetic_gene_er_n_vec \
  --global_seed 111 \
  --root_dir /data/selinda \
  --agent degree \
  --has_dynamics \
  --model_path synthetic_gene_sf_n_vec-agent-rl-gnn-seed-111_6/best-models/best_model.zip \
  --norandom_episode \
  --num_instances 30 \
  --nocase_study
