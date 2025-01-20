python -m attack_resilience_complex_networks.symbolic_regression \
    --sr_data_filename ./sr_data_sf_homo_policy.csv \
    --loss Sigmoid \
    --target logit \
    --nosingle_step \
    --primitives d \
    --primitives w_d \
    --binary_operators '+' \
    --binary_operators '*'
