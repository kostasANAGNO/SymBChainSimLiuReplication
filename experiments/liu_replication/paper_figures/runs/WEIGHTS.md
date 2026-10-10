# Trained network weights

`*.weights` are `torch.save` files (float32), kept next to the results they belong to. Load with `torch.load(path, weights_only=True)`.

- `fig5_n100*/<scheme>_seed42.weights`: `{"online": state_dict, "episode": n}` -- final online Q-network of that Fig. 5 run
  (20000 episodes with default links, 8000 with hi links). Optimizer, target network and replay buffer are NOT included
  (the full `.ckpt` files are ~290 MB each and stay out of git).
- `sweeps_n100*/<fig>_<scheme>.weights`, `fig9_cycles/<variant>/fig9_<scheme>.weights`: a plain `state_dict` -- the network after the
  LAST value of that sweep (weights are warm-started from value to value, so earlier values are not stored).

Network class: `topk_dqn.TopKQNetNF` (node attributes: stake/50, GHz/30, x, y). Config grids: `common_n100.make_grid(8.0)` for Fig. 5, `make_grid(10.0)` for the sweeps.
