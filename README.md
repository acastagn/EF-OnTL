# EF-OnTL

This package is a library for EF-OnTL (Expert-Free Online Transfer Learning), developed as a supplement to the thesis submitted at Trinity College Dublin, The University of Dublin, in fulfillment of the PhD degree.

Please be aware that if you are creating your own agent, it is essential to define the following methods in order to ensure compatibility with the rest of the framework:

- `get_loss(state, action, reward, state1, done)`
- `learn_from_samples(batch)`

These methods are fundamental for proper interaction with the rest of the EF-OnTL library.

## Citation

If you find this library useful for your work, please consider citing:

@article{castagna2023expert,
  title={Expert-Free Online Transfer Learning in Multi-Agent Reinforcement Learning},<br />
  author={Castagna, Alberto and Dusparic, Ivana},<br />
  journal={arXiv preprint arXiv:2303.01170},<br />
  year={2023}<br />
}
