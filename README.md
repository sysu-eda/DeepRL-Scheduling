# Deep-Reinforcement-Learning-Based Scheduler for High-Level Synthesis

This is the implementation of Deep-Reinforcement-Learning-Based Scheduler for High-Level Synthesis (HLS). This work has been published in ICCAD'19.

```
@inproceedings{DBLP:conf/iccad/ChenS19,
  author    = {Hongzheng Chen and
               Minghua Shen},
  editor    = {David Z. Pan},
  title     = {A Deep-Reinforcement-Learning-Based Scheduler for {FPGA} {HLS}},
  booktitle = {Proceedings of the International Conference on Computer-Aided Design,
               {ICCAD} 2019, Westminster, CO, USA, November 4-7, 2019},
  pages     = {1--8},
  publisher = {{ACM}},
  year      = {2019},
  url       = {https://doi.org/10.1109/ICCAD45719.2019.8942126},
  doi       = {10.1109/ICCAD45719.2019.8942126},
}
```

To run the program, please follow the instructions below.

```bash
# Generate DAGs for supervised learning
$ python3 dag_generator.py

# Supervised learning
$ python3 sl.py

# Reinforcement learning
# Use --use_network to pass in pre-trained SL networks
$ python3 rl.py
```

Prepare the test DAGs in `DAG` folder and name them as `dag_X.dot` (where `X` should be a number different from those DAGs in the training set).

```bash
# Test the Xth DAG
$ python3 rl.py --test X
```

Other parameter settings can be found in the source code.


## Requirements
* Python 3.6
* Pytorch v0.4
* Visdom v0.1
* Pulp v1.6.8
* Numpy v1.14
* Matplotlib v2.2.2