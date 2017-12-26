# Binary Stochastic Neurons in PyTorch

* http://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
* https://github.com/pytorch/examples/tree/master/mnist

## Results

| Model                                    | Epoch  | Accuracy |
|:----------------------------------------:|:------:|:--------:|
| BinaryNet-Deterministic-REINFORCE-False  | 41     | 0.8086   |
| BinaryNet-Deterministic-REINFORCE-True   | 33     | 0.8128   |
| BinaryNet-Deterministic-ST-False         | 94     | 0.972    |
| BinaryNet-Deterministic-ST-True          | 63     | 0.9709   |
| BinaryNet-Stochastic-REINFORCE-False     | 19     | 0.5937   |
| BinaryNet-Stochastic-REINFORCE-True      | 97     | 0.7095   |
| BinaryNet-Stochastic-ST-False            | 59     | **0.9748**   |
| BinaryNet-Stochastic-ST-True             | 89     | 0.9717   |
| NonBinaryNet-None-None-False             | 76     | 0.9734   |
| NonBinaryNet-None-None-True              | 50     | 0.9714   |
