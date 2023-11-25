# Summary of FFF
FFF shows promise to exponentially reduce the compute-power required by a feed-forward neural network layer, while retaining most of the neural-computer power.


# Goal
The purpose of this repo is to play with FastFeedForward Networks.

We plan to engineer a performant implementation.

Also as the innovation is new, we want to explore & tweak, rather than simply dive 100% into optimizations.

Chat with us on [Discord](https://discord.gg/Utv7tcGz)

NOTE: We're not affiliated with the authors of the FFF papers, but we'd be thrilled if they were to pop in and say hi!


# Repo contents
- We have provided a re-conceptualization of this innovation in [doc/theory.md](doc/theory.md). At the heart is the idea of dynamically choosing (per sample input) a most-appropriate-basis-pair (basis in INPUT space and basis in OUTPUT space), approximating our input x as a linear combination of X-basis vectors, and projecting into OUTPUT-space by applying these coefficients to our OUTPUT-space vectors. The basis is found by traversing a binary tree, where each node contains a X,Y pair of basis vectors.

- We've rewritten the core code [fff.py](fff.py) and it boils down to half a dozen lines of PyTorch/einsum. There's a `for` loop (for traversing the binary-tree) so naive solution is extremely non-optimized. We've treaked the weight-initialization and shot for a simpler target than do the original papers.

- We benchmark FFF against standard PyTorch FF (standard FeedForward layer). The [first benchmark](notebooks/FFF_layer_benchmark.ipynb) shows that for small layers FF wins, but as we increase the layer-size FFF starts to outperform FF. e.g. setting nIn = nOut = 2^14, FFF is already performing at 20x speed.

- Next we check that a FFF layer is actually learning. We create a simple CIFAR10 classifier NN ([here](notebooks/FFF_CIFAR10_benchmark.ipynb)) and replace the FF layers with FFF. We find that after 5 epochs FF has achieved ~52% accuracy whereas FFF has achieved ~48%. So FFF trains.


# TODO
- Tweaking, tinkering, benchmarking, analyzing learning, exploring
- Creating CPU and GPU/CUDA optimized implementations
- Exploring how we can use this innovation in other architectures (CNN, Attention, etc.) and whether it leads to novel architectures.


# Papers
- (18 Sep 2023)[Fast Feedforward Networks](https://arxiv.org/pdf/2308.14711.pdf) ([code](https://github.com/pbelcak/fastfeedforward))

- (15 Nov 2023)[Exponentially Faster Language Modelling](https://arxiv.org/abs/2311.10770) ([code](https://github.com/pbelcak/FastBERT/benchmark_pytorch/fff/fff_bmm.py))

Second revision of paper has updated repo [here](https://github.com/pbelcak/UltraFastBERT) containing CUDA code.


# Misc

2023.11.23
- π created https://github.com/pbelcak/UltraFastBERT/issues/1  
Observing the BERT benchmark performs slower than the vanilla BERT on HF  
    
- π created https://github.com/pbelcak/UltraFastBERT/issues/2  
An interpretation of the core algorithm, and a suggestion for improvement (remove the .gelu)  
Links to a gist demo of FFF operating over MNIST.
