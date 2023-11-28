# Summary of FFF
FFF shows promise to exponentially reduce the compute-power required by a feed-forward neural network layer, while retaining most of the neural-computer power.


# Goal
The purpose of this repo is to play with FastFeedForward Networks.

We plan to engineer a performant implementation.

Also as the innovation is new, we want to explore & tweak, rather than simply dive 100% into optimizations.

Chat with us on [Discord](https://discord.gg/Utv7tcGz)

NOTE: We're not affiliated with the authors of the FFF papers, but we'd be thrilled if they were to pop in and say hi!


# Repo contents
```
doc/
    Let's try to keep organized
    
    theory.md
        We have provided a re-conceptualization of this innovation. At the heart is the idea of dynamically choosing (per sample input) a most-appropriate-basis-pair (basis in INPUT space and basis in OUTPUT space), approximating our input x as a linear combination of X-basis vectors, and projecting into OUTPUT-space by applying these coefficients to our OUTPUT-space vectors. The basis-pair is found by traversing a binary tree, where each node contains a X,Y pair of basis vectors.
        TODO: tidy this up (π) -- images, clearer explanation, LaTeX.

`FFF/`
    "Production" code will go here.

    🔸fff.py
        We've rewritten the core code and it boils down to half a dozen lines of PyTorch/einsum. There's a `for` loop (for traversing the binary-tree) so this naive solution is extremely non-optimized. We've tweaked the weight-initialization and thrown out a .gelu that was in the original code.
        TODO: hold the input-vector and output-vector contiguously in memory to reduce indexing/lookup costs

    fff_jax.py
        It runs and produces output.
        TODO: profile & wrap. We want all "backends" to have the same interface if possible.
        So we can `fff_layer = FFF(nIn=1024, nOut=512, backend="jax")`

    fff_cuda.py
        Bojan is working on this.
        Currently forward pass is working. TODO: backward pass, profiling, optim, etc.

notebooks/
    Benchmarks are here:

    FFF_layer_benchmark.ipynb
        Simplest benchmark, shows that as we increase layer-size even the naive FFF fast outperforms FF

    FFF_CIFAR10_benchmark.ipynb
        There's experiments/fff_mnist.ipynb (TODO: move it over here)
        MNIST is rather trite tho' and any old NN can get 98% these days, so CIFAR10 is more challenging task that'll better show the neural-power of a NN.

    FFF_CUDA_benchmark.ipynb
        TODO: update this (CUDA impl currently WiP)

experiments/
    This is where we put up experiments.
    If we get something juicy we'll move it into the appropriate place in the repo.

    fff_mnist.ipynb
        Early experiments. TODO: tidy this up
    
    hf_bert.ipynb
        Authors of second paper published a FFF-BERT model on HF.
        We evaluate its performance compared against a standard BERT model.
        It isn't any faster on a M2 mac. Actually it's slower.
    
    🔹pi_slice.ipynb
        Some experiments riffing on the "basis transform theme"
            - what if we throw out the concept of a 'tree' and simply have latent-nodes
            - what if we compare our input against latent-nodes and pick top-k winners?
            - what if we ReLU on our lambdas?
            - what if we introduce an orthogonality costfunc?
```

## Benchmarks
- We benchmark FFF against standard PyTorch FF (standard FeedForward layer). The [first benchmark](notebooks/FFF_layer_benchmark.ipynb) shows that for small layers FF wins, but as we increase the layer-size FFF starts to outperform FF. e.g. setting nIn = nOut = 2^14, FFF is already performing at 20x speed.

- Next we check that a FFF layer is actually learning. We create a simple CIFAR10 classifier NN ([here](notebooks/FFF_CIFAR10_benchmark.ipynb)) and replace the FF layers with FFF. We find that after 5 epochs FF has achieved ~52% accuracy whereas FFF has achieved ~48%. So FFF trains.


# TODO
- Tweaking, tinkering, benchmarking, analyzing learning, exploring
- Creating CPU and GPU/CUDA optimized implementations
- PyPI package
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
