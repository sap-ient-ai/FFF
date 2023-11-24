The purpose of this repo is to play with FastFeedForward Networks.

Summary:
    FFF shows promise to exponentially reduce the compute-power required by a feed-forward neural network layer

First paper:
    Fast Feedforward Networks
    18 Sep 2023
    https://arxiv.org/pdf/2308.14711.pdf
        Example at https://github.com/pbelcak/fastfeedforward/blob/main/notebook/example.ipynb

Second paper:
    Exponentially Faster Language Modelling
    15 Nov 2023
    https://arxiv.org/abs/2311.10770
        https://github.com/pbelcak/FastBERT
            FastBERT/benchmark_pytorch/fff/fff_bmm.py <-- I've annotated this

    Second revision of paper has updated repo:
        https://github.com/pbelcak/UltraFastBERT
            This version contains CUDA code.

2023.11.23
    π created https://github.com/pbelcak/UltraFastBERT/issues/1
        Observing the BERT benchmark performs slower than the vanilla BERT on HF
    
    π created https://github.com/pbelcak/UltraFastBERT/issues/2
        An interpretation of the core algorithm, and a suggestion for improvement (remove the .gelu)
            Links to a gist demo of FFF operating over MNIST.
