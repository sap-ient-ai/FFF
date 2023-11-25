# Notation

Consider a DEPTH=4 binary tree of nodes:

```
7:3,0     8:3,1     9:3,2     a:3,3     b:3,4     c:3,5     d:3,6     e:3,7
  \         /         \         /         \         /         \         /
     3:2,0               4:2,1               5:2,2               6:2,3
       \                   /                   \                   /
              1:1,0                                    2:1,1
                \                                       /
                                  0:0,0
```

<index>: <layer>,<offset> (layer is 0 1 2 or 3)
    e.g. node[4] <-> node[2,1] <-> node[2][1]

Q: How to get index for node[layer, offset]?
    2^layer - 1 gets index for first node in that layer, so index = 2^layer - 1 + offset
    Check: (2, 1) -> 4: 2**2 - 1 + 1 = 4 ✅

Q: node[k] branches to which 2 nodes?
    Consider left branch (right branch will just be 1 higher)

    (layer, offset) has index: 2^layer - 1 + offset
    Branching up & left reaches (layer+1, 2*offset), which will have index: 
        2^(layer+1) - 1 + 2*offset
        = 2*(2^layer + offset) - 1
        = 2*(2^layer - 1 + offset  + 1) - 1
        = 2*(index + 1) - 1
        = 2*index + 1

    So `index` branches to 2*index + 1 and 2*index + 2


# Overview of algo
Each node has a .x which points somewhere in INPUT space

Think of this as the normal vector to a region-splitting hyperplane.

Consider input x

`node.score = x DOT node.x` projects our input vector `x` onto basis vector `node.x`

If `node.score` > 0, we're "sunny-side" of this hyperplane (same side as normal vector) and we branch up and left else we're "dark-side" and branch up and right.

So we're choosing our next splitting-hyperplane depending on `x`.

We'll end up with 4 winning nodes: e.g. if we fork left each time then `winners = [node[0], node[1], node[3], node[7]]`

4 layers means 4 decisions: 0000 0001 ... 1111, so 16 possible outcomes.

That's 16 possible basis-choices.

Write chosen basis as: `e0 e1 e2 e3`, where `e0 = winners[0].x`, etc

And we can write `x` as `lambda0 e0 + ... + lambda3 e3 + remainder` where `lambda_i = winners[i].score`

Note that there's some remainder -- it'll be an approximation / best-fit under chosen basis.

Now we give each `node` a `node.y` that points somewhere in OUTPUT space.

So we can consider a matching basis `f0 f1 f2 f3` in OUTPUT space.

And we TRANSFORM our input `x`:
    `y = lambda0 f0 + ... + lambda3 f3`

## Summary
    - We're dynamically choosing one of 2**DEPTH possible bases depending on `x`
    - We're finding the coeffs for `x` in terms of this basis
    - We're constructing a corresponoding `y` in terms of the matching basis over OUTPUT space.

And these input/output basis vectors will get moved around by backprop as the network trains.


# Coding refinements
    - store `node.x` and `node.y` separately; these are trainable weights
    - just store `scores` outside of the nodes; during a single pass we won't even consider most of the nodes


# Training/Inference costs

During training, if we use a small tree (e.g. depth 4, so 15 nodes) and large batch size (e.g. 64) we'd expect most nodes to get energized at least once, so we'd be backpropping thru the whole tree.

But if we have a big tree (e.g. depth 16, so 65535 nodes) and smaller batch-size (e.g. 64), most of the nodes won't even get energized and we're ONLY backpropping thru nodes that were relevant to this batch.

This could be a Big W!

During inference, we're only considering DEPTH nodes.


# NOTE on gelu

The original authors apply a `.gelu` to the `lambda`s.

I'm suspicious of this.

We've already introduced a nonlinearity by our decision-points and basis-choosing.

Adding another is just jettisoning half the info gleaned, effectively wiping out the contribution of any node that branches left.
