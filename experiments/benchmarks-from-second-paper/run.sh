#!/bin/bash

echo "🔸 Batch size 100"
echo "naive FF (batch matmult)"
python main.py  --model ff_bmm  --input-width 8000  --hidden-width 4000  --output-width 8000  --depth 8  --batch-size 100  --n-iters 10  --device cpu

echo "FFF (batch matmult)"
python main.py  --model fff_bmm  --input-width 8000  --hidden-width 4000  --output-width 8000  --depth 8  --batch-size 100  --n-iters 10  --device cpu


echo "🔸 Batch size 10"
echo "naive FF (batch matmult)"
python main.py  --model ff_bmm  --input-width 8000  --hidden-width 4000  --output-width 8000  --depth 8  --batch-size 10  --n-iters 10  --device cpu

echo "FFF (batch matmult)"
python main.py  --model fff_bmm  --input-width 8000  --hidden-width 4000  --output-width 8000  --depth 8  --batch-size 10  --n-iters 10  --device cpu


echo "🔸 Batch size 1"

echo "naive FF (batch matmult)"
python main.py  --model ff_bmm  --input-width 8000  --hidden-width 4000  --output-width 8000  --depth 8  --batch-size 1  --n-iters 10  --device cpu

echo "FFF (batch matmult)"
python main.py  --model fff_bmm  --input-width 8000  --hidden-width 4000  --output-width 8000  --depth 8  --batch-size 1  --n-iters 10  --device cpu
