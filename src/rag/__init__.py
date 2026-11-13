"""Retrieval-augmented context for BusEnv.

Idea: before training, pick the most useful historical facts about the road ahead of each
bus (demand at the next stops, travel times of the next segments) and save them as a small
table of numbers. During training, each bus's observation gets those numbers appended.
Nothing is retrieved while the simulation runs; it is a simple table lookup.

The work is split in two parts that run in different Python environments:

  Offline (Python 3.9, needs RL-Chunk / MA-Chunk and retrieval libraries):
      chunks.py     turn BusEnv's data files into small "chunks" of information
      scorers.py    score each chunk with text search (BM25) and embeddings (FAISS)
      labels.py     mark which chunks are worth showing to the bus (training labels)
      selectors.py  choose chunks: top-K, random, RL-Chunk, MA-Chunk
      featurize.py  turn the chosen chunks into a fixed list of 10 numbers
      table.py      save / load the final lookup table

  Online (Python 3.8, the MARLlib environment used for training):
      wrapper.py    add the numbers from the table to each bus's observation
      day_split.py  train on 2024 days, test on 2025 days

RL-Chunk and MA-Chunk live in their own git repositories. They are imported from where
they are cloned (see selectors.py); nothing from them is copied into BusEnv.

Files used by both parts (keys.py, places.py, table.py, featurize.py) must stay
compatible with Python 3.8.
"""
