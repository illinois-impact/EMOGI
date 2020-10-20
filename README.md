# EMOGI

## Compilation

``Makefile`` is provided for a comprehensive compilation. However, depending on the GPU you are using, you'll need to modify the ``compute_XX`` and ``arch_XX`` arguments in ``NVCCFLAGS``.

## Dataset

We use a custom format for the datasets. Our datasets are divided into the following categories:

```
<dataset_name>.bel.dst
<dataset_name>.bel.col
<dataset_name>.bel.val
```

All input files are in binary format. The first two 8-byte values belong to a header section. The first 8-byte shows the total
number of valid elements in the file. The second 8-byte is a placeholder and it can be ignored.
The rest of the file contains the data and the datatypes for ``*.bel.dst`` and ``*.bel.col`` are always 8-byte (32-bit version programs still read in 8-byte and convert to 4-byte internally). ``*.bel.val`` uses 4-byte datatype.

We use CSR format for the input graphs. ``*.bel.col`` should contain the row offsets of CSR. ``*.bel.dst`` should contain the column indices. ``*.bel.val`` should contain the edge weight values (only required by SSSP).

## How to Run

```
$ ./bfs -h
8-byte edge BFS
        -f | input file name (must end with .bel)
        -r | BFS root (unused when i > 1)
        -t | type of BFS to run
           | BASELINE = 0, COALESCE = 1, COALESCE_CHUNK = 2
        -m | memory allocation
           | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2
        -i | number of iterations to run
        -d | GPU device id (default=0)
        -h | help message
```

Example:
```
$ ./bfs -f com-Friendster.bel -r 0 -t 2 -m 2
```

## Citation
```
@article{min2020emogi,
author = {Min, Seung Won and Mailthody, Vikram Sharma and Qureshi, Zaid and Xiong, Jinjun and Ebrahimi, Eiman and Hwu, Wen-mei},
title = {EMOGI: Efficient Memory-access for Out-of-memory Graph-traversal in GPUs},
year = {2021},
publisher = {VLDB Endowment},
volume = {14},
number = {2},
journal = {Proc. VLDB Endow.},
}
```

## Contact
Seung Won Min, min16@illinois.edu
