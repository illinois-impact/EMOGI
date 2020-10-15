#ifndef HELPER_EMOGI_H_
#define HELPER_EMOGI_H_

#include <cuda.h>
#include <fstream>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <getopt.h>
#include "helper_cuda.h"

#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE 32

#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define MEM_ALIGN_64 (~(0xfULL))
#define MEM_ALIGN_32 (~(0x1fULL))

typedef enum {
    BASELINE = 0,
    COALESCE = 1,
    COALESCE_CHUNK = 2,
} impl_type;

typedef enum {
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
} mem_type;

#endif