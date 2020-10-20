/* References:
 *
 *    Harish, Pawan, and P. J. Narayanan.
 *    "Accelerating large graph algorithms on the GPU using CUDA."
 *    International conference on high-performance computing.
 *    Springer, Berlin, Heidelberg, 2007.
 *
 *    Hong, Sungpack, et al.
 *    "Accelerating CUDA graph algorithms at maximum warp."
 *    Acm Sigplan Notices 46.8 (2011): 267-276.
 *
 *    Lifeng Nai, Yinglong Xia, Ilie G. Tanase, Hyesoon Kim, and Ching-Yung Lin.
 *    GraphBIG: Understanding Graph Computing in the Context of Industrial Solutions,
 *    In the proccedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC),
 *    Nov. 2015
 *
 */

#include "helper_emogi.h"

#define MYINFINITY 0xFFFFFFFF
#define MEM_ALIGN MEM_ALIGN_64

typedef uint64_t EdgeT;

__global__ void kernel_baseline(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < vertex_count && label[tid] == level) {
        const uint64_t start = vertexList[tid];
        const uint64_t end = vertexList[tid+1];

        for(uint64_t i = start; i < end; i++) {
            const EdgeT next = edgeList[i];

            if(label[next] == MYINFINITY) {
                label[next] = level + 1;
                *changed = true;
            }
        }
    }
}

__global__ void kernel_coalesce(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if(warpIdx < vertex_count && label[warpIdx] == level) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                const EdgeT next = edgeList[i];

                if(label[next] == MYINFINITY) {
                    label[next] = level + 1;
                    *changed = true;
                }
            }
        }
    }
}

__global__ void kernel_coalesce_chunk(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;

    if((chunkIdx + CHUNK_SIZE) > vertex_count) {
        if ( vertex_count > chunkIdx )
            chunk_size = vertex_count - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        if(label[i] == level) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (j >= start) {
                    const EdgeT next = edgeList[j];

                    if(label[next] == MYINFINITY) {
                        label[next] = level + 1;
                        *changed = true;
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    std::ifstream file;
    std::string vertex_file, edge_file;
    std::string filename;

    bool changed_h, *changed_d, no_src = false;
    int c, num_run = 1, arg_num = 0, device = 0;
    impl_type type;
    mem_type mem;
    uint32_t *label_d, level, zero, iter;
    uint64_t *vertexList_h, *vertexList_d;
    EdgeT *edgeList_h, *edgeList_d;
    uint64_t vertex_count, edge_count, vertex_size, edge_size;
    uint64_t typeT, src;
    uint64_t numblocks, numthreads;

    float milliseconds;
    double avg_milliseconds;

    cudaEvent_t start, end;

    while ((c = getopt(argc, argv, "f:r:t:i:m:d:h")) != -1) {
        switch (c) {
            case 'f':
                filename = optarg;
                arg_num++;
                break;
            case 'r':
                if (!no_src)
                    src = atoll(optarg);
                arg_num++;
                break;
            case 't':
                type = (impl_type)atoi(optarg);
                arg_num++;
                break;
            case 'i':
                no_src = true;
                src = 0;
                num_run = atoi(optarg);
                arg_num++;
                break;
            case 'm':
                mem = (mem_type)atoi(optarg);
                arg_num++;
                break;
            case 'd':
                device = atoi(optarg);
                break;
            case 'h':
                printf("8-byte edge BFS\n");
                printf("\t-f | input file name (must end with .bel)\n");
                printf("\t-r | BFS root (unused when i > 1)\n");
                printf("\t-t | type of BFS to run\n");
                printf("\t   | BASELINE = 0, COALESCE = 1, COALESCE_CHUNK = 2\n");
                printf("\t-m | memory allocation\n");
                printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
                printf("\t-i | number of iterations to run\n");
                printf("\t-d | GPU device id (default=0)\n");
                printf("\t-h | help message\n");
                return 0;
            case '?':
                break;
            default:
                break;
        }
    }

    if (arg_num < 4) {
        printf("8-byte edge BFS\n");
        printf("\t-f | input file name (must end with .bel)\n");
        printf("\t-r | BFS root (unused when i > 1)\n");
        printf("\t-t | type of BFS to run\n");
        printf("\t   | BASELINE = 0, COALESCE = 1, COALESCE_CHUNK = 2\n");
        printf("\t-m | memory allocation\n");
        printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
        printf("\t-i | number of iterations to run\n");
        printf("\t-d | GPU device id (default=0)\n");
        printf("\t-h | help message\n");
        return 0;
    }

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    vertex_file = filename + ".col";
    edge_file = filename + ".dst";

    std::cout << filename << std::endl;

    // Read files
    file.open(vertex_file.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Vertex file open failed\n");
        exit(1);
    }

    file.read((char*)(&vertex_count), 8);
    file.read((char*)(&typeT), 8);

    vertex_count--;

    printf("Vertex: %lu, ", vertex_count);
    vertex_size = (vertex_count+1) * sizeof(uint64_t);

    vertexList_h = (uint64_t*)malloc(vertex_size);

    file.read((char*)vertexList_h, vertex_size);
    file.close();

    file.open(edge_file.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Edge file open failed\n");
        exit(1);
    }

    file.read((char*)(&edge_count), 8);
    file.read((char*)(&typeT), 8);

    printf("Edge: %lu\n", edge_count);
    fflush(stdout);
    edge_size = edge_count * sizeof(EdgeT);

    edgeList_h = NULL;

    // Allocate memory for GPU
    checkCudaErrors(cudaMalloc((void**)&vertexList_d, vertex_size));
    checkCudaErrors(cudaMalloc((void**)&label_d, vertex_count * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc((void**)&changed_d, sizeof(bool)));

    switch (mem) {
        case GPUMEM:
            edgeList_h = (EdgeT*)malloc(edge_size);
            file.read((char*)edgeList_h, edge_size);
            checkCudaErrors(cudaMalloc((void**)&edgeList_d, edge_size));

            break;
        case UVM_READONLY:
            checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));
            file.read((char*)edgeList_d, edge_size);

            checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetReadMostly, device));
            break;
        case UVM_DIRECT:
            checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));
            file.read((char*)edgeList_d, edge_size);

            checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetAccessedBy, device));
            break;
    }

    file.close();

    printf("Allocation finished\n");
    fflush(stdout);

    // Initialize values
    checkCudaErrors(cudaMemcpy(vertexList_d, vertexList_h, vertex_size, cudaMemcpyHostToDevice));

    if (mem == GPUMEM)
        checkCudaErrors(cudaMemcpy(edgeList_d, edgeList_h, edge_size, cudaMemcpyHostToDevice));

    numthreads = BLOCK_SIZE;

    switch (type) {
        case BASELINE:
            numblocks = ((vertex_count + numthreads) / numthreads);
            break;
        case COALESCE:
            numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
            break;
        case COALESCE_CHUNK:
            numblocks = ((vertex_count * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            break;
        default:
            fprintf(stderr, "Invalid type\n");
            exit(1);
            break;
    }

    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);

    avg_milliseconds = 0.0f;

    printf("Initialization done\n");
    fflush(stdout);

    // Set root
    for (int i = 0; i < num_run; i++) {
        zero = 0;
        checkCudaErrors(cudaMemset(label_d, 0xFF, vertex_count * sizeof(uint32_t)));
        checkCudaErrors(cudaMemcpy(&label_d[src], &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));

        level = 0;
        iter = 0;

        checkCudaErrors(cudaEventRecord(start, 0));

        // Run BFS
        do {
            changed_h = false;
            checkCudaErrors(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

            switch (type) {
                case BASELINE:
                    kernel_baseline<<<blockDim, numthreads>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d);
                    break;
                case COALESCE:
                    kernel_coalesce<<<blockDim, numthreads>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d);
                    break;
                case COALESCE_CHUNK:
                    kernel_coalesce_chunk<<<blockDim, numthreads>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d);
                    break;
                default:
                    fprintf(stderr, "Invalid type\n");
                    exit(1);
                    break;
            }

            iter++;
            level++;

            checkCudaErrors(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));
        } while(changed_h);

        checkCudaErrors(cudaEventRecord(end, 0));
        checkCudaErrors(cudaEventSynchronize(end));
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, end));

        printf("run %*d: ", 3, i);
        printf("src %*lu, ", 12, src);
        printf("iteration %*u, ", 3, iter);
        printf("time %*f ms\n", 12, milliseconds);
        fflush(stdout);

        avg_milliseconds += (double)milliseconds;

        src += vertex_count / num_run;

        if (i < num_run - 1) {
            EdgeT *edgeList_temp;

            // Flush GPU page cache for each iteration by re-allocating UVM
            switch (mem) {
                case UVM_READONLY:
                    checkCudaErrors(cudaMallocManaged((void**)&edgeList_temp, edge_size));
                    memcpy(edgeList_temp, edgeList_d, edge_size);
                    checkCudaErrors(cudaFree(edgeList_d));
                    edgeList_d = edgeList_temp;
                    checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetReadMostly, device));
                    break;
                default:
                    break;
            }
        }
    }

    printf("Average run time %f ms\n", avg_milliseconds / num_run);

    free(vertexList_h);
    if (edgeList_h)
        free(edgeList_h);
    checkCudaErrors(cudaFree(vertexList_d));
    checkCudaErrors(cudaFree(edgeList_d));
    checkCudaErrors(cudaFree(label_d));
    checkCudaErrors(cudaFree(changed_d));

    return 0;
}
