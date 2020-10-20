/* References:
 *
 *    Hong, Sungpack, et al.
 *    "Accelerating CUDA graph algorithms at maximum warp."
 *    Acm Sigplan Notices 46.8 (2011): 267-276.
 *
 *    There are so many PageRank algorithms available. We use something similar to:
 *        Galois: https://github.com/IntelligentSoftwareSystems/Galois/blob/master/lonestar/analytics/cpu/pagerank/PageRank-push.cpp
 *
 */

#include "helper_emogi.h"

#define MEM_ALIGN MEM_ALIGN_32

typedef uint32_t EdgeT;
typedef float ValueT;

__global__ void initialize(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT alpha) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count) {
        value[tid] = 1.0f - alpha;
        delta[tid] = (1.0f - alpha) * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
	}
}

__global__ void kernel_coalesce(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if(warpIdx < vertex_count && label[warpIdx]) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE)
            if (i >= start)
                atomicAdd(&residual[edgeList[i]], delta[warpIdx]);

        label[warpIdx] = false;
    }
}

__global__ void kernel_coalesce_chunk(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList) {
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
        if(label[i]) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE)
                if (j >= start)
                    atomicAdd(&residual[edgeList[j]], delta[i]);

            label[i] = false;
        }
    }
}

__global__ void update(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT tolerance, ValueT alpha, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count && residual[tid] > tolerance) {
        value[tid] += residual[tid];
        delta[tid] = residual[tid] * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
        *changed = true;
	}
}

int main(int argc, char *argv[]) {
    std::ifstream file;
    std::string vertex_file, edge_file;
    std::string filename;

    bool changed_h, *changed_d, *label_d;
    int c, arg_num = 0, device = 0;
    impl_type type;
    mem_type mem;
    ValueT *delta_d, *residual_d, *value_d, *value_h;
    ValueT tolerance, alpha;
    uint32_t iter, max_iter;
    uint64_t *vertexList_h, *vertexList_d;
    EdgeT *edgeList_h, *edgeList_d;
    uint64_t *edgeList64_h;
    uint64_t vertex_count, edge_count, vertex_size, edge_size;
    uint64_t numblocks, numblocks_update, numthreads;
    uint64_t typeT;

    float milliseconds;
    double avg_milliseconds;

    cudaEvent_t start, end;

    alpha = 0.85;
    tolerance = 0.001;
    max_iter = 5000;

    while ((c = getopt(argc, argv, "f:t:m:a:l:i:h")) != -1) {
        switch (c) {
            case 'f':
                filename = optarg;
                arg_num++;
                break;
            case 't':
                type = (impl_type)atoi(optarg);
                arg_num++;
                break;
            case 'm':
                mem = (mem_type)atoi(optarg);
                arg_num++;
                break;
            case 'd':
                device = atoi(optarg);
                break;
            case 'a':
                alpha = atof(optarg);
                break;
            case 'l':
                tolerance = atof(optarg);
                break;
            case 'i':
                max_iter = atoi(optarg);
                break;
            case 'h':
                printf("4-byte edge PageRank\n");
                printf("\t-f | input file name (must end with .bel)\n");
                printf("\t-t | type of PageRank to run\n");
                printf("\t   | COALESCE = 1, COALESCE_CHUNK = 2\n");
                printf("\t-m | memory allocation\n");
                printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
                printf("\t-d | GPU device id (default=0)\n");
                printf("\t-a | alpha (default=0.85)\n");
                printf("\t-l | tolerance (default=0.001)\n");
                printf("\t-i | max iteration (default=5000)\n");
                printf("\t-h | help message\n");
                return 0;
            case '?':
                break;
            default:
                break;
        }
    }

    if (arg_num < 3) {
        printf("4-byte edge PageRank\n");
        printf("\t-f | input file name (must end with .bel)\n");
        printf("\t-t | type of PageRank to run\n");
        printf("\t   | COALESCE = 1, COALESCE_CHUNK = 2\n");
        printf("\t-m | memory allocation\n");
        printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
        printf("\t-d | GPU device id (default=0)\n");
        printf("\t-a | alpha (default=0.85)\n");
        printf("\t-l | tolerance (default=0.001)\n");
        printf("\t-i | max iteration (default=5000)\n");
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

    edgeList64_h = (uint64_t*)malloc(edge_count * sizeof(uint64_t));
    file.read((char*)edgeList64_h, edge_count * sizeof(uint64_t));

    // Allocate memory for GPU
    checkCudaErrors(cudaMalloc((void**)&label_d, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&vertexList_d, vertex_size));
    checkCudaErrors(cudaMalloc((void**)&changed_d, sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&delta_d, vertex_count * sizeof(ValueT)));
    checkCudaErrors(cudaMalloc((void**)&residual_d, vertex_count * sizeof(ValueT)));
    checkCudaErrors(cudaMalloc((void**)&value_d, vertex_count * sizeof(ValueT)));

    value_h = (ValueT*)malloc(vertex_count * sizeof(ValueT));

    switch (mem) {
        case GPUMEM:
            checkCudaErrors(cudaMalloc((void**)&edgeList_h, edge_size));

            for (uint64_t i = 0; i < edge_count; i++)
                edgeList_h[i] = (uint32_t)edgeList64_h[i];

            break;
        case UVM_READONLY:
            checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));

            for (uint64_t i = 0; i < edge_count; i++)
                edgeList_d[i] = (uint32_t)edgeList64_h[i];

            checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetReadMostly, device));
            break;
        case UVM_DIRECT:
            checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));

            for (uint64_t i = 0; i < edge_count; i++)
                edgeList_d[i] = (uint32_t)edgeList64_h[i];

            checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetAccessedBy, device));
            break;
    }

    free(edgeList64_h);
    file.close();

    printf("Allocation finished\n");
    fflush(stdout);

    // Initialize values
    checkCudaErrors(cudaMemcpy(vertexList_d, vertexList_h, vertex_size, cudaMemcpyHostToDevice));

    if (mem == GPUMEM)
        checkCudaErrors(cudaMemcpy(edgeList_d, edgeList_h, edge_size, cudaMemcpyHostToDevice));

    numthreads = BLOCK_SIZE;

    switch (type) {
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

    numblocks_update = ((vertex_count + numthreads) / numthreads);

    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 blockDim_update(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);

    avg_milliseconds = 0.0f;

    iter = 0;

    printf("Initialization done\n");
    fflush(stdout);

    checkCudaErrors(cudaEventRecord(start, 0));

    initialize<<<blockDim_update, numthreads>>>(label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, alpha);

    // Run PageRank
    do {
        changed_h = false;
        checkCudaErrors(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

        switch (type) {
            case COALESCE:
                kernel_coalesce<<<blockDim, numthreads>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d);
                break;
            case COALESCE_CHUNK:
                kernel_coalesce_chunk<<<blockDim, numthreads>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d);
                break;
            default:
                fprintf(stderr, "Invalid type\n");
                exit(1);
                break;
        }

        update<<<blockDim_update, numthreads>>>(label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, tolerance, alpha, changed_d);

        checkCudaErrors(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));

        iter++;
    } while(changed_h && iter < max_iter);

    checkCudaErrors(cudaEventRecord(end, 0));
    checkCudaErrors(cudaEventSynchronize(end));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, end));

    printf("iteration %*u, ", 3, iter);
    printf("time %*f ms\n", 12, milliseconds);
    fflush(stdout);

    avg_milliseconds += (double)milliseconds;

    checkCudaErrors(cudaMemcpy(value_h, value_d, vertex_count * sizeof(ValueT), cudaMemcpyDeviceToHost));

    free(value_h);
    checkCudaErrors(cudaFree(label_d));
    checkCudaErrors(cudaFree(changed_d));
    checkCudaErrors(cudaFree(vertexList_d));
    checkCudaErrors(cudaFree(edgeList_d));
    checkCudaErrors(cudaFree(delta_d));
    checkCudaErrors(cudaFree(residual_d));
    checkCudaErrors(cudaFree(value_d));

    return 0;
}
