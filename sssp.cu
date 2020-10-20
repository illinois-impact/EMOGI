/* References:
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

#define MEM_ALIGN MEM_ALIGN_64

typedef uint64_t EdgeT;
typedef uint32_t WeightT;

__global__ void kernel_coalesce(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, const WeightT *weightList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < vertex_count && label[warpIdx]) {
        uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        uint64_t end = vertexList[warpIdx+1];

        WeightT cost = newCostList[warpIdx];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (newCostList[warpIdx] != cost)
                break;
            if (newCostList[edgeList[i]] > cost + weightList[i] && i >= start)
                atomicMin(&(newCostList[edgeList[i]]), cost + weightList[i]);
        }

        label[warpIdx] = false;
    }
}

__global__ void kernel_coalesce_chunk(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, const WeightT *weightList) {
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
        if (label[i]) {
            uint64_t start = vertexList[i];
            const uint64_t shift_start = start & MEM_ALIGN;
            uint64_t end = vertexList[i+1];

            WeightT cost = newCostList[i];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (newCostList[i] != cost)
                    break;
                if (newCostList[edgeList[j]] > cost + weightList[j] && j >= start)
                    atomicMin(&(newCostList[edgeList[j]]), cost + weightList[j]);
            }

            label[i] = false;
        }
    }
}

__global__ void update(bool *label, WeightT *costList, WeightT *newCostList, const uint32_t vertex_count, bool *changed) {
	uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < vertex_count) {
        if (newCostList[tid] < costList[tid]) {
            costList[tid] = newCostList[tid];
            label[tid] = true;
            *changed = true;
        }
    }
}

int main(int argc, char *argv[]) {
    std::ifstream file, file2;
    std::string vertex_file, edge_file, weight_file;
    std::string filename;

    bool changed_h, *changed_d, no_src = false, *label_d;
    int c, num_run = 1, arg_num = 0, device = 0;
    impl_type type;
    mem_type mem;
    uint32_t one, iter;
    WeightT offset = 0;
    WeightT zero;
    WeightT *costList_d, *newCostList_d, *weightList_h, *weightList_d;
    uint64_t *vertexList_h, *vertexList_d;
    EdgeT *edgeList_h, *edgeList_d;
    uint64_t vertex_count, edge_count, weight_count, vertex_size, edge_size, weight_size;
    uint64_t typeT, src;
    uint64_t numblocks_kernel, numblocks_update, numthreads;

    float milliseconds;
    double avg_milliseconds;

    cudaEvent_t start, end;

    while ((c = getopt(argc, argv, "f:r:t:i:m:d:o:h")) != -1) {
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
            case 'o':
                offset = atoi(optarg);
                break;
            case 'h':
                printf("8-byte edge SSSP with uint32 edge weight\n");
                printf("\t-f | input file name (must end with .bel)\n");
                printf("\t-r | SSSP root (unused when i > 1)\n");
                printf("\t-t | type of SSSP to run\n");
                printf("\t   | COALESCE = 1, COALESCE_CHUNK = 2\n");
                printf("\t-m | memory allocation\n");
                printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
                printf("\t-i | number of iterations to run\n");
                printf("\t-d | GPU device id (default=0)\n");
                printf("\t-o | edge weight offset (default=0)\n");
                printf("\t-h | help message\n");
                return 0;
            case '?':
                break;
            default:
                break;
        }
    }

    if (arg_num < 4) {
        printf("8-byte edge SSSP with uint32 edge weight\n");
        printf("\t-f | input file name (must end with .bel)\n");
        printf("\t-r | SSSP root (unused when i > 1)\n");
        printf("\t-t | type of SSSP to run\n");
        printf("\t   | COALESCE = 1, COALESCE_CHUNK = 2\n");
        printf("\t-m | memory allocation\n");
        printf("\t   | GPUMEM = 0, UVM_READONLY = 1, UVM_DIRECT = 2\n");
        printf("\t-i | number of iterations to run\n");
        printf("\t-d | GPU device id (default=0)\n");
        printf("\t-o | edge weight offset (default=0)\n");
        printf("\t-h | help message\n");
        return 0;
    }

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    vertex_file = filename + ".col";
    edge_file = filename + ".dst";
    weight_file = filename + ".val";

    std::cout << filename << std::endl;

    // Read files
    // Start reading vertex list
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

    // Start reading edge list
    file.open(edge_file.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Edge file open failed\n");
        exit(1);
    }

    file.read((char*)(&edge_count), 8);
    file.read((char*)(&typeT), 8);

    printf("Edge: %lu, ", edge_count);
    fflush(stdout);
    edge_size = edge_count * sizeof(EdgeT);

    edgeList_h = NULL;

    // Start reading edge weight list
    file2.open(weight_file.c_str(), std::ios::in | std::ios::binary);
    if (!file2.is_open()) {
        fprintf(stderr, "Edge file open failed\n");
        exit(1);
    }

    file2.read((char*)(&weight_count), 8);
    file2.read((char*)(&typeT), 8);

    printf("Weight: %lu\n", weight_count);
    fflush(stdout);
    weight_size = weight_count * sizeof(WeightT);

    weightList_h = NULL;

    switch (mem) {
        case GPUMEM:
            edgeList_h = (EdgeT*)malloc(edge_size);
            weightList_h = (WeightT*)malloc(weight_size);
            file.read((char*)edgeList_h, edge_size);
            file2.read((char*)weightList_h, weight_size);
            checkCudaErrors(cudaMalloc((void**)&edgeList_d, edge_size));
            checkCudaErrors(cudaMalloc((void**)&weightList_d, weight_size));

            for (uint64_t i = 0; i < weight_count; i++)
                weightList_h[i] += offset;

            break;
        case UVM_READONLY:
            checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));
            checkCudaErrors(cudaMallocManaged((void**)&weightList_d, weight_size));
            file.read((char*)edgeList_d, edge_size);
            file2.read((char*)weightList_d, weight_size);

            for (uint64_t i = 0; i < weight_count; i++)
                weightList_d[i] += offset;

            checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetReadMostly, device));
            checkCudaErrors(cudaMemAdvise(weightList_d, weight_size, cudaMemAdviseSetReadMostly, device));
            break;
        case UVM_DIRECT:
            checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));
            checkCudaErrors(cudaMallocManaged((void**)&weightList_d, weight_size));
            file.read((char*)edgeList_d, edge_size);
            file2.read((char*)weightList_d, weight_size);

            for (uint64_t i = 0; i < weight_count; i++)
                weightList_d[i] += offset;

            checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetAccessedBy, device));
            checkCudaErrors(cudaMemAdvise(weightList_d, weight_size, cudaMemAdviseSetAccessedBy, device));
            break;
    }

    file.close();
    file2.close();

    // Allocate memory for GPU
    checkCudaErrors(cudaMalloc((void**)&vertexList_d, vertex_size));
    checkCudaErrors(cudaMalloc((void**)&label_d, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&changed_d, sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&costList_d, vertex_count * sizeof(WeightT)));
    checkCudaErrors(cudaMalloc((void**)&newCostList_d, vertex_count * sizeof(WeightT)));

    printf("Allocation finished\n");
    fflush(stdout);

    // Initialize values
    checkCudaErrors(cudaMemcpy(vertexList_d, vertexList_h, vertex_size, cudaMemcpyHostToDevice));

    if (mem == GPUMEM) {
        checkCudaErrors(cudaMemcpy(edgeList_d, edgeList_h, edge_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(weightList_d, weightList_h, weight_size, cudaMemcpyHostToDevice));
    }

    numthreads = BLOCK_SIZE;

    switch (type) {
        case COALESCE:
            numblocks_kernel = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
            break;
        case COALESCE_CHUNK:
            numblocks_kernel = ((vertex_count * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
            break;
        default:
            fprintf(stderr, "Invalid type\n");
            exit(1);
            break;
    }

    numblocks_update = ((vertex_count + numthreads) / numthreads);

    dim3 blockDim_kernel(BLOCK_SIZE, (numblocks_kernel+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 blockDim_update(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);

    avg_milliseconds = 0.0f;

    printf("Initialization done\n");
    fflush(stdout);

    // Set root
    for (int i = 0; i < num_run; i++) {
        zero = 0;
        one = 1;
        checkCudaErrors(cudaMemset(costList_d, 0xFF, vertex_count * sizeof(WeightT)));
        checkCudaErrors(cudaMemset(newCostList_d, 0xFF, vertex_count * sizeof(WeightT)));
        checkCudaErrors(cudaMemset(label_d, 0x0, vertex_count * sizeof(bool)));
        checkCudaErrors(cudaMemcpy(&label_d[src], &one, sizeof(bool), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(&costList_d[src], &zero, sizeof(WeightT), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(&newCostList_d[src], &zero, sizeof(WeightT), cudaMemcpyHostToDevice));

        iter = 0;

        checkCudaErrors(cudaEventRecord(start, 0));

        // Run SSSP
        do {
            changed_h = false;
            checkCudaErrors(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

            switch (type) {
                case COALESCE:
                    kernel_coalesce<<<blockDim_kernel, numthreads>>>(label_d, costList_d, newCostList_d, vertex_count, vertexList_d, edgeList_d, weightList_d);
                    break;
                case COALESCE_CHUNK:
                    kernel_coalesce_chunk<<<blockDim_kernel, numthreads>>>(label_d, costList_d, newCostList_d, vertex_count, vertexList_d, edgeList_d, weightList_d);
                    break;
                default:
                    fprintf(stderr, "Invalid type\n");
                    exit(1);
                    break;
            }

            update<<<blockDim_update, numthreads>>>(label_d, costList_d, newCostList_d, vertex_count, changed_d);

            iter++;

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
            WeightT *weightList_temp;

            // Flush GPU page cache for each iteration by re-allocating UVM
            switch (mem) {
                case UVM_READONLY:
                    checkCudaErrors(cudaMallocManaged((void**)&edgeList_temp, edge_size));
                    checkCudaErrors(cudaMallocManaged((void**)&weightList_temp, weight_size));
                    memcpy(edgeList_temp, edgeList_d, edge_size);
                    memcpy(weightList_temp, weightList_d, weight_size);
                    checkCudaErrors(cudaFree(edgeList_d));
                    checkCudaErrors(cudaFree(weightList_d));
                    edgeList_d = edgeList_temp;
                    weightList_d = weightList_temp;
                    checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetReadMostly, device));
                    checkCudaErrors(cudaMemAdvise(weightList_d, weight_size, cudaMemAdviseSetReadMostly, device));
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
    if (weightList_h)
        free(weightList_h);
    checkCudaErrors(cudaFree(vertexList_d));
    checkCudaErrors(cudaFree(weightList_d));
    checkCudaErrors(cudaFree(edgeList_d));
    checkCudaErrors(cudaFree(costList_d));
    checkCudaErrors(cudaFree(newCostList_d));
    checkCudaErrors(cudaFree(label_d));
    checkCudaErrors(cudaFree(changed_d));

    return 0;
}
