NVCC=nvcc
NVCCFLAGS=-gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70

all: bfs bfs_32 cc cc_32 sssp sssp_float sssp_32 pagerank pagerank_32

bfs: bfs.cu
	${NVCC} -O3 ${NVCCFLAGS} -o $@ $^

bfs_32: bfs_32.cu
	${NVCC} -O3 ${NVCCFLAGS} -o $@ $^

cc: cc.cu
	${NVCC} -O3 ${NVCCFLAGS} -o $@ $^

cc_32: cc_32.cu
	${NVCC} -O3 ${NVCCFLAGS} -o $@ $^

sssp: sssp.cu
	${NVCC} -O3 ${NVCCFLAGS} -o $@ $^

sssp_float: sssp_float.cu
	${NVCC} -O3 ${NVCCFLAGS} -o $@ $^

sssp_32: sssp_32.cu
	${NVCC} -O3 ${NVCCFLAGS} -o $@ $^

pagerank: pagerank.cu
	${NVCC} -O3 ${NVCCFLAGS} -o $@ $^

pagerank_32: pagerank_32.cu
	${NVCC} -O3 ${NVCCFLAGS} -o $@ $^

clean:
	rm -f bfs bfs_32 cc cc_32 sssp sssp_float sssp_32 pagerank pagerank_32
