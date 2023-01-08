#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <cuda.h>

#define SUM true
#define MAX false
#define KERNEL 1

int n;
int N;
int T;
int *A;
int *treeSum;
int *treeMax;
int depth;

int ansReceiver;
__device__ int ansSender;
__constant__ int indexMapper[2048];

struct timespec start, end, temp;
double Input_time;
double Output_time;
double Memcpy_time;

inline int log2(int input) {
    return 31 ^ __builtin_clz(input);
}

void calIndexMapper() {
    int indexMapperTemp[2048];
    indexMapperTemp[0] = 0;
    indexMapperTemp[1] = 1;
    for (int i = 1; i < 1024; i++) {
        indexMapperTemp[i + (1 << log2(i))] = 2 * indexMapperTemp[i];
        indexMapperTemp[i + (2 << log2(i))] = 2 * indexMapperTemp[i] + 1;
    }
    for (int i = 1; i < 2048; i++) {
        indexMapperTemp[i] -= 1 << log2(indexMapperTemp[i]);
    }
    indexMapperTemp[0] = 0;
    cudaMemcpyToSymbol(indexMapper, &indexMapperTemp, 2048 * sizeof(float));
}

// Interleaved addressing less divergent
#if KERNEL == 1
__global__ void buildSumTree(int *treeSum, int offset) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int index = offset + blockIdx.x * blockDim.x + threadIdx.x;
    int blockDimLog2 = 31 ^ __clz(blockDim.x);
    sdata[tid] = treeSum[index];
    __syncthreads();
    for (int shift = 0; shift < blockDimLog2; shift++) {
        if (!(tid % (2 << shift))) {
            sdata[tid] += sdata[tid + (1 << shift)];
            treeSum[index >> (shift + 1)] = sdata[tid];
        }
        __syncthreads();
    }
    if (!tid) {
        treeSum[0] = 0;
    }
}
// Interleaved addressing
#elif KERNEL == 2
__global__ void buildSumTree(int *treeSum, int offset) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int indexBlock = offset + blockIdx.x * blockDim.x;
    unsigned int index = indexBlock + threadIdx.x;
    int blockDimLog2 = 31 ^ __clz(blockDim.x);
    sdata[tid] = treeSum[index];
    __syncthreads();
    for (int shift = 0; shift < blockDimLog2; shift++) {
        int newIndex = (2 << shift) * tid;
        if (newIndex < blockDim.x) {
            sdata[newIndex] += sdata[newIndex + (1 << shift)];
            treeSum[(indexBlock >> (shift + 1)) + tid] = sdata[newIndex];
        }
        __syncthreads();
    }
    if (!tid) {
        treeSum[0] = 0;
    }
}
// Sequential addressing
#elif KERNEL == 3
__global__ void buildSumTree(int *treeSum, int offset) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int mappedIndex = indexMapper[blockDim.x + tid];
    unsigned int index = offset + blockIdx.x * blockDim.x + mappedIndex;
    unsigned int indexBlock = offset + blockIdx.x * blockDim.x;
    int blockDimLog2 = 31 ^ __clz(blockDim.x);
    sdata[tid] = treeSum[index];
    __syncthreads();
    for (int shift = blockDimLog2; shift; shift--) {
        if (tid < (1 << (shift - 1))) {
            sdata[tid] += sdata[tid + (1 << (shift - 1))];
            treeSum[(indexBlock >> (blockDimLog2 - shift + 1)) + indexMapper[(1 << (shift - 1)) + tid]] = sdata[tid];
        }
        __syncthreads();
    }
    if (!tid) {
        treeSum[0] = 0;
    }
}
// Unroll the last warp, but the warp don't like global store
#elif KERNEL == 4
__global__ void buildSumTree(int *treeSum, int offset) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int mappedIndex = indexMapper[blockDim.x + tid];
    unsigned int index = offset + blockIdx.x * blockDim.x + mappedIndex;
    unsigned int indexBlock = offset + blockIdx.x * blockDim.x;
    int blockDimLog2 = 31 ^ __clz(blockDim.x);
    sdata[tid] = treeSum[index];
    __syncthreads();
    for (int shift = blockDimLog2; shift; shift--) {
        if (tid < (1 << (shift - 1))) {
            sdata[tid] += sdata[tid + (1 << (shift - 1))];
            treeSum[(indexBlock >> (blockDimLog2 - shift + 1)) + indexMapper[(1 << (shift - 1)) + tid]] = sdata[tid];
        }
        __syncthreads();
    }
    if (tid < 32) {
        switch(blockDim.x) {
        case 1024:
        case 512:
        case 256:
        case 128:
        case 64:
        case 32:
            /*sdata[tid] += sdata[tid + 32];
            treeSum[(indexBlock >> (blockDimLog2 - 5)) + indexMapper[32 + tid]] = sdata[tid];
            __syncthreads();*/
        case 16:
            /*sdata[tid] += sdata[tid + 16];
            treeSum[(indexBlock >> (blockDimLog2 - 4)) + indexMapper[16 + tid]] = sdata[tid];
            __syncthreads();*/
        case 8:
            /*sdata[tid] += sdata[tid + 8];
            treeSum[(indexBlock >> (blockDimLog2 - 3)) + indexMapper[8 + tid]] = sdata[tid];
            __syncthreads();*/
        case 4:
            /*sdata[tid] += sdata[tid + 4];
            treeSum[(indexBlock >> (blockDimLog2 - 2)) + indexMapper[4 + tid]] = sdata[tid];
            __syncthreads();*/
        case 2:
            /*sdata[tid] += sdata[tid + 2];
            treeSum[(indexBlock >> (blockDimLog2 - 1)) + indexMapper[2 + tid]] = sdata[tid];
            __syncthreads();*/
        case 1:
            /*sdata[tid] += sdata[tid + 1];
            treeSum[(indexBlock >> (blockDimLog2)) + indexMapper[1 + tid]] = sdata[tid];
            __syncthreads();*/
        }
    }
    if (!tid) {
        treeSum[0] = 0;
    }
}
#endif


void buildSum() {
    cudaMalloc((void**) &treeSum, (N << 1) * sizeof(int));
    cudaMemcpy(treeSum + N, A, N * sizeof(int), cudaMemcpyHostToDevice);
    const int Mi = (1 << 20);
    const int Ki = (1 << 10);
    int remainN = N;
    bool itsOver1024 = false;
    if (N > Mi) {
        dim3 grid(remainN >> 10);
        dim3 block(Ki);
        unsigned int shared_mem = 2 * Ki * sizeof(int);
        buildSumTree<<<grid, block, shared_mem>>>(treeSum, remainN);
        remainN >>= 10;
        itsOver1024 = true;
    }
    if (N > Ki) {
        dim3 grid(remainN >> 10);
        dim3 block(Ki);
        unsigned int shared_mem = 2 * Ki * sizeof(int);
        buildSumTree<<<grid, block, shared_mem>>>(treeSum, remainN);
        remainN >>= 9; // Considering index 0 and 1
        itsOver1024 = true;
    }
    dim3 grid(1);
    dim3 block(itsOver1024 ? remainN : remainN << 1);
    unsigned int shared_mem = 2 * remainN * sizeof(int);
    buildSumTree<<<grid, block, shared_mem>>>(treeSum, 0);
}


// Interleaved addressing less divergent
#if KERNEL == 1
__global__ void buildMaxTree(int *treeMax, int offset) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int index = offset + blockIdx.x * blockDim.x + threadIdx.x;
    int blockDimLog2 = 31 ^ __clz(blockDim.x);
    sdata[tid] = treeMax[index];
    __syncthreads();
    for (int shift = 0; shift < blockDimLog2; shift++) {
        if (!(tid % (2 << shift))) {
            sdata[tid] = max(sdata[tid], sdata[tid + (1 << shift)]);
            treeMax[index >> (shift + 1)] = sdata[tid];
        }
        __syncthreads();
    }
    if (!tid) {
        treeMax[0] = 0;
    }
}
// Interleaved addressing
#elif KERNEL == 2
__global__ void buildMaxTree(int *treeMax, int offset) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int indexBlock = offset + blockIdx.x * blockDim.x;
    unsigned int index = indexBlock + threadIdx.x;
    int blockDimLog2 = 31 ^ __clz(blockDim.x);
    sdata[tid] = treeMax[index];
    __syncthreads();
    for (int shift = 0; shift < blockDimLog2; shift++) {
        int newIndex = (2 << shift) * tid;
        if (newIndex < blockDim.x) {
            sdata[newIndex] = max(sdata[newIndex], sdata[newIndex + (1 << shift)]);
            treeMax[(indexBlock >> (shift + 1)) + tid] = sdata[newIndex];
        }
        __syncthreads();
    }
    if (!tid) {
        treeMax[0] = 0;
    }
}
// Sequential addressing
#elif KERNEL == 3
__global__ void buildMaxTree(int *treeMax, int offset) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int mappedIndex = indexMapper[blockDim.x + tid];
    unsigned int index = offset + blockIdx.x * blockDim.x + mappedIndex;
    unsigned int indexBlock = offset + blockIdx.x * blockDim.x;
    int blockDimLog2 = 31 ^ __clz(blockDim.x);
    sdata[tid] = treeMax[index];
    __syncthreads();
    for (int shift = blockDimLog2; shift; shift--) {
        if (tid < (1 << (shift - 1))) {
            sdata[tid] = max(sdata[tid], sdata[tid + (1 << (shift - 1))]);
            treeMax[(indexBlock >> (blockDimLog2 - shift + 1)) + indexMapper[(1 << (shift - 1)) + tid]] = sdata[tid];
        }
        __syncthreads();
    }
    if (!tid) {
        treeMax[0] = 0;
    }
}
#endif

void buildMax() {
    cudaMalloc((void**) &treeMax, (N << 1) * sizeof(int));
    cudaMemcpy(treeMax + N, A, N * sizeof(int), cudaMemcpyHostToDevice);
    const int Mi = (1 << 20);
    const int Ki = (1 << 10);
    int remainN = N;
    bool itsOver1024 = false;
    if (N > Mi) {
        dim3 grid(remainN >> 10);
        dim3 block(Ki);
        unsigned int shared_mem = 2 * Ki * sizeof(int);
        buildMaxTree<<<grid, block, shared_mem>>>(treeMax, remainN);
        remainN >>= 10;
        itsOver1024 = true;
    }
    if (N > Ki) {
        dim3 grid(remainN >> 10);
        dim3 block(Ki);
        unsigned int shared_mem = 2 * Ki * sizeof(int);
        buildMaxTree<<<grid, block, shared_mem>>>(treeMax, remainN);
        remainN >>= 9; // Considering index 0 and 1
        itsOver1024 = true;
    }
    dim3 grid(1);
    dim3 block(itsOver1024 ? remainN : remainN << 1);
    unsigned int shared_mem = 2 * remainN * sizeof(int);
    buildMaxTree<<<grid, block, shared_mem>>>(treeMax, 0);
}

__global__ void updateSum(int *treeSum, int index, int delta) {
    unsigned int tid = threadIdx.x;
    treeSum[index >> tid] += delta;
}

__global__ void updateMax(int *treeMax, int index, int data) {
    unsigned int tid = threadIdx.x;
    treeMax[index >> tid] = max(treeMax[index >> tid], data);
}

// Interleaved addressing
#if KERNEL == 1
__global__ void querySumReduce(int *treeSum, int l, int r) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    l >>= tid;
    r >>= tid;
    sdata[tid] = 0;
    if (l ^ r ^ 1 && l != r) {
        if (~l & 1) {
            sdata[tid] += treeSum[l ^ 1];
        }
        if (r & 1) {
            sdata[tid] += treeSum[r ^ 1];
        }
    }
    __syncthreads();
    for (unsigned int shift = 1; shift < blockDim.x; shift <<= 1) {
        if (!(tid % (shift << 1)) && tid + shift < blockDim.x) {
            sdata[tid] += sdata[tid + shift];
        }
        __syncthreads();
    }
    if(!tid) {
        ansSender = sdata[0];
    }
}
// Interleaved addressing less divergent
#elif KERNEL == 2
__global__ void querySumReduce(int *treeSum, int l, int r) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    l >>= tid;
    r >>= tid;
    sdata[tid] = 0;
    if (l ^ r ^ 1 && l != r) {
        if (~l & 1) {
            sdata[tid] += treeSum[l ^ 1];
        }
        if (r & 1) {
            sdata[tid] += treeSum[r ^ 1];
        }
    }
    __syncthreads();
    for (unsigned int shift = 1; shift < blockDim.x; shift <<= 1) {
        int newIndex = 2 * shift * tid;
        if (newIndex + shift < blockDim.x) {
            sdata[newIndex] += sdata[newIndex + shift];
        }
        __syncthreads();
    }
    if(!tid) {
        ansSender = sdata[0];
    }
}
// Sequential addressing
#elif KERNEL == 3
__global__ void querySumReduce(int *treeSum, int l, int r) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    l >>= tid;
    r >>= tid;
    sdata[tid] = 0;
    if (l ^ r ^ 1 && l != r) {
        if (~l & 1) {
            sdata[tid] += treeSum[l ^ 1];
        }
        if (r & 1) {
            sdata[tid] += treeSum[r ^ 1];
        }
    }
    __syncthreads();
    for (unsigned int shift = blockDim.x >> 1; shift; shift >>= 1) {
        if (tid < shift) {
            sdata[tid] += sdata[tid + shift];
        }
        __syncthreads();
    }
    if(!tid) {
        ansSender = sdata[0];
    }
}
#endif

int querySum(int l, int r) {
    querySumReduce<<<1, 2 << log2(depth), (4 << log2(depth)) * sizeof(int)>>>(treeSum, l + N, r + N);
    cudaMemcpyFromSymbol(&ansReceiver, ansSender, sizeof(int), 0, cudaMemcpyDeviceToHost);
    return ansReceiver;
}

// Interleaved addressing
#if KERNEL == 1
__global__ void queryMaxReduce(int *treeMax, int l, int r) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    l >>= tid;
    r >>= tid;
    sdata[tid] = 0;
    if (l ^ r ^ 1 && l != r) {
        if (~l & 1) {
            sdata[tid] = max(sdata[tid], treeMax[l ^ 1]);
        }
        if (r & 1) {
            sdata[tid] = max(sdata[tid], treeMax[r ^ 1]);
        }
    }
    __syncthreads();
    for (unsigned int shift = 1; shift < blockDim.x; shift <<= 1) {
        if (!(tid % (shift << 1)) && tid + shift < blockDim.x) {
            sdata[tid] = max(sdata[tid], sdata[tid + shift]);
        }
        __syncthreads();
    }
    if(!tid) {
        ansSender = sdata[0];
    }
}
// Interleaved addressing less divergent
#elif KERNEL == 2
__global__ void queryMaxReduce(int *treeMax, int l, int r) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    l >>= tid;
    r >>= tid;
    sdata[tid] = 0;
    if (l ^ r ^ 1 && l != r) {
        if (~l & 1) {
            sdata[tid] = max(sdata[tid], treeMax[l ^ 1]);
        }
        if (r & 1) {
            sdata[tid] = max(sdata[tid], treeMax[r ^ 1]);
        }
    }
    __syncthreads();
    for (unsigned int shift = 1; shift < blockDim.x; shift <<= 1) {
        int newIndex = 2 * shift * tid;
        if (newIndex + shift < blockDim.x) {
            sdata[newIndex] = max(sdata[newIndex], sdata[newIndex + shift]);
        }
        __syncthreads();
    }
    if(!tid) {
        ansSender = sdata[0];
    }
}
// Sequential addressing
#elif KERNEL == 3
__global__ void queryMaxReduce(int *treeMax, int l, int r) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    l >>= tid;
    r >>= tid;
    sdata[tid] = 0;
    if (l ^ r ^ 1 && l != r) {
        if (~l & 1) {
            sdata[tid] = max(sdata[tid], treeMax[l ^ 1]);
        }
        if (r & 1) {
            sdata[tid] = max(sdata[tid], treeMax[r ^ 1]);
        }
    }
    __syncthreads();
    for (unsigned int shift = blockDim.x >> 1; shift; shift >>= 1) {
        if (tid < shift) {
            sdata[tid] = max(sdata[tid], sdata[tid + shift]);
        }
        __syncthreads();
    }
    if(!tid) {
        ansSender = sdata[0];
    }
}
#endif

int queryMax(int l, int r) {
    queryMaxReduce<<<1, 2 << log2(depth), (2 << log2(depth)) * sizeof(int)>>>(treeMax, l + N, r + N);
    cudaMemcpyFromSymbol(&ansReceiver, ansSender, sizeof(int), 0, cudaMemcpyDeviceToHost);
    return ansReceiver;
}

void cal(char* infile, char* outfile) {
#if KERNEL == 3
    calIndexMapper();
#endif
    clock_gettime(CLOCK_MONOTONIC, &start);
    FILE* fin = fopen(infile, "r");
    FILE* fout = fopen(outfile, "w");
    fscanf(fin, "%d %d", &n, &T);
    clock_gettime(CLOCK_MONOTONIC, &end);
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    Input_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    depth = log2((n << 1) - 1) + 1;
    N = 1 << depth - 1;
    cudaMallocHost((void**) &A, N * sizeof(int), cudaHostAllocDefault);
    clock_gettime(CLOCK_MONOTONIC, &start);
    int i;
    for (i = 0; i < n; i++) {
        fscanf(fin, "%d", &A[i]);
    }
    for (; i < N; i++) {
        A[i] = 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    Input_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    int action, param1, param2, query;
#if SUM == true
    buildSum();
    for (int t = T; t; t--) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        fscanf(fin, "%d %d %d", &action, &param1, &param2);
        clock_gettime(CLOCK_MONOTONIC, &end);
        if ((end.tv_nsec - start.tv_nsec) < 0) {
            temp.tv_sec = end.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
        } else {
            temp.tv_sec = end.tv_sec - start.tv_sec;
            temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        }
        Input_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
        switch(action) {
        case 0:
            query = querySum(param1 - 1, param2 + 1);
            clock_gettime(CLOCK_MONOTONIC, &start);
            fprintf(fout, "%d\n", query);
            clock_gettime(CLOCK_MONOTONIC, &end);
            if ((end.tv_nsec - start.tv_nsec) < 0) {
                temp.tv_sec = end.tv_sec-start.tv_sec-1;
                temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
            } else {
                temp.tv_sec = end.tv_sec - start.tv_sec;
                temp.tv_nsec = end.tv_nsec - start.tv_nsec;
            }
            Output_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
            break;
        case 1:
            updateSum<<<1, depth>>>(treeSum, param1 + N, param2);
            break;
        }
    }
    cudaFree(treeSum);
#endif
#if MAX == true
    buildMax();
    for (int t = T; t; t--) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        fscanf(fin, "%d %d %d", &action, &param1, &param2);
        clock_gettime(CLOCK_MONOTONIC, &end);
        if ((end.tv_nsec - start.tv_nsec) < 0) {
            temp.tv_sec = end.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
        } else {
            temp.tv_sec = end.tv_sec - start.tv_sec;
            temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        }
        Input_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
        switch(action) {
        case 0:
            clock_gettime(CLOCK_MONOTONIC, &start);
            fprintf(fout, "%d\n", queryMax(param1 - 1, param2 + 1));
            clock_gettime(CLOCK_MONOTONIC, &end);
            if ((end.tv_nsec - start.tv_nsec) < 0) {
                temp.tv_sec = end.tv_sec-start.tv_sec-1;
                temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
            } else {
                temp.tv_sec = end.tv_sec - start.tv_sec;
                temp.tv_nsec = end.tv_nsec - start.tv_nsec;
            }
            Output_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
            break;
        case 1:
            updateMax<<<1, depth>>>(treeMax, param1 + N, param2);
            break;
        }
    }
    cudaFree(treeMax);
#endif
    cudaFreeHost(A);
    fclose(fin);
    fclose(fout);
}

int main(int argc, char* argv[]) {
    cal(argv[1], argv[2]);
    printf("Input:\t\t%f\n", Input_time);
    printf("Output:\t\t%f\n", Output_time);
    return 0;
}