#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <cuda.h>

#define SUM true
#define MAX false
#define KERNEL 3

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
            /*
            // Debug
            if (offset == (1 << 21) && blockIdx.x == 0 && (shift == 10 || shift == 0 || shift == 0) && (indexBlock >> (blockDimLog2 - shift + 1)) + mappedIndex <= 1048576 + 4) {
                printf("%d %d %d\n%d %d %d %d\n", (indexBlock >> (blockDimLog2 - shift + 1)) + mappedIndex, (indexBlock >> (blockDimLog2 - shift + 1)) + tid, treeSum[(indexBlock >> (blockDimLog2 - shift + 1)) + tid],
                tid, tid + (1 << (shift - 1)), mappedIndex, indexMapper[blockDim.x + tid + (1 << (shift - 1))]);
            }
            */
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
    if (N > Mi) {
        dim3 grid(remainN >> 10);
        dim3 block(Ki);
        unsigned int shared_mem = 2 * Ki * sizeof(int);
        buildSumTree<<<grid, block, shared_mem>>>(treeSum, remainN);
        remainN >>= 10;
    }
    if (N > Ki) {
        dim3 grid(remainN >> 10);
        dim3 block(Ki);
        unsigned int shared_mem = 2 * Ki * sizeof(int);
        buildSumTree<<<grid, block, shared_mem>>>(treeSum, remainN);
        remainN >>= 9; // Considering index 0 and 1
    }
    dim3 grid(1);
    dim3 block(remainN);
    unsigned int shared_mem = 2 * remainN * sizeof(int);
    buildSumTree<<<grid, block, shared_mem>>>(treeSum, 0);

    int *test;
    cudaMallocHost((void**) &test, (N << 1) * sizeof(int));
    cudaMemcpy(test, treeSum, (N << 1) * sizeof(int), cudaMemcpyDeviceToHost);
    /*
    // Debug
    int acc = 0;
    for (int a = (N << 1) - 1; a; a--) {
        acc += test[a];
        if (__builtin_popcount(a) == 1) {
            printf("%d %d\n", a, acc);
            acc = 0;
        }
    }
    cudaDeviceSynchronize();
    */
}

void buildMax() {
    for (int i = 0; i < N; i++) {
        treeMax[i + N] = A[i];
    }
    for (int i = N - 1; i; i--) {
        treeMax[i] = std::max(treeMax[i << 1], treeMax[i << 1 | 1]);
    }
    treeMax[0] = 0;
}

__global__ void updateSum(int *treeSum, int index, int delta) {
    unsigned int tid = threadIdx.x;
    treeSum[index >> tid] += delta;
}

void updateMax(int index, int data) {
    for (int i = index + N; i; i >>= 1) {
        treeMax[i] = std::max(treeMax[i], data);
    }
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
        //printf("%d\n", sdata[tid]);
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
        //printf("%d %d %d\n", l, r, sdata[tid]);
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
    //printf("Test: %d %d\n", depth, 1 << log2(depth));
    querySumReduce<<<1, 2 << log2(depth), (4 << log2(depth)) * sizeof(int)>>>(treeSum, l + N, r + N);
    cudaMemcpyFromSymbol(&ansReceiver, ansSender, sizeof(int), 0, cudaMemcpyDeviceToHost);
    return ansReceiver;
}

int queryMax(int l, int r) {
    int ans = 0;
    for (l += N, r += N; l ^ r ^ 1; l >>= 1, r >>= 1) {
        if (~l & 1) {
            ans = std::max(treeMax[l ^ 1], ans);
        }
        if (r & 1) {
            ans = std::max(treeMax[r ^ 1], ans);
        }
    }
    return ans;
}

void cal(char* infile, char* outfile) {
    calIndexMapper();
    FILE* fin = fopen(infile, "r");
    FILE* fout = fopen(outfile, "w");
    fscanf(fin, "%d %d", &n, &T);
    depth = log2((n << 1) - 1) + 1;
    N = 1 << depth - 1;
    cudaMallocHost((void**) &A, N * sizeof(int), cudaHostAllocDefault);
    int i;
    for (i = 0; i < n; i++) {
        fscanf(fin, "%d", &A[i]);
    }
    for (; i < N; i++) {
        A[i] = 0;
    }
    int action, param1, param2;
#if SUM == true
    buildSum();
    for (int t = T; t; t--) {
        fscanf(fin, "%d %d %d", &action, &param1, &param2);
        switch(action) {
        case 0:
            fprintf(fout, "%d\n", querySum(param1 - 1, param2 + 1));
            break;
        case 1:
            updateSum<<<1, depth>>>(treeSum, param1 + N, param2);
            break;
        }
    }
    cudaFree(treeSum);
#endif
#if MAX == true
    cudaMalloc((void**) &treeMax, (N << 1) * sizeof(int));
    buildMax();
    for (int t = T; t; t--) {
        fscanf(fin, "%d %d %d", &action, &param1, &param2);
        switch(action) {
        case 0:
            fprintf(fout, "%d\n", queryMax(param1 - 1, param2 + 1));
            break;
        case 1:
            updateMax(param1 + N, param2);
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
    /*struct timespec start, end, temp;
    double time_used;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Do whatever
    clock_gettime(CLOCK_MONOTONIC, &end);
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    printf("%f second\n", time_used);*/

    cal(argv[1], argv[2]);
    return 0;
}