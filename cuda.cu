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

inline int log2(int input) {
    return 31 ^ __builtin_clz(input);
}

// Interleaved addressing
#if KERNEL == 1
__global__ void buildSumTree(int *treeSum, int N) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int index = N + blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = treeSum[index];
    __syncthreads();
    for (int shift = 0; shift <= (31 ^ __clz(blockDim.x)); shift++) {
        if (!(tid % (2 << shift))) {
            sdata[tid] += sdata[tid + (1 << shift)];
            treeSum[index >> (shift + 1)] = sdata[tid];
        }
        __syncthreads();
    }
    if(!tid) {
        treeSum[0] = 0;
    }
}
#endif

void buildSum() {
    cudaMalloc((void**) &treeSum, (N << 1) * sizeof(int));
    cudaMemcpy(treeSum + N, A, N * sizeof(int), cudaMemcpyHostToDevice);
    int blockSize = N > 1024 ? 1024 : N;
    dim3 grid(N / blockSize);
    dim3 block(blockSize);
    unsigned int shared_mem = 2 * blockSize * sizeof(int);
    buildSumTree<<<grid, block, shared_mem>>>(treeSum, N);
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
    }
    __syncthreads();
    for (unsigned shift = 0; shift < (31 ^ __clz(blockDim.x)); shift++) {
        if (!(tid % (1 << shift))) {
            sdata[tid] += sdata[tid + (1 << shift)];
        }
        __syncthreads();
    }
    if(!tid) {
        ansSender = sdata[0];
    }
}
#endif

int querySum(int l, int r) {
    querySumReduce<<<1, depth, 2 * depth * sizeof(int)>>>(treeSum, l + N, r + N);
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
            fprintf(fout, "%d ", querySum(param1 - 1, param2 + 1));
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
            fprintf(fout, "%d ", queryMax(param1 - 1, param2 + 1));
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