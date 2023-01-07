#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <omp.h>

#define SUM true
#define MAX false

int n;
int N;
int T;
int *A;
int *treeSum;
int *treeMax;
int depth;

inline int log2(int input) {
    return 31 ^ __builtin_clz(input);
}

void buildSum() {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        treeSum[i + N] = A[i];
    }
    for (int NN = N; NN > 1; NN >>= 1) {
        #pragma omp parallel for
        for (int i = NN >> 1; i < NN; i++) {
            treeSum[i] = treeSum[i << 1] + treeSum[i << 1 | 1];
        }
    }
    treeSum[0] = 0;
}

void buildMax() {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        treeMax[i + N] = A[i];
    }
    for (int NN = N; NN > 1; NN >>= 1) {
        #pragma omp parallel for
        for (int i = NN >> 1; i < NN; i++) {
            treeMax[i] = std::max(treeMax[i << 1], treeMax[i << 1 | 1]);
        }
    }
    treeMax[0] = 0;
}

void updateSum(int index, int delta) {
    int leafIndex = index + N;
    int updateIndex[depth];
    #pragma omp parallel for
    for (int i = 0; i < depth; i++) {
        updateIndex[i] = leafIndex >> i;
    }
    #pragma omp parallel for
    for (int i = 0; i < depth; i++) {
        treeSum[updateIndex[i]] += delta;
    }
}

void updateMax(int index, int data) {
    int leafIndex = index + N;
    int updateIndex[depth];
    #pragma omp parallel for
    for (int i = 0; i < depth; i++) {
        updateIndex[i] = leafIndex >> i;
    }
    #pragma omp parallel for
    for (int i = 0; i < depth; i++) {
        treeMax[updateIndex[i]] = std::max(treeMax[updateIndex[i]], data);
    }
}

int querySum(int l, int r) {
    int ans = 0;
    l += N;
    r += N;
    int updateIndexL[depth];
    int updateIndexR[depth];
    #pragma omp parallel for
    for (int i = 0; i < depth; i++) {
        int currentIndexL = l >> i;
        int currentIndexR = r >> i;
        if (currentIndexL ^ currentIndexR ^ 1 && currentIndexL != currentIndexR) {
            updateIndexL[i] = currentIndexL;
            updateIndexR[i] = currentIndexR;
        }
        else {
            updateIndexL[i] = 1;
            updateIndexR[i] = 0;
        }
    }
    #pragma omp parallel for reduction(+:ans)
    for (int i = 0; i < depth; i++) {
        if (~updateIndexL[i] & 1) {
            ans += treeSum[updateIndexL[i] ^ 1];
        }
        if (updateIndexR[i] & 1) {
            ans += treeSum[updateIndexR[i] ^ 1];
        }
    }
    return ans;
}

int queryMax(int l, int r) {
    int ans = 0;
    l += N;
    r += N;
    int updateIndexL[depth];
    int updateIndexR[depth];
    #pragma omp parallel for
    for (int i = 0; i < depth; i++) {
        int currentIndexL = l >> i;
        int currentIndexR = r >> i;
        if (currentIndexL ^ currentIndexR ^ 1 && currentIndexL != currentIndexR) {
            updateIndexL[i] = currentIndexL;
            updateIndexR[i] = currentIndexR;
        }
        else {
            updateIndexL[i] = 1;
            updateIndexR[i] = 0;
        }
    }
    #pragma omp parallel for reduction(max:ans)
    for (int i = 0; i < depth; i++) {
        if (~updateIndexL[i] & 1) {
            ans = treeMax[updateIndexL[i] ^ 1] > ans ? treeMax[updateIndexL[i] ^ 1] : ans;
        }
        if (updateIndexR[i] & 1) {
            ans = treeMax[updateIndexR[i] ^ 1] > ans ? treeMax[updateIndexR[i] ^ 1] : ans;
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
    A = new int[N];
    int i;
    for (i = 0; i < n; i++) {
        fscanf(fin, "%d", &A[i]);
    }
    for (; i < N; i++) {
        A[i] = 0;
    }
    int action, param1, param2;
#if SUM == true
    treeSum = new int[N << 1];
    buildSum();
    for (int t = T; t; t--) {
        fscanf(fin, "%d %d %d", &action, &param1, &param2);
        switch(action) {
        case 0:
            fprintf(fout, "%d\n", querySum(param1 - 1, param2 + 1));
            break;
        case 1:
            updateSum(param1, param2);
            break;
        }
    }
    delete [] treeSum;
#endif
#if MAX == true
    treeMax = new int[N << 1];
    buildMax();
    for (int t = T; t; t--) {
        fscanf(fin, "%d %d %d", &action, &param1, &param2);
        switch(action) {
        case 0:
            fprintf(fout, "%d\n", queryMax(param1 - 1, param2 + 1));
            break;
        case 1:
            updateMax(param1, param2);
            break;
        }
    }
    delete [] treeMax;
#endif
    delete [] A;
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