#include <stdio.h>
#include <time.h>
#include <algorithm>

#define SUM true
#define MAX false

int n;
int N;
int T;
int *A;

void updateSum(int index, int delta) {
    A[index] += delta;
}

void updateMax(int index, int data) {
    A[index] = std::max(A[index], data);
}

int querySum(int l, int r) {
    int ans = 0;
    for (; l <= r; l++) {
        ans += A[l];
    }
    return ans;
}

int queryMax(int l, int r) {
    int ans = 0;
    for (; l <= r; l++) {
        ans += std::max(A[l], ans);
    }
    return ans;
}

void cal(char* infile, char* outfile) {
    FILE* fin = fopen(infile, "r");
    FILE* fout = fopen(outfile, "w");
    fscanf(fin, "%d %d", &n, &T);
    A = new int[n];
    int i;
    for (i = 0; i < n; i++) {
        fscanf(fin, "%d", &A[i]);
    }
    int action, param1, param2;
    for (int t = T; t; t--) {
#if SUM == true
        fscanf(fin, "%d %d %d", &action, &param1, &param2);
        switch(action) {
        case 0:
            fprintf(fout, "%d ", querySum(param1, param2));
            break;
        case 1:
            updateSum(param1, param2);
            break;
        }
#endif
#if Max == true
    fscanf(fin, "%d %d %d", &action, &param1, &param2);
    switch(action) {
    case 0:
        fprintf(fout, "%d ", queryMax(param1, param2));
        break;
    case 1:
        updateMax(param1, param2);
        break;
    }
#endif
    }
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