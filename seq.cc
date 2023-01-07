#include <stdio.h>
#include <time.h>
#include <algorithm>

#define SUM true
#define MAX false

int n;
int N;
int T;
int *A;
int *treeSum;
int *treeMax;

struct timespec start, end, temp;
double Input_time;
double Build_tree_time;
double Output_time;
double Query_time;
double Update_time;

inline int log2(int input) {
    return 31 ^ __builtin_clz(input);
}

void buildSum() {
    for (int i = 0; i < N; i++) {
        treeSum[i + N] = A[i];
    }
    for (int i = N - 1; i; i--) {
        treeSum[i] = treeSum[i << 1] + treeSum[i << 1 | 1];
    }
    treeSum[0] = 0;
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

void updateSum(int index, int delta) {
    for (int i = index + N; i; i >>= 1) {
        treeSum[i] += delta;
    }
}

void updateMax(int index, int data) {
    for (int i = index + N; i; i >>= 1) {
        treeMax[i] = std::max(treeMax[i], data);
    }
}

int querySum(int l, int r) {
    int ans = 0;
    for (l += N, r += N; l ^ r ^ 1; l >>= 1, r >>= 1) {
        if (~l & 1) {
            ans += treeSum[l ^ 1];
        }
        if (r & 1) {
            ans += treeSum[r ^ 1];
        }
    }
    return ans;
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
    clock_gettime(CLOCK_MONOTONIC, &start);
    FILE* fin = fopen(infile, "r");
    FILE* fout = fopen(outfile, "w");
    fscanf(fin, "%d %d", &n, &T);
    N = 1 << log2((n << 1) - 1);
    A = new int[N];
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
    clock_gettime(CLOCK_MONOTONIC, &start);
    treeSum = new int[N << 1];
    buildSum();
    clock_gettime(CLOCK_MONOTONIC, &end);
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    Build_tree_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
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
            query = querySum(param1 - 1, param2 + 1);
            clock_gettime(CLOCK_MONOTONIC, &end);
            if ((end.tv_nsec - start.tv_nsec) < 0) {
                temp.tv_sec = end.tv_sec-start.tv_sec-1;
                temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
            } else {
                temp.tv_sec = end.tv_sec - start.tv_sec;
                temp.tv_nsec = end.tv_nsec - start.tv_nsec;
            }
            Query_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
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
            clock_gettime(CLOCK_MONOTONIC, &start);
            updateSum(param1, param2);
            clock_gettime(CLOCK_MONOTONIC, &end);
            if ((end.tv_nsec - start.tv_nsec) < 0) {
                temp.tv_sec = end.tv_sec-start.tv_sec-1;
                temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
            } else {
                temp.tv_sec = end.tv_sec - start.tv_sec;
                temp.tv_nsec = end.tv_nsec - start.tv_nsec;
            }
            Update_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
            break;
        }
    }
    delete [] treeSum;
#endif
#if MAX == true
    clock_gettime(CLOCK_MONOTONIC, &start);
    treeMax = new int[N << 1];
    buildMax();
    clock_gettime(CLOCK_MONOTONIC, &end);
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    Build_tree_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
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
            query = queryMax(param1 - 1, param2 + 1);
            clock_gettime(CLOCK_MONOTONIC, &end);
            if ((end.tv_nsec - start.tv_nsec) < 0) {
                temp.tv_sec = end.tv_sec-start.tv_sec-1;
                temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
            } else {
                temp.tv_sec = end.tv_sec - start.tv_sec;
                temp.tv_nsec = end.tv_nsec - start.tv_nsec;
            }
            Query_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
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
            clock_gettime(CLOCK_MONOTONIC, &start);
            updateMax(param1, param2);
            clock_gettime(CLOCK_MONOTONIC, &end);
            if ((end.tv_nsec - start.tv_nsec) < 0) {
                temp.tv_sec = end.tv_sec-start.tv_sec-1;
                temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
            } else {
                temp.tv_sec = end.tv_sec - start.tv_sec;
                temp.tv_nsec = end.tv_nsec - start.tv_nsec;
            }
            Update_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
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
    cal(argv[1], argv[2]);
    printf("Input:\t\t%f\n", Input_time);
    printf("Build tree:\t%f\n", Build_tree_time);
    printf("Output:\t\t%f\n", Output_time);
    printf("Query:\t\t%f\n", Query_time);
    printf("Update:\t\t%f\n", Update_time);
    return 0;
}