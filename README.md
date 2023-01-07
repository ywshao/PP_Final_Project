# PP Final Project
## How to Make Binaries
```
make
```
## How to Run the Program
```
./Binary Infile Outfile
```
Binary: naive, omp, and seq\
Infile: test case file name\
Outfile: result file name
## Report
### OpenMP
-c1, -c2, -c3, etc.
### CUDA
3 versions of optimized kernels
### Test case size
10, 100, 1000, ... , 10000000, etc.
### Optimization Flags
-O0, -O1, -O2, and -O3
### Time Profiling
IO (read N, T, and data) time\
Preprocessing (build tree) time\
Calculation time (update, and query)\
Calculation IO time (fscanf and fprintf)\
see naive.cc int main commented code\
nvprof for CUDA version
### Differences Between Each Version
O(TN)\
naive: can not handle large T * N\
O(N + TlogN)\
seq\
omp\
cuda
### Report
