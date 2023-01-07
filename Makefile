OPTFLAGS = -O3
NVFLAGS  := -std=c++11 $(OPTFLAGS) -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm
CXXFLAGS = $(OPTFLAGS) -pthread
EXES     := naive seq omp cuda naive_timing seq_timing cuda_timing omp_timing

alls: $(EXES)

clean:
	rm -f $(EXES)

naive: naive.cc
	g++ $(CXXFLAGS) -o $@ $?

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

omp: omp.cc
	g++ $(CXXFLAGS) -fopenmp -o $@ $?

cuda: cuda.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

naive_timing: naive_timing.cc
	g++ $(CXXFLAGS) -o $@ $?

seq_timing: seq_timing.cc
	g++ $(CXXFLAGS) -o $@ $?

omp_timing: omp_timing.cc
	g++ $(CXXFLAGS) -fopenmp -o $@ $?

cuda_timing: cuda_timing.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?
