NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm
#CXXFLAGS = -O3 -pthread
EXES     := naive seq omp cuda

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
