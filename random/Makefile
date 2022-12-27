CXXFLAGS = -O3
EXES     := RNG

alls: $(EXES)

clean:
	rm -f $(EXES)

RNG: RNG.cc
	g++ $(CXXFLAGS) -o $@ $?
