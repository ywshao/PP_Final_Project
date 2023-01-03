#include <stdio.h>
#include <stdlib.h>
#include <random>

// Max number - 1
#define DATA_MAX 100
#define DELTA_MAX 50

void generate(char* outfile, char* charN, char* charT, char* charPortion) {
    std::mt19937 rng(std::random_device{}());
    FILE* fout = fopen(outfile, "w");
    int N = atoi(charN);
    int T = atoi(charT);
    int portion = atoi(charPortion);
    fprintf(fout, "%d %d\n", N, T);
    for (int i = N; i; i--) {
        fprintf(fout, "%d ", rng() % DATA_MAX);
    }
    fprintf(fout, "\n");
    for (int i = T; i; i--) {
        int action = (rng() % 100 + 1 <= portion) ? 0 : 1;
        switch(action) {
        case 0:
            fprintf(fout, "%d %d %d\n", action, rng() % N, rng() % DELTA_MAX);
            break;
        case 1:
            fprintf(fout, "%d %d %d\n", action, rng() % N, rng() % N);
            break;
        }
    }
    fclose(fout);
}

int main(int argc, char* argv[]) {
    generate(argv[1], argv[2], argv[3], argv[4]);
    return 0;
}