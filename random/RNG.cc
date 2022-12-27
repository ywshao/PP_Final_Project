#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));
    int N;
    scanf("%d", &N);
    FILE *fout = fopen("out", "w");
    for (int a = N; a; a--) {
        fprintf(fout, "%d ", rand() % 100);
    }
    fclose(fout);
    return 0;
}