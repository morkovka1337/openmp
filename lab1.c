#include <ctype.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
  int i = 0;
  int answer = 0;
  char cur;
#pragma omp parallel for
  for (i = 0; i < strlen(argv[1]); i++) {
#pragma omp atomic update
    answer += (isspace(argv[1][i]) > 0);
  }

  printf("Num words in %s = %d\n", argv[1], answer);
}