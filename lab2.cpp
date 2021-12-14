#include <omp.h>

#include <iostream>
using namespace std;
int main(int argc, char* argv[]) {
  int n = atoi(argv[1]);
  double* a = new double[n];
  for (long int i = 0; i < n; i++) {
    a[i] = (double)rand() / RAND_MAX;
	// printf("a[%d] = %f\n", i, a[i]);
  }
  double max_element = a[0];
  printf("initial max value is %f\n", max_element);

#pragma omp parallel for reduction(max \
                                   : max_element)
  for (int idx = 0; idx < n; idx++)
    max_element = max_element > a[idx] ? max_element : a[idx];
  delete[] a;
  printf("max value is %f\n", max_element);
}