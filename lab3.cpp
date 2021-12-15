#include <omp.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>

using namespace std;

// The function of allocating memory for a two-dimensional array
double** malloc_array(long int n) {
  double** matrix = new double*[n];
  for (int i = 0; i < n; i++)
    matrix[i] = new double[n];
  return matrix;
}

// Memory free function
void free_array(double** matrix, long int n) {
  for (int i = 0; i < n; i++)
    delete[] matrix[i];
  delete[] matrix;
}

// The function of initializing a matrix with random numbers from [0,1]
void rand_init_matrix(double** matrix, long int n) {
  srand(time(NULL));
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      matrix[i][j] = rand() / RAND_MAX;
}

// Matrix zeroing function
void zero_init_matrix(double** matrix, long int n) {
  srand(time(NULL));
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      matrix[i][j] = 0.0;
}

double measure_time(double** A, double** B, double** C, long int N, string order) {
  clock_t t;
  if (order == "ijk") {
    t = clock();
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
          C[i][j] += A[i][k] * B[k][j];
    t = clock() - t;
  } else if (order == "jki") {
    t = clock();
    for (int j = 0; j < N; j++)
      for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++)
          C[i][j] += A[i][k] * B[k][j];
    t = clock() - t;
  } else if (order == "ikj") {
    t = clock();
    for (int i = 0; i < N; i++)
      for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
          C[i][j] += A[i][k] * B[k][j];
    t = clock() - t;
  }
  return t / CLOCKS_PER_SEC;
}
double parallel_measure_time(double** A, double** B, double** C, long int N, string order, int num_thr) {
  double dtime;
  if (order == "ijk") {
    dtime = omp_get_wtime();
#pragma omp parallel num_threads(num_thr)
    {
#pragma omp for
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
          for (int k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];
    }
    dtime = omp_get_wtime() - dtime;
    return dtime;
  }
  if (order == "jki") {
    dtime = omp_get_wtime();
#pragma omp parallel num_threads(num_thr)
    {
#pragma omp for
      for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
          for (int i = 0; i < N; i++)
            C[i][j] += A[i][k] * B[k][j];
    }
    dtime = omp_get_wtime() - dtime;
    return dtime;
  }
  if (order == "ikj") {
    dtime = omp_get_wtime();
#pragma omp parallel num_threads(num_thr)
    {
#pragma omp for
      for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
          for (int j = 0; j < N; j++)
            C[i][j] += A[i][k] * B[k][j];
    }
    dtime = omp_get_wtime() - dtime;
    return dtime;
  }
}

int main(int argc, char* argv[]) {
  int N = atoi(argv[1]);
  double **A, **B, **C;

  // Memory allocation for matrices A, B, C
  A = malloc_array(N);
  B = malloc_array(N);
  C = malloc_array(N);

  // Initialization of matrices
  rand_init_matrix(A, N);
  rand_init_matrix(B, N);
  zero_init_matrix(C, N);

  cout << "Measure ijk loops" << endl;
  auto sequnce_time = measure_time(A, B, C, N, "ijk");
  cout << "Sequence time " << sequnce_time << endl;
  for (int num_thr = 1; num_thr <= 10; num_thr++) {
    zero_init_matrix(C, N);
    auto parallel_time = parallel_measure_time(A, B, C, N, "ijk", num_thr);
    cout << "Parallel time " << parallel_time << endl;
    cout << "Efficiency (" << num_thr << " threads) = " << sequnce_time / parallel_time << endl;
  }

  zero_init_matrix(C, N);

  cout << "Measure jki loops" << endl;
  sequnce_time = measure_time(A, B, C, N, "jki");
  cout << "Sequence time " << sequnce_time << endl;
  for (int num_thr = 1; num_thr <= 10; num_thr++) {
    zero_init_matrix(C, N);
    auto parallel_time = parallel_measure_time(A, B, C, N, "jki", num_thr);
    cout << "Parallel time " << parallel_time << endl;
    cout << "Efficiency (" << num_thr << " threads) = " << sequnce_time / parallel_time << endl;
  }

  zero_init_matrix(C, N);

  cout << "Measure ikj loops" << endl;
  sequnce_time = measure_time(A, B, C, N, "ikj");
  cout << "Sequence time " << sequnce_time << endl;
  for (int num_thr = 1; num_thr <= 10; num_thr++) {
    zero_init_matrix(C, N);
    auto parallel_time = parallel_measure_time(A, B, C, N, "ikj", num_thr);
    cout << "Parallel time " << parallel_time << endl;
    cout << "Efficiency (" << num_thr << " threads) = " << sequnce_time / parallel_time << endl;
  }

  // Freeing memory occupied by matrices A, B, C
  free_array(A, N);
  free_array(B, N);
  free_array(C, N);

  return 0;
}
