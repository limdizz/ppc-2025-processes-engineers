#include "klimenko_v_seidel_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <vector>

#include "klimenko_v_seidel_method/common/include/common.hpp"

namespace klimenko_v_seidel_method {

KlimenkoVSeidelMethodMPI::KlimenkoVSeidelMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KlimenkoVSeidelMethodMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int is_valid = 0;
  if (rank == 0) {
    is_valid = ((GetInput() > 0) && (GetOutput() == 0)) ? 1 : 0;
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return is_valid != 0;
}

bool KlimenkoVSeidelMethodMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 1) {
    return false;
  }

  GetOutput() = 0;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KlimenkoVSeidelMethodMPI::RunImpl() {
  int n = GetInput();
  if (n <= 0) {
    return false;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> row_counts(size);
  std::vector<int> row_displs(size);
  std::vector<int> matrix_counts(size);
  std::vector<int> matrix_displs(size);
  computeRowDistribution(n, size, row_counts, row_displs, matrix_counts, matrix_displs);

  int local_rows = row_counts[rank];
  int start_row = row_displs[rank];

  std::vector<double> flat_matrix;
  std::vector<double> b;

  if (rank == 0) {
    initializeMatrixAndVector(flat_matrix, b, n);
  }

  std::vector<double> local_matrix((size_t)local_rows * n, 0.0);
  MPI_Scatterv(flat_matrix.data(), matrix_counts.data(), matrix_displs.data(), MPI_DOUBLE, local_matrix.data(),
               local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_b(local_rows, 0.0);
  MPI_Scatterv(b.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE, local_b.data(), local_rows, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  std::vector<double> x(n, 0.0);

  const double epsilon = 1e-6;
  const int max_iterations = 1000;

  for (int iteration = 0; iteration < max_iterations; iteration++) {
    std::vector<double> x_old = x;

    for (int i = 0; i < local_rows; i++) {
      int global_i = start_row + i;

      double sum_off_diag = 0.0;
      for (int j = 0; j < n; j++) {
        if (j != global_i) {
          sum_off_diag += local_matrix[(size_t)i * n + j] * x[j];
        }
      }

      x[global_i] = (local_b[i] - sum_off_diag) / local_matrix[(size_t)i * n + global_i];
    }

    MPI_Allgatherv(x.data() + start_row, local_rows, MPI_DOUBLE, x.data(), row_counts.data(), row_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    double local_diff = 0.0;
    for (int i = 0; i < local_rows; i++) {
      int gi = start_row + i;
      double d = x[gi] - x_old[gi];
      local_diff += d * d;
    }

    double global_diff = 0.0;
    MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    global_diff = std::sqrt(global_diff);

    if (global_diff < epsilon) {
      break;
    }
  }

  if (rank == 0) {
    GetOutput() = computeFinalResult(x, n);
  }

  MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

bool KlimenkoVSeidelMethodMPI::PostProcessingImpl() {
  return GetOutput() > 0;
}

void KlimenkoVSeidelMethodMPI::computeRowDistribution(int n, int size, std::vector<int> &row_counts,
                                                      std::vector<int> &row_displs, std::vector<int> &matrix_counts,
                                                      std::vector<int> &matrix_displs) {
  int row_offset = 0;
  int matrix_offset = 0;
  for (int proc = 0; proc < size; proc++) {
    int base_rows = n / size;
    int extra = (proc < (n % size)) ? 1 : 0;
    int proc_rows = base_rows + extra;

    row_counts[proc] = proc_rows;
    row_displs[proc] = row_offset;
    matrix_counts[proc] = proc_rows * n;
    matrix_displs[proc] = matrix_offset;

    row_offset += proc_rows;
    matrix_offset += proc_rows * n;
  }
}

int KlimenkoVSeidelMethodMPI::computeFinalResult(const std::vector<double> &x, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return static_cast<int>(std::round(sum));
}

void KlimenkoVSeidelMethodMPI::initializeMatrixAndVector(std::vector<double> &flat_matrix, std::vector<double> &b,
                                                         int n) {
  flat_matrix.resize(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; i++) {
    flat_matrix[(static_cast<std::size_t>(i) * n) + i] = 1.0;
  }
  b.resize(n, 1.0);
}

}  // namespace klimenko_v_seidel_method
