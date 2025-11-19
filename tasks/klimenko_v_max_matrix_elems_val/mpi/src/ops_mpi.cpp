#include "klimenko_v_max_matrix_elems_val/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <climits>
#include <cstddef>

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"

namespace klimenko_v_max_matrix_elems_val {

KlimenkoVMaxMatrixElemsValMPI::KlimenkoVMaxMatrixElemsValMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0;
}

bool KlimenkoVMaxMatrixElemsValMPI::ValidationImpl() {
  return GetOutput() == 0;
}

bool KlimenkoVMaxMatrixElemsValMPI::PreProcessingImpl() {
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::RunImpl() {
  const auto &matrix = GetInput();
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = matrix.size();
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (n == 0) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }
  int local_size = n / size;
  int remainder = n % size;
  int start_idx;
  int end_idx;
  if (rank < remainder) {
    start_idx = rank * (local_size + 1);
    end_idx = start_idx + local_size + 1;
  } else {
    start_idx = remainder * (local_size + 1) + (rank - remainder) * local_size;
    end_idx = start_idx + local_size;
  }
  int local_max = INT_MIN;
  for (int i = start_idx; i < end_idx && i < n; i++) {
    for (size_t j = 0; j < matrix[i].size(); j++) {
      local_max = std::max(matrix[i][j], local_max);
    }
  }
  if (local_size == 0 && rank >= n) {
    local_max = INT_MIN;
  }
  int global_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  GetOutput() = global_max;
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PostProcessingImpl() {
  return true;
}

}  // namespace klimenko_v_max_matrix_elems_val
