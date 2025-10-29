#include "klimenko_v_max_matrix_elems_val/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"
#include "util/include/util.hpp"

namespace klimenko_v_max_matrix_elems_val {

KlimenkoVMaxMatrixElemsValMPI::KlimenkoVMaxMatrixElemsValMPI(const InType &in) : matrix_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KlimenkoVMaxMatrixElemsValMPI::ValidationImpl() {
  const auto &matrix = GetInput();
  return !matrix.empty() && !matrix[0].empty();
}

bool KlimenkoVMaxMatrixElemsValMPI::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::RunImpl() {
  const auto &matrix = GetInput();
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n_rows = static_cast<int>(matrix.size());

  int base_rows = n_rows / size;
  int extra_rows = n_rows % size;

  int local_rows = base_rows + (rank < extra_rows ? 1 : 0);
  int start_row = rank * base_rows + std::min(rank, extra_rows);
  int end_row = start_row + local_rows;

  int local_max = std::numeric_limits<int>::min();

  if (local_rows > 0 && start_row < n_rows) {
    for (int i = start_row; i < end_row && i < n_rows; i++) {
      if (!matrix[i].empty()) {
        int row_max = *std::max_element(matrix[i].begin(), matrix[i].end());
        local_max = std::max(local_max, row_max);
      }
    }
  }

  if (local_rows == 0) {
    local_max = std::numeric_limits<int>::min();
  }

  int global_max = std::numeric_limits<int>::min();
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PostProcessingImpl() {
  return GetOutput() != std::numeric_limits<int>::min();
}

}  // namespace klimenko_v_max_matrix_elems_val
