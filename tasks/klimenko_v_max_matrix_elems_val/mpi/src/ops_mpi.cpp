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
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Только процесс 0 проверяет входные данные
  if (rank == 0) {
    const auto &matrix = GetInput();
    return !matrix.empty() && !matrix[0].empty();
  }

  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::RunImpl() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int local_max = std::numeric_limits<int>::min();
  if (rank == 0) {
    const auto &matrix = GetInput();
    for (const auto &row : matrix) {
      if (!row.empty()) {
        int row_max = *std::max_element(row.begin(), row.end());
        local_max = std::max(local_max, row_max);
      }
    }
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
