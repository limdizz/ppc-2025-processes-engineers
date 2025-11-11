#include "klimenko_v_max_matrix_elems_val/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"
#include "util/include/util.hpp"

namespace klimenko_v_max_matrix_elems_val {

KlimenkoVMaxMatrixElemsValMPI::KlimenkoVMaxMatrixElemsValMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0;
}

bool KlimenkoVMaxMatrixElemsValMPI::ValidationImpl() {
  return ((!GetInput().empty()) && (GetOutput() == 0));
}

bool KlimenkoVMaxMatrixElemsValMPI::PreProcessingImpl() {
  return !GetInput().empty();
}

bool KlimenkoVMaxMatrixElemsValMPI::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &matrix = GetInput();
  auto total_rows = matrix.size();

  if (total_rows == 0) {
    return false;
  }

  auto rows_per_process = total_rows / size;
  auto remainder = total_rows % size;

  size_t start_row = ((rank * rows_per_process) + (std::min(static_cast<size_t>(rank), remainder)));
  size_t end_row = start_row + rows_per_process;
  if (std::cmp_less(static_cast<size_t>(rank), remainder)) {
    end_row += 1;
  }

  int local_max = INT_MIN;
  for (size_t i = start_row; i < end_row; ++i) {
    for (int element : matrix[i]) {
      local_max = std::max(element, local_max);
    }
  }

  int global_max = 0;

  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;

  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PostProcessingImpl() {
  return !GetInput().empty();
}

}  // namespace klimenko_v_max_matrix_elems_val
