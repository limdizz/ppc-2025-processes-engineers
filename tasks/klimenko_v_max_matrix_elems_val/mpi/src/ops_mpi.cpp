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
  GetOutput() = std::numeric_limits<int>::min();
  ;
}

bool KlimenkoVMaxMatrixElemsValMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool KlimenkoVMaxMatrixElemsValMPI::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_elems = 0;
  if (rank == 0) {
    total_elems = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&total_elems, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (total_elems <= 0) {
    GetOutput() = std::numeric_limits<int>::min();
    return false;
  }

  std::vector<int> full_matrix;
  if (rank == 0) {
    full_matrix = GetInput();
  } else {
    full_matrix.resize(total_elems);
  }

  MPI_Bcast(full_matrix.data(), total_elems, MPI_INT, 0, MPI_COMM_WORLD);

  int elems_per_proc = total_elems / size;
  int remainder = total_elems % size;

  int start_idx = rank * elems_per_proc + std::min(rank, remainder);
  int local_count = elems_per_proc + (rank < remainder ? 1 : 0);
  int end_idx = start_idx + local_count;

  int local_max = std::numeric_limits<int>::min();
  for (int idx = start_idx; idx < end_idx; ++idx) {
    local_max = std::max(local_max, full_matrix[idx]);
  }

  int global_max = std::numeric_limits<int>::min();
  MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PostProcessingImpl() {
  return GetOutput() != std::numeric_limits<int>::min();
}

}  // namespace klimenko_v_max_matrix_elems_val
