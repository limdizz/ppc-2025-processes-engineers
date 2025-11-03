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
  const std::vector<int> &matrix = GetInput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_elems = static_cast<int>(matrix.size());

  // Определяем количество элементов для каждого процесса
  int elems_per_proc = total_elems / size;
  int remainder = total_elems % size;

  int start_idx = rank * elems_per_proc + std::min(rank, remainder);
  int local_count = elems_per_proc + (rank < remainder ? 1 : 0);

  std::vector<int> local_data(local_count);
  if (rank == 0) {
    // Формируем counts и displacements
    std::vector<int> counts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; i++) {
      counts[i] = elems_per_proc + (i < remainder ? 1 : 0);
      displs[i] = i * elems_per_proc + std::min(i, remainder);
    }

    // Рассылаем данные
    MPI_Scatterv(matrix.data(), counts.data(), displs.data(), MPI_INT, local_data.data(), local_count, MPI_INT, 0,
                 MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_data.data(), local_count, MPI_INT, 0, MPI_COMM_WORLD);
  }

  // Находим локальный максимум
  int local_max = std::numeric_limits<int>::min();
  for (int val : local_data) {
    local_max = std::max(local_max, val);
  }

  // Глобальное объединение максимумов
  int global_max = std::numeric_limits<int>::min();
  MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  // Передаем результат всем
  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PostProcessingImpl() {
  return GetOutput() != std::numeric_limits<int>::min();
}

}  // namespace klimenko_v_max_matrix_elems_val
