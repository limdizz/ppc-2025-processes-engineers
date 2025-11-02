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
  if (matrix.empty() || matrix[0].empty()) {
    return false;
  }
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::RunImpl() {
  const std::vector<std::vector<int>> &inputMatrix = GetInput();

  int pid, pCount;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &pCount);

  std::vector<int> flatMatrix;
  int totalElems = 0;
  if (pid == 0) {
    for (const auto &row : inputMatrix) {
      flatMatrix.insert(flatMatrix.end(), row.begin(), row.end());
    }
    totalElems = static_cast<int>(flatMatrix.size());
  }

  MPI_Bcast(&totalElems, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sizes(pCount);
  std::vector<int> offsets(pCount);
  if (pid == 0) {
    int baseSize = totalElems / pCount;
    int remainder = totalElems % pCount;
    int step = 0;
    for (int i = 0; i < pCount; i++) {
      sizes[i] = baseSize + (i < remainder ? 1 : 0);
      offsets[i] = step;
      step += sizes[i];
    }
  }

  MPI_Bcast(sizes.data(), pCount, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(offsets.data(), pCount, MPI_INT, 0, MPI_COMM_WORLD);

  int localSize = sizes[pid];
  std::vector<int> localData(localSize);

  MPI_Scatterv(flatMatrix.data(), sizes.data(), offsets.data(), MPI_INT, localData.data(), localSize, MPI_INT, 0,
               MPI_COMM_WORLD);

  int localMax =
      localData.empty() ? std::numeric_limits<int>::min() : *std::max_element(localData.begin(), localData.end());

  int globalMax = 0;
  MPI_Reduce(&localMax, &globalMax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Bcast(&globalMax, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = globalMax;

  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PostProcessingImpl() {
  return GetOutput() != std::numeric_limits<int>::min();
}

}  // namespace klimenko_v_max_matrix_elems_val
