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
}

bool KlimenkoVMaxMatrixElemsValMPI::ValidationImpl() {
  int pid = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  if (pid == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::RunImpl() {
  const std::vector<int> &inputVec = GetInput();

  int pid = 0, pCount = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &pCount);

  int elemsCount = static_cast<int>(inputVec.size());

  std::vector<int> sizes(pCount);
  std::vector<int> offsets(pCount);

  if (pid == 0) {
    int baseSize = elemsCount / pCount;
    int remainder = elemsCount % pCount;
    int step = 0;
    for (int i = 0; i < pCount; ++i) {
      sizes[i] = baseSize + (i < remainder ? 1 : 0);
      offsets[i] = step;
      step += sizes[i];
    }
  }

  MPI_Bcast(sizes.data(), pCount, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(offsets.data(), pCount, MPI_INT, 0, MPI_COMM_WORLD);

  int localSize = sizes[pid];
  std::vector<int> localData(localSize);

  const int *sendbuf = (pid == 0 && !inputVec.empty()) ? inputVec.data() : nullptr;
  int *recvbuf = (localSize > 0) ? localData.data() : nullptr;

  MPI_Scatterv(sendbuf, sizes.data(), offsets.data(), MPI_INT, recvbuf, localSize, MPI_INT, 0, MPI_COMM_WORLD);

  int localMax = std::numeric_limits<int>::min();
  if (localSize > 0) {
    localMax = *std::max_element(localData.begin(), localData.end());
  }

  int globalMax = std::numeric_limits<int>::min();
  MPI_Reduce(&localMax, &globalMax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Bcast(&globalMax, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = globalMax;

  return true;
}

bool KlimenkoVMaxMatrixElemsValMPI::PostProcessingImpl() {
  return true;
}

}  // namespace klimenko_v_max_matrix_elems_val
