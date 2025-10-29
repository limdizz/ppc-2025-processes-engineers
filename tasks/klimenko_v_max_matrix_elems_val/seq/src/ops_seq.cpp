#include "klimenko_v_max_matrix_elems_val/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"
#include "util/include/util.hpp"

namespace klimenko_v_max_matrix_elems_val {

KlimenkoVMaxMatrixElemsValSEQ::KlimenkoVMaxMatrixElemsValSEQ(const InType &in) : Task() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KlimenkoVMaxMatrixElemsValSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty() || matrix[0].empty()) {
    return false;
  }
  return true;
}

bool KlimenkoVMaxMatrixElemsValSEQ::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool KlimenkoVMaxMatrixElemsValSEQ::RunImpl() {
  const auto &matrix = GetInput();
  int global_max = std::numeric_limits<int>::min();

  for (const auto &row : matrix) {
    auto row_max = *std::max_element(row.begin(), row.end());
    if (row_max > global_max) {
      global_max = row_max;
    }
  }

  GetOutput() = global_max;
  return true;
}

bool KlimenkoVMaxMatrixElemsValSEQ::PostProcessingImpl() {
  return GetOutput() != std::numeric_limits<int>::min();
}

}  // namespace klimenko_v_max_matrix_elems_val
