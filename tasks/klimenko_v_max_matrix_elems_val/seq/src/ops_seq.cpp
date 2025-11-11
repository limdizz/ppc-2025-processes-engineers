#include "klimenko_v_max_matrix_elems_val/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"
#include "util/include/util.hpp"

namespace klimenko_v_max_matrix_elems_val {

KlimenkoVMaxMatrixElemsValSEQ::KlimenkoVMaxMatrixElemsValSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0;
}

bool KlimenkoVMaxMatrixElemsValSEQ::ValidationImpl() {
  return ((!GetInput().empty()) && (GetOutput() == 0));
}

bool KlimenkoVMaxMatrixElemsValSEQ::PreProcessingImpl() {
  return !GetInput().empty();
}

bool KlimenkoVMaxMatrixElemsValSEQ::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  int max_element = matrix[0][0];
  for (const auto &row : matrix) {
    for (int element : row) {
      max_element = std::max(element, max_element);
    }
  }

  GetOutput() = max_element;
  return true;
}

bool KlimenkoVMaxMatrixElemsValSEQ::PostProcessingImpl() {
  return !GetInput().empty();
}

}  // namespace klimenko_v_max_matrix_elems_val
