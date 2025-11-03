#include "klimenko_v_max_matrix_elems_val/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"
#include "util/include/util.hpp"

namespace klimenko_v_max_matrix_elems_val {

KlimenkoVMaxMatrixElemsValSEQ::KlimenkoVMaxMatrixElemsValSEQ(const InType &in) : matrix_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::numeric_limits<int>::min();
}

bool KlimenkoVMaxMatrixElemsValSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool KlimenkoVMaxMatrixElemsValSEQ::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool KlimenkoVMaxMatrixElemsValSEQ::RunImpl() {
  const auto &data = GetInput();
  if (data.empty()) {
    return false;
  }

  GetOutput() = *std::max_element(data.begin(), data.end());
  return true;
}

bool KlimenkoVMaxMatrixElemsValSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace klimenko_v_max_matrix_elems_val
