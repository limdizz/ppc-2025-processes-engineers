#pragma once

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimenko_v_max_matrix_elems_val {

class KlimenkoVMaxMatrixElemsValSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KlimenkoVMaxMatrixElemsValSEQ(const InType &in);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::vector<int>> matrix_;
  int max_val_ = 0;
};

}  // namespace klimenko_v_max_matrix_elems_val
