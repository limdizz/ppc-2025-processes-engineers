#pragma once

#include <vector>

#include "klimenko_v_seidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimenko_v_seidel_method {

class KlimenkoVSeidelMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KlimenkoVSeidelMethodSEQ(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::vector<double>> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  int n_;
  double epsilon_;
  int max_iterations_;

  void GenerateRandomMatrix(int size, std::vector<std::vector<double>> &matrix, std::vector<double> &vector);
};

}  // namespace klimenko_v_seidel_method
