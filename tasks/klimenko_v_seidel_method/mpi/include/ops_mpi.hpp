#pragma once

#include "klimenko_v_seidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimenko_v_seidel_method {

class KlimenkoVSeidelMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KlimenkoVSeidelMethodMPI(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int computeFinalResult(const std::vector<double> &x, int n);
  void initializeMatrixAndVector(std::vector<double> &flat_matrix, std::vector<double> &b, int n);
  void computeRowDistribution(int n, int size, std::vector<int> &row_counts, std::vector<int> &row_displs,
                              std::vector<int> &matrix_counts, std::vector<int> &matrix_displs);
};

}  // namespace klimenko_v_seidel_method
