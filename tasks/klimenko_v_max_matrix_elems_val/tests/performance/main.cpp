#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"
#include "klimenko_v_max_matrix_elems_val/mpi/include/ops_mpi.hpp"
#include "klimenko_v_max_matrix_elems_val/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace klimenko_v_max_matrix_elems_val {

class KlimenkoVMaxMatrixElemsValPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000;
  std::unique_ptr<InType> input_data_;

  void SetUp() override {
    input_data_ = std::make_unique<InType>(kCount_, std::vector<int>(kCount_));
    int val = 1;
    for (int i = 0; i < kCount_; i++) {
      for (int j = 0; j < kCount_; j++) {
        (*input_data_)[i][j] = val++;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data > 0;
  }

  InType GetTestInputData() final {
    return *input_data_;
  }
};

TEST_P(KlimenkoVMaxMatrixElemsValPerfTests, TestPipelineRun) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KlimenkoVMaxMatrixElemsValMPI, KlimenkoVMaxMatrixElemsValSEQ>(
        PPC_SETTINGS_klimenko_v_max_matrix_elems_val);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = KlimenkoVMaxMatrixElemsValPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerfTests, KlimenkoVMaxMatrixElemsValPerfTests, kGtestValues, kPerfTestName);

}  // namespace klimenko_v_max_matrix_elems_val
