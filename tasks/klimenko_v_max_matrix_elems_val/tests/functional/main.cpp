#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"
#include "klimenko_v_max_matrix_elems_val/mpi/include/ops_mpi.hpp"
#include "klimenko_v_max_matrix_elems_val/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace klimenko_v_max_matrix_elems_val {

class KlimenkoVMaxMatrixElemsValFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int n = std::get<0>(params);

    input_data_.resize(n, std::vector<int>(n));
    int val = 1;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        input_data_[i][j] = val++;
      }
    }
    expected_max_ = n * n;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_max_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_max_ = 0;
};

namespace {

TEST_P(KlimenkoVMaxMatrixElemsValFuncTests, FindMatrixMax) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {std::make_tuple(5, "5x5"), std::make_tuple(10, "10x10"),
                                            std::make_tuple(100, "100x100"), std::make_tuple(500, "500x500");

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KlimenkoVMaxMatrixElemsValMPI, InType>(
                                               kTestParam, PPC_SETTINGS_klimenko_v_max_matrix_elems_val),
                                           ppc::util::AddFuncTask<KlimenkoVMaxMatrixElemsValSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_klimenko_v_max_matrix_elems_val));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KlimenkoVMaxMatrixElemsValFuncTests::PrintFuncTestName<KlimenkoVMaxMatrixElemsValFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixFuncTests, KlimenkoVMaxMatrixElemsValFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace klimenko_v_max_matrix_elems_val
