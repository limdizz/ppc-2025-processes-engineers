#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
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

static InType GenerateTestMatrix(int size) {
  if (size == 0) {
    return std::vector<std::vector<int>>{};
  }
  InType matrix(size, std::vector<int>(size));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(1, 50000);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix[i][j] = dist(gen);
    }
  }

  matrix[size / 2][size / 2] = 99999;

  return matrix;
}

class KlimenkoVMaxMatrixElemsValFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "Size_" + std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);

    input_data_ = GenerateTestMatrix(size);
    reference_max_ = CalculateReferenceMax(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Всегда проверяем результат для всех типов задач
    bool result_correct = (output_data == reference_max_);
    if (!result_correct) {
      std::cout << "Expected " << reference_max_ << ", got " << output_data << "\n";
    }
    return result_correct;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType reference_max_ = 0;

  static OutType CalculateReferenceMax(const InType &matrix) {
    if (matrix.empty()) {
      return 0;
    }

    OutType max_val = matrix[0][0];
    for (const auto &row : matrix) {
      for (int val : row) {
        max_val = std::max(val, max_val);
      }
    }
    return max_val;
  }
};

namespace {

TEST_P(KlimenkoVMaxMatrixElemsValFuncTests, TestFindMaxElement) {
  ExecuteTest(GetParam());
}

// Тестовые случаи
const std::array<TestType, 5> kTestParam = {std::make_tuple(7, "7x7"), std::make_tuple(20, "20x20"),
                                            std::make_tuple(50, "50x50"), std::make_tuple(100, "100x100"),
                                            std::make_tuple(11, "11x11")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KlimenkoVMaxMatrixElemsValMPI, InType>(
                                               kTestParam, PPC_SETTINGS_klimenko_v_max_matrix_elems_val),
                                           ppc::util::AddFuncTask<KlimenkoVMaxMatrixElemsValSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_klimenko_v_max_matrix_elems_val));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = KlimenkoVMaxMatrixElemsValFuncTests::PrintFuncTestName<KlimenkoVMaxMatrixElemsValFuncTests>;

INSTANTIATE_TEST_SUITE_P(FuncTests, KlimenkoVMaxMatrixElemsValFuncTests, kGtestValues, kFuncTestName);

// Индивидуальные тест кейсы

TEST(KlimenkoVMaxMatrixElemsValFuncTestsMPI, testZeroMatrix) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }
  auto matrix = GenerateTestMatrix(0);
  KlimenkoVMaxMatrixElemsValMPI task(matrix);

  EXPECT_FALSE(task.Validation());

  EXPECT_FALSE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
  EXPECT_FALSE(task.PostProcessing());
}

TEST(KlimenkoVMaxMatrixElemsValFuncTestsSEQ, testZeroMatrix) {
  auto matrix = GenerateTestMatrix(0);
  KlimenkoVMaxMatrixElemsValSEQ task(matrix);

  EXPECT_FALSE(task.Validation());

  EXPECT_FALSE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
  EXPECT_FALSE(task.PostProcessing());
}

TEST(KlimenkoVMaxMatrixElemsValFuncTestsMPI, testSmallMatrix) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }
  auto matrix = GenerateTestMatrix(30);
  KlimenkoVMaxMatrixElemsValMPI task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_GT(task.GetOutput(), 0);
}

TEST(KlimenkoVMaxMatrixElemsValFuncTestsSEQ, testSmallMatrix) {
  auto matrix = GenerateTestMatrix(30);
  KlimenkoVMaxMatrixElemsValSEQ task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_GT(task.GetOutput(), 0);
}

}  // namespace

}  // namespace klimenko_v_max_matrix_elems_val
