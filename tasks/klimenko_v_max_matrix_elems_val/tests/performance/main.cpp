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
static InType GeneratePerfTestMatrix(int size) {
  InType matrix(size, std::vector<int>(size));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(1, 5000);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix[i][j] = dist(gen);
    }
  }

  matrix[size / 2][size / 2] = 10000;

  return matrix;
}

class KlimenkoVMaxMatrixElemsValPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000;
  InType input_data_;

  void SetUp() override {
    input_data_ = GeneratePerfTestMatrix(kCount_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data > 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KlimenkoVMaxMatrixElemsValPerfTests, FindMatrixMax) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KlimenkoVMaxMatrixElemsValMPI, KlimenkoVMaxMatrixElemsValSEQ>(
        PPC_SETTINGS_klimenko_v_max_matrix_elems_val);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = KlimenkoVMaxMatrixElemsValPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(MatrixTestsPerf, KlimenkoVMaxMatrixElemsValPerfTests, kGtestValues, kPerfTestName);

TEST(KlimenkoVMaxMatrixElemsValMPI, TestPipelineRun) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto matrix = GeneratePerfTestMatrix(5000);
  KlimenkoVMaxMatrixElemsValMPI task(matrix);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());

  auto start_time = std::chrono::high_resolution_clock::now();
  EXPECT_TRUE(task.Run());
  auto end_time = std::chrono::high_resolution_clock::now();

  EXPECT_TRUE(task.PostProcessing());
  EXPECT_GT(task.GetOutput(), 0);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  if (rank == 0) {
    std::cout << "MPI Pipeline time: " << duration.count() << "ms\n";
  }
}

TEST(KlimenkoVMaxMatrixElemsValMPI, TestTaskRun) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto matrix = GeneratePerfTestMatrix(5000);
  KlimenkoVMaxMatrixElemsValMPI task(matrix);

  auto start_time = std::chrono::high_resolution_clock::now();
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  auto end_time = std::chrono::high_resolution_clock::now();

  EXPECT_GT(task.GetOutput(), 0);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  if (rank == 0) {
    std::cout << "MPI Task time: " << duration.count() << "ms\n";
  }
}

TEST(KlimenkoVMaxMatrixElemsValSEQ, TestPipelineRun) {
  auto matrix = GeneratePerfTestMatrix(5000);
  KlimenkoVMaxMatrixElemsValSEQ task(matrix);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());

  auto start_time = std::chrono::high_resolution_clock::now();
  EXPECT_TRUE(task.Run());
  auto end_time = std::chrono::high_resolution_clock::now();

  EXPECT_TRUE(task.PostProcessing());
  EXPECT_GT(task.GetOutput(), 0);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "SEQ Pipeline time: " << duration.count() << "ms\n";
}

TEST(KlimenkoVMaxMatrixElemsValSEQ, TestTaskRun) {
  auto matrix = GeneratePerfTestMatrix(5000);
  KlimenkoVMaxMatrixElemsValSEQ task(matrix);

  auto start_time = std::chrono::high_resolution_clock::now();
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  auto end_time = std::chrono::high_resolution_clock::now();

  EXPECT_GT(task.GetOutput(), 0);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "SEQ Task time: " << duration.count() << "ms\n";
}
}  // namespace klimenko_v_max_matrix_elems_val
