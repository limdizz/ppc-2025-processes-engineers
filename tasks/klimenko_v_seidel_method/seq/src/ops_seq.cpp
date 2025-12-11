#include "klimenko_v_seidel_method/seq/include/ops_seq.hpp"

#include <climits>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "klimenko_v_seidel_method/common/include/common.hpp"

namespace klimenko_v_seidel_method {

KlimenkoVSeidelMethodSEQ::KlimenkoVSeidelMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;

  n_ = 0;
  epsilon_ = 0.0;
  max_iterations_ = 0;
}

bool KlimenkoVSeidelMethodSEQ::ValidationImpl() {
  n_ = GetInput();

  return n_ > 0;
}

bool KlimenkoVSeidelMethodSEQ::PreProcessingImpl() {
  GenerateRandomMatrix(n_, A_, b_);

  std::vector<double> x_exact(n_, 1.0);
  b_.assign(n_, 0.0);

  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      b_[i] += A_[i][j] * x_exact[j];
    }
  }

  for (int i = 0; i < n_; ++i) {
    if (std::abs(A_[i][i]) < 1e-12) {
      return false;
    }
  }

  epsilon_ = 1e-6;
  max_iterations_ = 10000;
  x_.resize(n_, 0.0);

  return true;
}

bool KlimenkoVSeidelMethodSEQ::RunImpl() {
  int iteration = 0;

  while (iteration < max_iterations_) {
    double diff_sq = 0.0;
    for (int i = 0; i < n_; ++i) {
      double old = x_[i];
      double sum_off_diag = 0.0;

      for (int j = 0; j < n_; ++j) {
        if (i != j) {
          sum_off_diag += A_[i][j] * x_[j];
        }
      }

      x_[i] = (b_[i] - sum_off_diag) / A_[i][i];
      diff_sq += std::pow(x_[i] - old, 2);
    }

    if (std::sqrt(diff_sq) < epsilon_) {
      break;
    }

    ++iteration;
  }

  double sum = std::accumulate(x_.begin(), x_.end(), 0.0);
  GetOutput() = static_cast<int>(std::round(sum));
  return true;
}

bool KlimenkoVSeidelMethodSEQ::PostProcessingImpl() {
  return true;
}

void KlimenkoVSeidelMethodSEQ::GenerateRandomMatrix(int size, std::vector<std::vector<double>> &matrix,
                                                    std::vector<double> &vector) {
  matrix.resize(size);
  for (int i = 0; i < size; ++i) {
    matrix[i].assign(size, 0.0);
  }
  vector.resize(size, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 10);
  std::uniform_int_distribution<> dist_diag(1, 5);

  for (int i = 0; i < size; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < size; ++j) {
      if (i != j) {
        matrix[i][j] = static_cast<double>(dist(gen));
        row_sum += std::abs(matrix[i][j]);
      }
    }

    matrix[i][i] = row_sum + static_cast<double>(dist_diag(gen));
  }
}

}  // namespace klimenko_v_seidel_method
