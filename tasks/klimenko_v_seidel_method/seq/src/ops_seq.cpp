#include "klimenko_v_seidel_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cmath>  // для std::abs
#include <numeric>
#include <vector>

#include "klimenko_v_seidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimenko_v_seidel_method {

KlimenkoVSeidelMethodSEQ::KlimenkoVSeidelMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KlimenkoVSeidelMethodSEQ::ValidationImpl() {
  n = GetInput();
  if (n <= 0) {
    return false;
  }
  return true;
}

bool KlimenkoVSeidelMethodSEQ::PreProcessingImpl() {
  generateRandomMatrix(n, A, b);

  std::vector<double> x_exact(n, 1.0);
  b.assign(n, 0.0);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      b[i] += A[i][j] * x_exact[j];
    }
  }

  for (int i = 0; i < n; ++i) {
    if (std::abs(A[i][i]) < 1e-12) {
      return false;
    }
  }

  epsilon = 1e-6;
  max_iterations = 10000;
  x.resize(n, 0.0);

  int iteration = 0;

  while (iteration < max_iterations) {
    double diff_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      double old = x[i];
      double sum_off_diag = 0.0;

      for (int j = 0; j < n; ++j) {
        if (i != j) {
          sum_off_diag += A[i][j] * x[j];
        }
      }

      x[i] = (b[i] - sum_off_diag) / A[i][i];
      diff_sq += std::pow(x[i] - old, 2);
    }

    if (std::sqrt(diff_sq) < epsilon) {
      break;
    }

    ++iteration;
  }

  double sum = std::accumulate(x.begin(), x.end(), 0.0);
  GetOutput() = static_cast<int>(std::round(sum));

  return true;
}

bool KlimenkoVSeidelMethodSEQ::RunImpl() {
  return true;
}

bool KlimenkoVSeidelMethodSEQ::PostProcessingImpl() {
  return true;
}

bool KlimenkoVSeidelMethodSEQ::converge(const std::vector<double> &x_new) {
  double residual_norm = 0.0;
  for (int i = 0; i < n; ++i) {
    double Ax_i = 0.0;
    for (int j = 0; j < n; ++j) {
      Ax_i += A[i][j] * x_new[j];
    }
    residual_norm += std::pow(Ax_i - b[i], 2);
  }
  return std::sqrt(residual_norm) < epsilon;
}

void KlimenkoVSeidelMethodSEQ::generateRandomMatrix(int size, std::vector<std::vector<double>> &matrix,
                                                    std::vector<double> &vector) {
  matrix.resize(size);
  for (int i = 0; i < size; ++i) {
    matrix[i].assign(size, 0.0);
  }
  vector.resize(size, 0.0);

  std::srand(static_cast<unsigned>(std::time(nullptr)));

  for (int i = 0; i < size; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < size; ++j) {
      if (i != j) {
        matrix[i][j] = static_cast<double>(std::rand() % 10 + 1);
        row_sum += std::abs(matrix[i][j]);
      }
    }

    matrix[i][i] = row_sum + static_cast<double>(std::rand() % 5 + 1);
    // vector[i] = static_cast<double>(std::rand() % 20 + 1);
  }
}

}  // namespace klimenko_v_seidel_method
