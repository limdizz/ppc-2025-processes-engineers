#include "klimenko_v_seidel_method/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

#include "klimenko_v_seidel_method/common/include/common.hpp"

namespace klimenko_v_seidel_method {

KlimenkoVSeidelMethodSEQ::KlimenkoVSeidelMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KlimenkoVSeidelMethodSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool KlimenkoVSeidelMethodSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KlimenkoVSeidelMethodSEQ::RunImpl() {
  int n = GetInput();
  if (n <= 0) {
    return false;
  }

  std::vector<double> x(n, 0.0);
  std::vector<double> b(n, 1.0);
  std::vector<std::vector<double>> a(n, std::vector<double>(n, 0.0));

  // A = Identity
  for (int i = 0; i < n; i++) {
    a[i][i] = 1.0;
  }

  const double epsilon = 1e-6;
  const int max_iterations = 1000;

  for (int iteration = 0; iteration < max_iterations; iteration++) {
    std::vector<double> x_old = x;
    double max_diff = 0.0;

    for (int i = 0; i < n; i++) {
      double sum = 0.0;

      //
      for (int j = 0; j < i; j++) {
        sum += a[i][j] * x[j];
      }

      for (int j = i + 1; j < n; j++) {
        sum += a[i][j] * x_old[j];
      }

      double x_new = (b[i] - sum) / a[i][i];

      double diff = std::abs(x_new - x[i]);
      if (diff > max_diff) {
        max_diff = diff;
      }

      x[i] = x_new;
    }

    if (max_diff < epsilon) {
      break;
    }
  }

  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }

  GetOutput() = static_cast<int>(std::round(sum));
  return true;
}

bool KlimenkoVSeidelMethodSEQ::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace klimenko_v_seidel_method
