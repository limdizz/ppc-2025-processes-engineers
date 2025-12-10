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

  const double tau = 0.5;
  const double epsilon = 1e-6;
  const int max_iterations = 1000;
  int iteration = 0;

  for (; iteration < max_iterations; iteration++) {
    double diff = 0.0;

    for (int i = 0; i < n; i++) {
      double ax_i = 0.0;

      for (int j = 0; j < n; j++) {
        ax_i += a[i][j] * x[j];
      }

      double old = x[i];

      x[i] = x[i] - tau * (ax_i - b[i]);

      double d = x[i] - old;
      diff += d * d;
    }

    diff = std::sqrt(diff);
    if (diff < epsilon) {
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
