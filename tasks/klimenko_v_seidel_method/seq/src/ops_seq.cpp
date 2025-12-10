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
  return GetInput() > 0;
}

bool KlimenkoVSeidelMethodSEQ::PreProcessingImpl() {
  return true;
}

bool KlimenkoVSeidelMethodSEQ::RunImpl() {
  int n = GetInput();
  if (n <= 0) {
    return false;
  }

  // Оптимизация: используем одномерный массив для матрицы
  std::vector<double> A(n * n, 0.0);
  std::vector<double> b(n, 1.0);
  std::vector<double> x(n, 0.0);
  std::vector<double> x_old(n, 0.0);

  // Инициализация матрицы A
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        A[i * n + j] = n + 1.0;  // диагональное преобладание
      } else {
        A[i * n + j] = 1.0;
      }
    }
  }

  const double epsilon = 1e-6;
  const int max_iterations = 10000;
  bool converge = false;
  int iteration = 0;

  while (!converge && iteration < max_iterations) {
    // Сохраняем предыдущее приближение
    std::copy(x.begin(), x.end(), x_old.begin());

    // Одна итерация метода Зейделя
    for (int i = 0; i < n; i++) {
      double sum = 0.0;

      // Используем уже обновленные значения для j < i
      for (int j = 0; j < i; j++) {
        sum += A[i * n + j] * x[j];
      }

      // Используем старые значения для j > i
      for (int j = i + 1; j < n; j++) {
        sum += A[i * n + j] * x_old[j];
      }

      // Обновляем x[i]
      x[i] = (b[i] - sum) / A[i * n + i];
    }

    // Проверка сходимости
    double diff_norm = 0.0;
    for (int i = 0; i < n; i++) {
      double diff = x[i] - x_old[i];
      diff_norm += diff * diff;
    }
    diff_norm = std::sqrt(diff_norm);

    converge = (diff_norm <= epsilon);
    iteration++;
  }

  // Сумма решения
  double sum = std::accumulate(x.begin(), x.end(), 0.0);
  GetOutput() = static_cast<int>(std::round(sum));

  return true;
}

bool KlimenkoVSeidelMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace klimenko_v_seidel_method
