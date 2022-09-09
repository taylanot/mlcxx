#include <torch/torch.h>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

int main() {
  torch::manual_seed(0);
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  using namespace boost::numeric::ublas;
  matrix<double> m (3, 3);
  for (unsigned i = 0; i < m.size1 (); ++ i)
    for (unsigned j = 0; j < m.size2 (); ++ j)
      m (i, j) = 3 * i + j;
  std::cout << m << std::endl;
}

