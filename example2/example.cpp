#include <jem/base/String.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <torch/torch.h>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using jem::System;

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  using jem::String;
  using jem::io::endl;
  using jem::util::Properties;

  Properties  conf;
  String      header;
  int         maxIter;
  double      initValue;
  double      tolerance;

  conf.parseFile ( "input.pro" );

  conf.find ( header,    "header" );
  conf.get  ( maxIter,   "solver.maxIter",   1, 1000 );
  conf.get  ( initValue, "solver.initValue" );
  conf.get  ( tolerance, "solver.tolerance", 1e-20, 1e20 );
  
  System::out() << header << endl;
  torch::manual_seed(0);
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  // using namespace boost::numeric::ublas;
  // matrix<double> m (3, 3);
  // for (unsigned i = 0; i < m.size1 (); ++ i)
  //   for (unsigned j = 0; j < m.size2 (); ++ j)
  //     m (i, j) = 3 * i + j;
  // std::cout << m << std::endl;
  return 0;
}
int main ()
{
  System::exec ( run );
}
