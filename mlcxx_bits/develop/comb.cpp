/**
 * @file comb.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to create a function that creates combinations for ova related
 * problems...
 */

#include <headers.h>


int main() 
{
  arma::Row<size_t>  labels = {0,1,2,2,2,0};
  arma::mat scores =  {
        {0.18205878, 0.46212909, 0.35581214},
        {0.65738127, 0.17132261, 0.17129612},
        {0.03807826, 0.56784481, 0.39407693},
        {0.41686469, 0.01211874, 0.57101657},
        {0.67865488, 0.173111,   0.14823412},
        {0.18115758, 0.3005149,  0.51832752}
    };
  arma::inplace_trans(scores);

  arma::Row<size_t> unq = arma::unique(labels);

  mlpack::ROCAUCScore<1> auc;
  arma::rowvec aucscores(unq.n_elem);
  for (size_t i=0;i<unq.n_elem;i++)
  {
    auto binlabels = arma::conv_to<arma::Row<size_t>>::from(labels==unq(i));
    aucscores(i) = auc.Evaluate(binlabels,scores.row(unq(i)));
  }
  PRINT_VAR(arma::mean(aucscores));
  PRINT_VAR(aucscores);

  scores.print("hello");
  scores.resize(3,7);
  scores.print("wello");
  return 0;
}
