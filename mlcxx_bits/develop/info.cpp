/**
 * @file info.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying out the info extractor of mlpack
 *
 */
#define DTYPE double

#include <headers.h>

using LREG  = algo::classification::LogisticRegression<>;  // x
using LREG2  = mlpack::LogisticRegression<>;  // x
using Dataset = data::oml::Dataset<size_t>;

template<class MODEL, class METRIC=mlpack::Accuracy,class CV=mlpack::SimpleCV<MODEL,METRIC> >
class FOO
{
  public:
  FOO( const Dataset& data ) : data_(data) { }

  CV Do( )
  {
    if constexpr (!mlpack::MetaInfoExtractor<MODEL>::TakesNumClasses)
      return CV(0.2,data_.inputs_,data_.labels_);
    else
      return CV(0.2,data_.inputs_,data_.labels_,data_.num_class_);
  };
  
  Dataset data_;
  mlpack::MetaInfoExtractor<MODEL> info_;
};

int main(int argc, char** argv)
{

    arma::wall_clock total_timer;
    total_timer.tic();

    Dataset data(11);

    arma::irowvec Ns = {10,5};
    arma::Row<DTYPE> lambdas = {1,10};
    src::LCurveHPT<LREG,mlpack::Accuracy> lcurve(Ns, 2,0.2);
    lcurve.Additive(data.inputs_,data.labels_,lambdas);
    PRINT_VAR(lcurve.GetResults());
    PRINT_TIME(total_timer.toc());

    return 0;
}

