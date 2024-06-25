/**
 * @file lcurve_impl.h
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
 *
 * TODO: 
 *
 * - You should put a warning for not using parallelization for neurla networks.
 *   
 *
 *
 */

#ifndef LCURVE_IMPL_H
#define LCURVE_IMPL_H

namespace src {

//=============================================================================
// LCurve
//=============================================================================
template<class MODEL,
         class LOSS,class O>
LCurve<MODEL,LOSS,O>::LCurve ( const arma::irowvec& Ns,
                               const double repeat,
                               const bool parallel, 
                               const bool save,
                               const std::string name,
                               const bool save_data ) :

repeat_(repeat), Ns_(Ns), parallel_(parallel), save_(save),
save_data_(save_data), name_(name)
{
  test_errors_.resize(repeat_,Ns_.n_elem);
  train_errors_.resize(repeat_,Ns_.n_elem);
}

//=============================================================================
// LCurve::Bootstrap
//=============================================================================     
template<class MODEL,
         class LOSS,class O>
template<class T, class... Ts>
void LCurve<MODEL,LOSS,O>::Bootstrap ( const arma::Mat<O>& inputs,
                                       const T& labels,
                                       const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(labels.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );
  LOSS loss;

  ProgressBar pb("LCurve.Bootstrap", Ns_.n_elem*repeat_);

  #pragma omp parallel for collapse(2) if(parallel_)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = utils::data::Split(inputs, labels, size_t(Ns_(i)));
      arma::Mat<O> Xtrn = std::get<0>(res);
      arma::Mat<O> Xtst = std::get<1>(res);
      T ytrn = std::get<2>(res);
      T ytst = std::get<3>(res);

      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = loss.Evaluate(model, Xtst, ytst);
      train_errors_(j,i) = loss.Evaluate(model, Xtrn, ytrn);
      pb.Update();
    }
  }

    arma::Mat<O> train = arma::join_cols(arma::mean(train_errors_),
                                         arma::stddev(train_errors_));

    arma::Mat<O> test = arma::join_cols(arma::mean(test_errors_),
                                        arma::stddev(test_errors_));

    results_ = 
      arma::join_cols(arma::conv_to<arma::Row<O>>::from(Ns_), train, test);

    stats_ = std::make_tuple(std::move(train),
                             std::move(test));

}

//=============================================================================
// LCurve::Additive
//=============================================================================     
template<class MODEL,
         class LOSS,class O>
template<class T, class... Ts>
void LCurve<MODEL,LOSS,O>::Additive ( const arma::Mat<O>& inputs,
                                      const T& labels,
                                      const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(labels.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );
  LOSS loss;

  ProgressBar pb("LCurve.Additive", Ns_.n_elem*repeat_);

  #pragma omp parallel for if(parallel_)
  for(size_t j=0; j < size_t(repeat_); j++)
  {
    const auto res = utils::data::Split(inputs,labels,size_t(Ns_(0)));

    arma::Mat Xtrn = std::get<0>(res);
    arma::Mat Xrest = std::get<1>(res);

    T ytrn = std::get<2>(res);
    T yrest = std::get<3>(res);

    MODEL model(Xtrn, ytrn, args...);
    test_errors_(j,0) = loss.Evaluate(model,Xrest,yrest);
    train_errors_(j,0) = loss.Evaluate(model, Xtrn, ytrn);
    for (size_t i=1; i < size_t(Ns_.n_elem) ; i++)
    {
      utils::data::Migrate(Xtrn,ytrn,Xrest,yrest, Ns_[i]-Ns_[i-1]);
      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = loss.Evaluate(model,Xrest,yrest);
      train_errors_(j,i) = loss.Evaluate(model, Xtrn, ytrn);
      pb.Update();
    }
    pb.Update();
  }

    arma::Mat<O> train = arma::join_cols(arma::mean(train_errors_),
                                      arma::stddev(train_errors_));
    arma::Mat<O> test = arma::join_cols(arma::mean(test_errors_),
                                     arma::stddev(test_errors_));

    results_ = 
      arma::join_cols(arma::conv_to<arma::Row<O>>::from(Ns_), train, test);

    stats_ = std::make_tuple(std::move(train),
                             std::move(test)); 

}

template<class MODEL,
         class LOSS,class O>
template<class T, class... Ts>
void LCurve<MODEL,LOSS,O>::Split( const T& trainset,
                                  const T& testset,
                                  const Ts&... args )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(trainset.inputs_.n_cols), 
        "There are not enough data for learning curve generation!" );
  BOOST_ASSERT_MSG( int(trainset.labels_.n_rows) == int(1) &&
                    int(testset.labels_.n_rows) == int(1), 
                    "Only 1D outputs are allowed!" );
  LOSS loss;

  ProgressBar pb("LCurve.Split", Ns_.n_elem*repeat_);

  #pragma omp parallel for collapse(2) if(parallel_)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = utils::data::Split(trainset.inputs_,
                                          trainset.labels_,
                                          size_t(Ns_(i)));

      arma::Mat<O> Xtrn = std::get<0>(res);
      arma::Row<O> ytrn = std::get<2>(res);

      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = loss.Evaluate(model, testset.inputs_,
                          arma::conv_to<arma::rowvec>::from(testset.labels_));
      train_errors_(j,i) = loss.Evaluate(model, Xtrn, ytrn);
      pb.Update();
    }
  }

    arma::Mat<O> train = arma::join_cols(arma::mean(train_errors_),
                                         arma::stddev(train_errors_));
    arma::Mat<O> test = arma::join_cols(arma::mean(test_errors_),
                                        arma::stddev(test_errors_));

    results_ = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

    stats_ = std::make_tuple(std::move(train),
                             std::move(test)); 
}

} // namespace src
#endif
