/**
 * @file learning_curve_impl.h
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
 *
 * TODO: 
 * - Add training errors.
 * - Write it much cleaner.
 *
 *
 */

#ifndef LEARNING_CURVE_IMPL_H
#define LEARNING_CURVE_IMPL_H

template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
LCurve_HPT<MODEL,LOSS,CV>::LCurve_HPT(  const arma::irowvec& Ns,
                                        const double& repeat,
                                        const double& tune_ratio    )
{
  tune_ratio_ = tune_ratio;
  Ns_ = Ns;   
  test_errors_.resize(repeat,Ns_.n_elem);
  train_errors_.resize(repeat,Ns_.n_elem);
  repeat_ = repeat;  
} 
      
template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
template<class... Ts>
std::tuple<arma::mat, arma::mat> 
  LCurve_HPT<MODEL,LOSS,CV>::Generate(  const std::string filename,
                                        const arma::mat& inputs,
                                        const arma::rowvec& labels,
                                        const Ts&... args            )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );

  #pragma omp parallel for collapse(2)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = utils::data::Split(inputs, labels, Ns_(i));
      arma::mat Xtrn = std::get<0>(res);
      arma::mat Xtst = std::get<1>(res);
      arma::rowvec ytrn = std::get<2>(res);
      arma::rowvec ytst = std::get<3>(res);
      mlpack::HyperParameterTuner<MODEL, LOSS, CV> hpt(0.2, Xtrn, ytrn);
      hpt.Optimize(args...);
      MODEL bestmodel = hpt.BestModel();
      test_errors_(j,i) = bestmodel.ComputeError(Xtst, ytst);
      train_errors_(j,i) = hpt.BestObjective();
    }
  }

    arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                      arma::stddev(train_errors_));
    arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                     arma::stddev(test_errors_));
    arma::mat results = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);
    utils::Save(filename,results);

    stats_ = std::make_tuple(std::move(train),
                             std::move(test));
    return stats_;
}

template<class MODEL,
         class LOSS>
LCurve<MODEL,LOSS>::LCurve(  const arma::irowvec& Ns,
                             const double& repeat      )
{
  Ns_ = Ns;   
  test_errors_.resize(repeat,Ns_.n_elem);
  train_errors_.resize(repeat,Ns_.n_elem);
  repeat_ = repeat;  
} 
      
template<class MODEL,
         class LOSS>
template<class... Ts>
std::tuple<arma::mat, arma::mat> 
  LCurve<MODEL,LOSS>::Generate( const std::string filename,
                                const arma::mat& inputs,
                                const arma::rowvec& labels,
                                const Ts&... args           )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );

  #pragma omp parallel for collapse(2)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = utils::data::Split(inputs, labels, Ns_(i));
      arma::mat Xtrn = std::get<0>(res);
      arma::mat Xtst = std::get<1>(res);
      arma::rowvec ytrn = std::get<2>(res);
      arma::rowvec ytst = std::get<3>(res);
      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = model.ComputeError(Xtst, ytst);
      train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
    }
  }

    arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                      arma::stddev(train_errors_));
    arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                     arma::stddev(test_errors_));
    arma::mat results = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);
    utils::Save(filename,results);

    stats_ = std::make_tuple(std::move(train),
                             std::move(test));
    return stats_;
}

#endif

