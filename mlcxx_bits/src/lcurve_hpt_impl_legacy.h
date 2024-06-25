/**
 * @file lcurve_hpt_impl_legacy.h
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
 *
 * TODO: 
 * - 
 * - LOSS in the bare version is not doing anything! Why did I put it there?
 *   # Now, I remember why... It is planned for situations when the 
 *   ComputeError or ComputeAccuracy are not available!
 * - omp parallel is not working with NN implementations because two omp 
 *   parallels are recursing probably?!
 *   
 *
 *
 */

#ifndef LCURVE_HPT_IMPL_LEGACY_H
#define LCURVE_HPT_IMPL_LEGACY_H

namespace src {
namespace regression {



//=============================================================================
// LCurve_HPT 
//=============================================================================

template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
LCurve_HPT<MODEL,LOSS,CV>::LCurve_HPT ( const arma::irowvec& Ns,
                                        const double& repeat,
                                        const double& tune_ratio ) :
Ns_(Ns), repeat_(repeat), tune_ratio_(tune_ratio)
{
  test_errors_.resize(repeat_,Ns_.n_elem);
  train_errors_.resize(repeat_,Ns_.n_elem);
} 
      
template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
template<class... Ts>
void LCurve_HPT<MODEL,LOSS,CV>::Generate ( const std::filesystem::path filename,
                                           const arma::mat& inputs,
                                           const arma::rowvec& labels,
                                           const Ts&... args )
{
  Generate(inputs, labels, args...);
  Save(filename);
}

template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
template<class... Ts>
void LCurve_HPT<MODEL,LOSS,CV>::Generate ( const std::filesystem::path filename,
                                           const arma::mat& inputs,
                                           const arma::mat& labels,
                                           const Ts&... args )
{
  Generate(inputs, labels, args...);
  Save(filename);
}
template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
template<class... Ts>
void LCurve_HPT<MODEL,LOSS,CV>::Generate ( const arma::mat& inputs,
                                           const arma::rowvec& labels,
                                           const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );

  ProgressBar pb("LCurveHPT.Generate", Ns_.n_elem*repeat_);
  #pragma omp parallel for collapse(2)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = utils::data::Split(inputs, labels, size_t(Ns_(i)));
      arma::mat Xtrn = std::get<0>(res);
      arma::mat Xtst = std::get<1>(res);
      arma::rowvec ytrn = std::get<2>(res);
      arma::rowvec ytst = std::get<3>(res);
      mlpack::HyperParameterTuner<MODEL,LOSS,CV> hpt(tune_ratio_, Xtrn, ytrn);
      hpt.Optimize(args...);
      MODEL bestmodel = hpt.BestModel();
      bestmodel.Train(Xtrn, ytrn);
      test_errors_(j,i) = bestmodel.ComputeError(Xtst, ytst);
      train_errors_(j,i) = bestmodel.ComputeError(Xtrn, ytrn);
      /* pb.Update(); */
    }
  }

    arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                      arma::stddev(train_errors_));
    arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                     arma::stddev(test_errors_));
    results_ = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);


    stats_ = std::make_tuple(std::move(train),
                             std::move(test));
}

template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
template<class... Ts>
void LCurve_HPT<MODEL,LOSS,CV>::Generate ( const arma::mat& inputs,
                                           const arma::mat& labels,
                                           const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );

  ProgressBar pb("LCURVE_HPT.Generate", Ns_.n_elem*repeat_);
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = utils::data::Split(inputs, labels, size_t(Ns_(i)));
      arma::mat Xtrn = std::get<0>(res);
      arma::mat Xtst = std::get<1>(res);
      arma::mat ytrn = std::get<2>(res);
      arma::mat ytst = std::get<3>(res);
      mlpack::HyperParameterTuner<MODEL,LOSS,CV> hpt(tune_ratio_, Xtrn, ytrn);
      hpt.Optimize(args...);
      MODEL bestmodel = hpt.BestModel();
      bestmodel.Train(Xtrn, ytrn);
      test_errors_(j,i) = bestmodel.ComputeError(Xtst, ytst);
      train_errors_(j,i) = bestmodel.ComputeError(Xtrn, ytrn);
      /* pb.Update(); */
    }
  }

    arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                      arma::stddev(train_errors_));
    arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                     arma::stddev(test_errors_));
    results_ = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);


    stats_ = std::make_tuple(std::move(train),
                             std::move(test));
}
template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
template<class... Ts>
void  LCurve_HPT<MODEL,LOSS,CV>::Save ( const std::filesystem::path filename )
{
  utils::Save(filename, results_);
}

} // namespace regression
} // namespace src

namespace src {
namespace classification {


//=============================================================================
// LCurve_HPT 
//=============================================================================

template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
LCurve_HPT<MODEL,LOSS,CV>::LCurve_HPT ( const arma::irowvec& Ns,
                                        const double& repeat,
                                        const double& tune_ratio ) :
Ns_(Ns), repeat_(repeat), tune_ratio_(tune_ratio)
{
  test_errors_.resize(repeat_,Ns_.n_elem);
  train_errors_.resize(repeat_,Ns_.n_elem);
} 
      
template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
template<class... Ts>
void LCurve_HPT<MODEL,LOSS,CV>::Generate ( const std::filesystem::path filename,
                                           const arma::mat& inputs,
                                           const arma::Row<size_t>& labels,
                                           const Ts&... args )
{
  Generate(inputs, labels, args...);
  Save(filename);
}

template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
template<class... Ts>
void LCurve_HPT<MODEL,LOSS,CV>::Generate ( const arma::mat& inputs,
                                           const arma::Row<size_t>& labels,
                                           const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );

  #pragma omp parallel for collapse(2)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = utils::data::StratifiedSplit(inputs,
                                                    labels, size_t(Ns_(i)));
      arma::mat Xtrn = std::get<0>(res);
      arma::mat Xtst = std::get<1>(res);
      arma::rowvec ytrn = std::get<2>(res);
      arma::Row<size_t> ytst = std::get<3>(res);
      mlpack::HyperParameterTuner<MODEL,LOSS,CV> hpt(tune_ratio_, Xtrn, ytrn);
      hpt.Optimize(args...);
      MODEL bestmodel = hpt.BestModel();
      bestmodel.Train(Xtrn, ytrn);
      test_errors_(j,i) = (100. - bestmodel.ComputeAccuracy(Xtst, ytst))
                                                                        / 100.;
      train_errors_(j,i) = (100. - bestmodel.ComputeAccuracy(Xtrn, ytrn))
                                                                        / 100.;
    }
  }

    arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                      arma::stddev(train_errors_));
    arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                     arma::stddev(test_errors_));
    results_ = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);


    stats_ = std::make_tuple(std::move(train),
                             std::move(test));
}
template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
template<class... Ts>
void  LCurve_HPT<MODEL,LOSS,CV>::Save ( const std::filesystem::path filename )
{
  utils::Save(filename, results_);
}

} // namespace classification
} // namespace src

#endif
