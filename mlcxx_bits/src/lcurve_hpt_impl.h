 /**
 * @file lcurve_hpt_impl.h
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator with inside hyperparameter tuning for an MLALgorithm
 * given a dataset
 *
 * TODO: 
 *
 * - You should put a warning for not using parallelization for neurla networks.
 *   
 *
 *
 */

#ifndef LCURVE_HPT_IMPL_H
#define LCURVE_HPT_IMPL_H

namespace src {

//=============================================================================
// LCurve
//=============================================================================
template<class MODEL,
         class LOSS,
         class SPLIT,
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
LCurveHPT<MODEL,LOSS,SPLIT,CV,OPT,O>::LCurveHPT ( const arma::irowvec& Ns,
                                                  const double repeat,
                                                  const double cvp,
                                                  const bool parallel, 
                                                  const bool prog ) :
repeat_(repeat), Ns_(Ns), parallel_(parallel), prog_(prog), cvp_(cvp)
{
  test_errors_.resize(repeat_,Ns_.n_elem);
}
//=============================================================================
// LCurve::Bootstrap
//=============================================================================     
template<class MODEL,
         class LOSS,
         class SPLIT,
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
template<class T, class... Ts>
void LCurveHPT<MODEL,LOSS,SPLIT,CV,OPT,O>::Bootstrap ( const arma::Mat<O>& inputs,
                                                       const T& labels,
                                                       const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(labels.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );
  LOSS loss;

  ProgressBar pb("LCurveHPT.Bootstrap", Ns_.n_elem*repeat_);

  #pragma omp parallel for collapse(2) if(parallel_)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = split_(inputs, labels, size_t(Ns_(i)));
      arma::Mat<O> Xtrn = std::get<0>(res);
      arma::Mat<O> Xtst = std::get<1>(res);
      T ytrn = std::get<2>(res);
      T ytst = std::get<3>(res);

      mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>>
                                                          hpt(cvp_, Xtrn, ytrn);

      hpt.Optimize(args...);
      MODEL model = std::move(hpt.BestModel());
      model.Train(Xtrn, ytrn);

      test_errors_(j,i) = loss.Evaluate(model,Xtst,ytst);
      
      if (prog_)
        pb.Update();
    }
  }


}

//=============================================================================
// LCurve::Additive
//=============================================================================     
template<class MODEL,
         class LOSS,
         class SPLIT,
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
template<class T, class... Ts>
void LCurveHPT<MODEL,LOSS,SPLIT,CV,OPT,O>::Additive ( const arma::Mat<O>& inputs,
                                                      const T& labels,
                                                      const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(labels.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );
  LOSS loss;

  ProgressBar pb("LCurveHPT.Additive", Ns_.n_elem*repeat_);

  #pragma omp parallel for if(parallel_)
  for(size_t j=0; j < size_t(repeat_); j++)
  {
    const auto res = split_(inputs,labels,size_t(Ns_(0)));

    arma::Mat<O> Xtrn = std::get<0>(res);
    arma::Mat<O> Xrest = std::get<1>(res);

    T ytrn = std::get<2>(res);
    T yrest = std::get<3>(res);

    
    mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>>
                                                          hpt(cvp_, Xtrn, ytrn);
    hpt.Optimize(args...);
    MODEL model = hpt.BestModel();
    model.Train(Xtrn, ytrn);
    test_errors_(j,0) = loss.Evaluate(model,Xrest,yrest);

    for (size_t i=1; i < size_t(Ns_.n_elem) ; i++)
    {
      data::Migrate(Xtrn,ytrn,Xrest,yrest, Ns_[i]-Ns_[i-1]);

      mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>>
                                                          hpt(cvp_, Xtrn, ytrn);
      hpt.Optimize(args...);
      MODEL model = std::move(hpt.BestModel());
      model.Train(Xtrn, ytrn);
      test_errors_(j,i) = loss.Evaluate(model,Xrest,yrest);

      if (prog_)
        pb.Update();
    }
    if (prog_)
      pb.Update();
  }

}
//=============================================================================
// LCurve::Split
//=============================================================================     
template<class MODEL,
         class LOSS,
         class SPLIT,
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
template<class T, class... Ts>
void LCurveHPT<MODEL,LOSS,SPLIT,CV,OPT,O>::Split( const T& trainset,
                                                  const T& testset,
                                                  const Ts&... args )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(trainset.inputs_.n_cols), 
        "There are not enough data for learning curve generation!" );
  BOOST_ASSERT_MSG( int(trainset.labels_.n_rows) == int(1) &&
                    int(testset.labels_.n_rows) == int(1), 
                    "Only 1D outputs are allowed!" );
  LOSS loss;

  ProgressBar pb("LCurveHPT.Split", Ns_.n_elem*repeat_);

  #pragma omp parallel for collapse(2) if(parallel_)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = split_(trainset.inputs_,trainset.labels_,size_t(Ns_(i)));

      arma::Mat<O> Xtrn = std::get<0>(res);
      auto ytrn = std::get<2>(res);

      mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>>
                                                          hpt(cvp_, Xtrn, ytrn);
      hpt.Optimize(args...);
      MODEL model = std::move(hpt.BestModel());
      model.Train(Xtrn, ytrn);


      test_errors_(j,i) = static_cast<O>(loss.Evaluate(model, testset.inputs_,
                            testset.labels_));

      if (prog_)
        pb.Update();
    }
  }


}

} // namespace src
#endif
