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
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
LCurveHPT<MODEL,LOSS,CV,OPT,O>::LCurveHPT ( const arma::irowvec& Ns,
                                            const size_t repeat,
                                            const CVP cvp,
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
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
template<class T, class... Ts>
void LCurveHPT<MODEL,LOSS,CV,OPT,O>::Bootstrap ( const T& dataset,
                                                 const Ts&... args )
{
  Bootstrap(dataset.inputs_,dataset.labels_, args...);
}

template<class MODEL,
         class LOSS,
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
template<class T, class... Ts>
void LCurveHPT<MODEL,LOSS,CV,OPT,O>::Bootstrap ( const arma::Mat<O>& inputs,
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
      const auto idx = arma::randi<arma::uvec>(Ns_[i],
                                arma::distr_param(0,labels.n_elem-1));

      const arma::Mat<O> inps = inputs.cols(idx);
      const T  labs = labels.cols(idx);
      /* mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>> hpt(cvp_, */
      /*     inps, labs); */

      auto hpt = _GetHpt(inps,labs);
      hpt.Optimize(args...);
      MODEL model = std::move(hpt.BestModel());
      model.Train(inps,labs);

      test_errors_(j,i) = loss.Evaluate(model,inputs,labels);
      
      if (prog_)
        pb.Update();
    }
  }
}
//=============================================================================
// LCurve::RandomSet
//=============================================================================     
template<class MODEL,
         class LOSS,
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
template<class SPLIT, class T, class... Ts>
void LCurveHPT<MODEL,LOSS,CV,OPT,O>::RandomSet ( const arma::Mat<O>& inputs,
                                                 const T& labels,
                                                 const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(labels.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );
  LOSS loss;

  ProgressBar pb("LCurveHPT.RandomSet", Ns_.n_elem*repeat_);

  SPLIT split;
  #pragma omp parallel for collapse(2) if(parallel_)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = split(inputs, labels, size_t(Ns_(i)));
      arma::Mat<O> Xtrn = std::get<0>(res);
      arma::Mat<O> Xtst = std::get<1>(res);
      T ytrn = std::get<2>(res);
      T ytst = std::get<3>(res);

      /* mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>> */
      /*                                                     hpt(cvp_, Xtrn, ytrn); */

      auto hpt = _GetHpt(Xtrn,ytrn);
      hpt.Optimize(args...);
      MODEL model = std::move(hpt.BestModel());
      model.Train(Xtrn, ytrn);

      test_errors_(j,i) = loss.Evaluate(model,Xtst,ytst);
      
      if (prog_)
        pb.Update();
    }
  }
}
template<class MODEL,
         class LOSS,
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
template<class SPLIT, class T, class... Ts>
void LCurveHPT<MODEL,LOSS,CV,OPT,O>::RandomSet ( const T& dataset,
                                                 const Ts&... args )
{
  RandomSet(dataset.inputs_,dataset.labels_,args...);
}

//=============================================================================
// LCurve::Additive
//=============================================================================     
template<class MODEL,
         class LOSS,
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
template<class SPLIT, class T, class... Ts>
void LCurveHPT<MODEL,LOSS,CV,OPT,O>::Additive ( const arma::Mat<O>& inputs,
                                                const T& labels,
                                                const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(labels.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );
  LOSS loss;

  ProgressBar pb("LCurveHPT.Additive", Ns_.n_elem*repeat_);

  SPLIT split;
  #pragma omp parallel for if(parallel_)
  for(size_t j=0; j < size_t(repeat_); j++)
  {
    const auto res = split(inputs,labels,size_t(Ns_(0)));

    arma::Mat<O> Xtrn = std::get<0>(res);
    arma::Mat<O> Xrest = std::get<1>(res);

    T ytrn = std::get<2>(res);
    T yrest = std::get<3>(res);

    
    /* mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>> */
    /*                                                       hpt(cvp_, Xtrn, ytrn); */

    auto hpt = _GetHpt(Xtrn,ytrn);
    hpt.Optimize(args...);
    MODEL model = hpt.BestModel();
    model.Train(Xtrn, ytrn);
    test_errors_(j,0) = loss.Evaluate(model,Xrest,yrest);

    for (size_t i=1; i < size_t(Ns_.n_elem) ; i++)
    {
      data::Migrate(Xtrn,ytrn,Xrest,yrest, Ns_[i]-Ns_[i-1]);

      /* mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>> */
      /*                                              hpt(cvp_, Xtrn, ytrn); */
      auto hpt = _GetHpt(Xtrn,ytrn);
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
template<class MODEL,
         class LOSS,
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
template<class SPLIT, class T, class... Ts>
void LCurveHPT<MODEL,LOSS,CV,OPT,O>::Additive ( const T& dataset,
                                                const Ts&... args )
{
  Additive(dataset.inputs_, dataset.labels_, args...);
}
//=============================================================================
// LCurve::Split
//=============================================================================     
template<class MODEL,
         class LOSS,
         template<typename, typename, typename, typename, typename> class CV,
         class OPT,
         class O>
template<class SPLIT, class T, class... Ts>
void LCurveHPT<MODEL,LOSS,CV,OPT,O>::Split( const T& trainset,
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

  SPLIT split;
  #pragma omp parallel for collapse(2) if(parallel_)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = split(trainset.inputs_,trainset.labels_,size_t(Ns_(i)));

      const arma::Mat<O> Xtrn = std::get<0>(res);
      const auto ytrn = std::get<2>(res);

      /* mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>> */
      /*                                                     hpt(cvp_, Xtrn, ytrn); */
      auto hpt = _GetHpt(Xtrn,ytrn);
      hpt.Optimize(args...);
      MODEL model = std::move(hpt.BestModel());
      model.Train(Xtrn, ytrn);

      test_errors_(j,i) = loss.Evaluate(model,testset.inputs_,testset.labels_);
      if (prog_)
        pb.Update();
    }
  }

}

template<class MODEL,
         class LOSS,
         template<class, class, class, class, class> class CV,
         class OPT,
         class O>
template<class T>
mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>> 
 LCurveHPT<MODEL,LOSS,CV,OPT,O>::_GetHpt ( const arma::Mat<DTYPE>& Xtrn,
                                           const T& ytrn )
{
  if constexpr (!mlpack::MetaInfoExtractor<MODEL>::TakesNumClasses)
    return mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>>
      (cvp_, Xtrn, ytrn);
  else
    return mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>>
      (cvp_, Xtrn, ytrn,size_t(arma::unique(ytrn).eval().n_elem));
}

} // namespace src
#endif
