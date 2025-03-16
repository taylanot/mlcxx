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
                               const bool prog,
                               const std::string type,
                               const std::string name ) :

repeat_(repeat), Ns_(Ns), parallel_(parallel),
  prog_(prog), type_(type), name_(name)
{
  RegisterSignalHandler( );
  globalSafeFailFunc = [this]() { this->CleanUp(); };
  test_errors_.resize(repeat_,Ns_.n_elem);
}

//=============================================================================
// LCurve::Bootstrap
//=============================================================================     
template<class MODEL,
         class LOSS,class O>
template<class T, class... Ts>
void LCurve<MODEL,LOSS,O>::Bootstrap ( const T& dataset,
                                       const Ts&... args )
{
  Bootstrap(dataset.inputs_,dataset.labels_,args...);
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
  // With this version you cannot use the split version, fyi....
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(labels.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );

  ProgressBar pb("LCurve.Bootstrap", Ns_.n_elem*repeat_);

  #pragma omp parallel for collapse(2) if(parallel_)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto idx = arma::randi<arma::uvec>(Ns_[i],
                                arma::distr_param(0,labels.n_elem-1));
      const arma::Mat<O> inps = inputs.cols(idx); 
      const T labs = labels.cols(idx); 
      MODEL model(inps,labs,args...);

      test_errors_(j,i) = loss_.Evaluate(model,inputs,
                                               labels);
      if (prog_)
        pb.Update();
    }

}
//=============================================================================
// LCurve::RandomSet
//=============================================================================     
template<class MODEL,
         class LOSS,class O>
template<class SPLIT,class T, class... Ts>
void LCurve<MODEL,LOSS,O>::RandomSet ( const arma::Mat<O>& inputs,
                                       const T& labels,
                                       const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(labels.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );

  ProgressBar pb("LCurve.RandomSet", Ns_.n_elem*repeat_);

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

      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = loss_.Evaluate(model, Xtst, ytst);
      if (prog_)
        pb.Update();
    }
  }

}

template<class MODEL,
         class LOSS,class O>
template<class SPLIT,class T,class... Ts>
void LCurve<MODEL,LOSS,O>::RandomSet( const T& dataset,
                                      const Ts&... args )
{
  RandomSet(dataset.inputs_,dataset.labels_,args...);
}

//=============================================================================
// LCurve::Additive
//=============================================================================     
template<class MODEL,
         class LOSS,class O>
template<class SPLIT,class T,class... Ts>
void LCurve<MODEL,LOSS,O>::Additive ( const arma::Mat<O>& inputs,
                                            const T& labels,
                                            const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );
  BOOST_ASSERT_MSG( int(labels.n_rows) == int(1), 
        "Only 1D outputs are allowed!" );

  ProgressBar pb("LCurve.Additive", Ns_.n_elem*repeat_);

  SPLIT split;
  #pragma omp parallel for if(parallel_)
  for(size_t j=0; j < size_t(repeat_); j++)
  {
    const auto res = split(inputs,labels,size_t(Ns_(0)));

    arma::Mat<O> Xtrn = std::get<0>(res);
    arma::Mat<O> Xrest = std::get<1>(res);

    T ytrn = std::get<2>(res);
    T yrest = std::get<3>(res);

    MODEL model(Xtrn, ytrn, args...);
    test_errors_(j,0) = loss_.Evaluate(model,Xrest,yrest);
    for (size_t i=1; i < size_t(Ns_.n_elem) ; i++)
    {
      data::Migrate(Xtrn,ytrn,Xrest,yrest, Ns_[i]-Ns_[i-1]);
      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = loss_.Evaluate(model,Xrest,yrest);
      if (prog_)
        pb.Update();
    }
    if (prog_)
      pb.Update();
  }

}

template<class MODEL,
         class LOSS,class O>
template<class SPLIT,class T,class... Ts>
void LCurve<MODEL,LOSS,O>::Additive ( const T& dataset,
                                      const Ts&... args )
{
  Additive(dataset.inputs_, dataset.labels_,args...);
}

//=============================================================================
// LCurve::Split
//=============================================================================     
template<class MODEL,
         class LOSS,class O>
template<class SPLIT,class T,class... Ts>
void LCurve<MODEL,LOSS,O>::Split( const T& trainset,
                                  const T& testset,
                                  const Ts&... args )
{
  BOOST_ASSERT_MSG( int(Ns_.max()) < int(trainset.inputs_.n_cols), 
        "There are not enough data for learning curve generation!" );
  BOOST_ASSERT_MSG( int(trainset.labels_.n_rows) == int(1) &&
                    int(testset.labels_.n_rows) == int(1), 
                    "Only 1D outputs are allowed!" );
  ProgressBar pb("LCurve.Split", Ns_.n_elem*repeat_);

  SPLIT split;
  #pragma omp parallel for collapse(2) if(parallel_)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = split(trainset.inputs_,trainset.labels_,size_t(Ns_(i)));

      arma::Mat<O> Xtrn = std::get<0>(res);
      auto ytrn = std::get<2>(res);

      MODEL model(Xtrn, ytrn, args...);
      /* test_errors_(j,i) = static_cast<O>(loss_.Evaluate(model, testset.inputs_, */
      /*                     testset.labels_)); */

      test_errors_(j,i) = loss_.Evaluate(model, testset.inputs_,testset.labels_);
      if (prog_)
        pb.Update();
    }
  }
}

//=============================================================================
// LCurve::CleanUp
//=============================================================================     
template<class MODEL,
         class LOSS,class O>
void LCurve<MODEL,LOSS,O>::CleanUp ( )
{
  LOG("\rCleanUp is called!"<<std::flush);
  Save(name_+".bin");
}

//=============================================================================
// LCurve::RegisterSignalHandler
//=============================================================================     
template<class MODEL,
         class LOSS,class O>
void LCurve<MODEL,LOSS,O>::RegisterSignalHandler( )
{
  signal(SIGALRM, LCurve<MODEL,LOSS,O>::SignalHandler);
}

//=============================================================================
// LCurve::SignalHandler
//=============================================================================     
template<class MODEL,
         class LOSS,class O>
void LCurve<MODEL,LOSS,O>::SignalHandler( int sig )
{
  if (globalSafeFailFunc) globalSafeFailFunc();  
  LOG("\rTime limit exceeded! Exiting..." << std::flush);
  std::quick_exit(0);
}

//=============================================================================
// LCurve::Save
//=============================================================================     
template<class MODEL,
         class LOSS,class O>
void LCurve<MODEL,LOSS,O>::Save ( const std::string& filename )
{
  std::ofstream file(filename, std::ios::binary);
  if (!file) 
    ERR("\rCannot open file for writing: " << filename << std::flush);

  cereal::BinaryOutputArchive archive(file);
  archive(cereal::make_nvp("LCurve", *this));  // Serialize the current object
  LOG("\rLCurve object saved to " << filename << std::flush);
}

//=============================================================================
// LCurve::Load
//=============================================================================
template<class MODEL,
         class LOSS,class O>
std::shared_ptr<LCurve<MODEL,LOSS,O>> LCurve<MODEL,LOSS,O>::Load
                                              ( const std::string& filename )
{
  std::ifstream file(filename, std::ios::binary);
  if (!file) 
  {
    ERR("\rError: Cannot open file for reading: " << filename);
    return nullptr;
  }
  cereal::BinaryInputArchive archive(file);
  auto lcurve = std::make_shared<LCurve<MODEL,LOSS,O>>();
  archive(cereal::make_nvp("LCurve", *lcurve));  // Deserialize into a new object
  LOG("\rLCurve oaded from " << filename);
  return lcurve;
}

} // namespace src
#endif
