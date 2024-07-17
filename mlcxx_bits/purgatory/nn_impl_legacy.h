/**
 * @file nn_impl.h
 * @author Ozgur Taylan Turan
 *
 * Simple Neural Network Wrapper
 *
 */
#ifndef NN_IMPL_H
#define NN_IMPL_H

namespace algo { 
namespace regression {

//=============================================================================
// ANN 
//=============================================================================
template<class Optimizer,class T>
template<class... Args>
ANN<Optimizer,T>::ANN ( const arma::Row<size_t>& layer_info,
                        const size_t& nonlintype,
                        const bool& early,
                        const Args&... args ) 
                        : layer_info_(layer_info), nonlintype_(nonlintype),
                          archtype_(0), early_(early), opt_(args...)
{
  arma::Row<size_t>::iterator it = layer_info_.begin();
  arma::Row<size_t>::iterator it_end = layer_info_.end();

  for (; it!=it_end-1; it++)
  {
    model_.Add<mlpack::LinearType<arma::Mat<T>>>(size_t(*it));
    switch (nonlintype_)
    {
      case 1:
        model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
        break;
      case 2:
        model_.Add<mlpack::HardSigmoidType<arma::Mat<T>>>();
        break;
      case 3:
        model_.Add<mlpack::BaseLayer<mlpack::TanhFunction,arma::Mat<T>>>();
        break;
      case 4:
        model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::Mat<T>>>();
        break;
      default:
        model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
    }
  }
  model_.Add<mlpack::LinearType<arma::Mat<T>>>(size_t(*it));
}

template<class Optimizer,class T>
template<class... Args>
ANN<Optimizer,T>::ANN ( const arma::Mat<T>& inputs,
                        const arma::Mat<T>& labels,
                        const arma::Row<size_t>& layer_info,
                        const size_t& nonlintype,
                        const bool& early,
                        const Args&... args ) 
                      : ANN<Optimizer>::ANN( layer_info, nonlintype, early, args...) 
{
  Train(inputs, labels);
}


template<class Optimizer,class T>
template<class... Args>
ANN<Optimizer,T>::ANN ( const arma::Mat<T>& inputs,
                        const arma::Mat<T>& labels,
                        const size_t& archtype,
                        const size_t& nonlintype,
                        const bool& early,
                        const Args&... args ) 
                    : layer_info_( ), nonlintype_(nonlintype),
                      archtype_(archtype), early_(early), opt_(args...)
{
  switch (archtype_)
  {
    case 1:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(labels.n_rows);
      break;
    case 2:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(10);
      switch (nonlintype_)
      {
        case 1:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
          break;
        case 2:
          model_.Add<mlpack::HardSigmoidType<arma::Mat<T>>>();
          break;
        case 3:
          model_.template Add<mlpack::BaseLayer<mlpack::TanhFunction,arma::Mat<T>>>();
          break;
        case 4:
          model_.template Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::Mat<T>>>();
          break;
        default:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
      }
      break;
    case 3:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(10);
      switch (nonlintype_)
      {
        case 1:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
          break;
        case 2:
          model_.Add<mlpack::HardSigmoidType<arma::Mat<T>>>();
          break;
        case 3:
          model_.Add<mlpack::BaseLayer<mlpack::TanhFunction,arma::Mat<T>>>();
          break;
        case 4:
          model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::Mat<T>>>();
          break;
        default:
          model_.Add<mlpack::ReLUType<arma::Mat<T>>>();
      }
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(labels.n_rows);
      break;
    default:
      model_.Add<mlpack::LinearType<arma::Mat<T>>>(labels.n_rows);
  }
  Train(inputs, labels);
}

template<class Optimizer,class T>
void ANN<Optimizer,T>::Train ( const arma::Mat<T>& inputs,
                               const arma::Mat<T>& labels )
{
  if (!early_)
    model_.Train(inputs, labels, opt_);
  else
  {
    arma::Mat<T> inp,lab,val_inp,val_lab;

    // If there is not enough data for the split train-test is the same
    mlpack::data::Split(inputs,labels,
                        inp,val_inp,lab,val_lab,potetialsplitratio_);

    auto func = [&](const arma::Mat<T>& inputs )
    {
      arma::mat pred;
      model_.Predict(val_inp, pred);
      return mlpack::SquaredEuclideanDistance::Evaluate(pred, val_lab)
                                            /pred.n_cols;
    };

    opt_.StepSize() = 0.001;
    ens::EarlyStopAtMinLossType<arma::Mat<T>> stop(func, ptc_);
    /* ens::EarlyStopAtMinLossType<arma::Mat<T>> stop(ptc_); */
    model_.Train(inp,lab,opt_,stop);
  }
}

template<class Optimizer,class T>
const arma::Mat<T>& ANN<Optimizer,T>::Parameters ( ) const
{
  return model_.Parameters();
}

template<class Optimizer,class T>
arma::Mat<T>& ANN<Optimizer,T>::Parameters ( ) 
{
  return model_.Parameters();
}


template<class Optimizer,class T>
void ANN<Optimizer,T>::Predict ( const arma::Mat<T>& inputs,
                                 arma::Mat<T>& labels )
{
  model_.Predict(inputs, labels);
}

template<class Optimizer,class T>
T ANN<Optimizer,T>::ComputeError( const arma::Mat<T>& inputs,
                                  const arma::Mat<T>& labels ) 
{
  arma::Mat<T> temp;
  Predict(inputs, temp);
  const T n_points = inputs.n_cols;
  return mlpack::SquaredEuclideanDistance::Evaluate(temp, labels)
                                                             / n_points;
}

template<class Optimizer,class T>
void ANN<Optimizer,T>::StepSize ( const T& lr )
{
  opt_.StepSize() = lr;
}

template<class Optimizer,class T>
void ANN<Optimizer,T>::Patience ( const size_t& ptc )
{
  ptc_ = ptc;
}

template<class Optimizer,class T>
void ANN<Optimizer,T>::BatchSize ( const size_t& bs )
{
  opt_.BatchSize() = bs;
}                            

template<class Optimizer,class T>
void ANN<Optimizer,T>::MaxIterations ( const size_t& epochs )
{
  opt_.MaxIterations() = epochs;
}                            

template<class Optimizer,class T>
void ANN<Optimizer,T>::Save ( const std::string& filename ) 
{
  mlpack::data::Save(filename, "model_", model_);
}

template<class Optimizer,class T>
void ANN<Optimizer,T>::Load ( const std::string& filename ) 
{
  mlpack::data::Load(filename, "model_", model_);
}

/* //============================================================================= */
/* // NN */
/* //============================================================================= */
/* template<class Optimizer> */
/* template<class... Args> */
/* NN<Optimizer>::NN ( const arma::Row<size_t>& layer_info, */
/*                     const Args&... args ) */ 
/*                     : layer_info_(layer_info), opt_(args...) */
/* { } */
/* template<class Optimizer> */
/* template<class... Args> */
/* NN<Optimizer>::NN ( const arma::mat& inputs, */
/*                     const arma::mat& labels, */
/*                     const arma::Row<size_t>& layer_info, */
/*                     const Args&... args ) */ 
/*                     : layer_info_(layer_info), opt_(args...) */
/* { */

/*   Train(inputs, labels); */
/* } */

/* template<class Optimizer> */
/* template<class... Args> */
/* NN<Optimizer>::NN ( const arma::mat& inputs, */
/*                     const arma::rowvec& labels, */
/*                     const arma::Row<size_t>& layer_info, */
/*                     const Args&... args ) */ 
/*                     : layer_info_(layer_info), opt_(args...) */
/* { */
/*   arma::mat labs = arma::conv_to<arma::mat>::from(labels); */
/*   Train(inputs, labs); */
/* } */
/* template<class Optimizer> */
/* void NN<Optimizer>::Train ( const arma::mat& inputs, */
/*                             const arma::mat& labels ) */
/* { */
/*   model_.Add<mlpack::Linear>(inputs.n_rows); */
/*   model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::mat>>(); */

/*   arma::Row<size_t>::iterator it = layer_info_.begin(); */
/*   arma::Row<size_t>::iterator it_end = layer_info_.end(); */

/*   for (; it!=it_end; it++) */
/*   { */
/*     model_.Add<mlpack::Linear>(size_t(*it)); */
/*     model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::mat>>(); */
/*   } */


/*   model_.Add<mlpack::Linear>(labels.n_rows); */
/*   model_.Train(inputs, labels, opt_); */
/* } */

/* template<class Optimizer> */
/* void NN<Optimizer>::Predict ( const arma::mat& inputs, */
/*                               arma::mat& labels ) */
/* { */
/*   model_.Predict(inputs, labels); */
/* } */

/* template<class Optimizer> */
/* double NN<Optimizer>::ComputeError( const arma::mat& inputs, */
/*                                     const arma::mat& labels ) */ 
/* { */
/*   arma::mat temp; */
/*   Predict(inputs, temp); */
/*   const size_t n_points = inputs.n_cols; */

/*   return mlpack::SquaredEuclideanDistance::Evaluate(temp, labels) */
/*                                                              / n_points; */
/* } */

/* template<class Optimizer> */
/* void NN<Optimizer>::StepSize ( const double& lr ) */
/* { */
/*   opt_.StepSize() = lr; */
/* } */

/* template<class Optimizer> */
/* void NN<Optimizer>::BatchSize ( const size_t& bs ) */
/* { */
/*   opt_.BatchSize() = bs; */
/* } */                            

/* template<class Optimizer> */
/* void NN<Optimizer>::MaxIterations ( const size_t& epochs ) */
/* { */
/*   opt_.MaxIterations() = epochs; */
/* } */                            

/* template<class Optimizer> */
/* void NN<Optimizer>::Save ( const std::string& filename ) */ 
/* { */
/*   mlpack::data::Save(filename, "model_", model_); */
/* } */

/* template<class Optimizer> */
/* void NN<Optimizer>::Load ( const std::string& filename ) */ 
/* { */
/*   mlpack::data::Load(filename, "model_", model_); */
/* } */
} // regression
} // algo

#endif 
