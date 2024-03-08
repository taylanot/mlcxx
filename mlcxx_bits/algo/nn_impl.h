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
template<class Optimizer>
template<class... Args>
ANN<Optimizer>::ANN ( const arma::Row<size_t>& layer_info,
                      const size_t& nonlintype,
                      const Args&... args ) 
                      : layer_info_(layer_info), nonlintype_(nonlintype),
                        archtype_(0), opt_(args...)
{

  arma::Row<size_t>::iterator it = layer_info_.begin();
  arma::Row<size_t>::iterator it_end = layer_info_.end();

  for (; it!=it_end-1; it++)
  {
    model_.Add<mlpack::Linear>(size_t(*it));
    switch (nonlintype_)
    {
      case 1:
        model_.Add<mlpack::ReLU>();
        break;
      case 2:
        model_.Add<mlpack::HardSigmoid>();
        break;
      case 3:
        model_.Add<mlpack::BaseLayer<mlpack::TanhFunction,arma::mat>>();
        break;
      case 4:
        model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::mat>>();
        break;
      default:
        model_.Add<mlpack::ReLU>();
    }
  }
  model_.Add<mlpack::Linear>(size_t(*it));
}

template<class Optimizer>
template<class... Args>
ANN<Optimizer>::ANN ( const arma::mat& inputs,
                      const arma::mat& labels,
                      const arma::Row<size_t>& layer_info,
                      const size_t& nonlintype,
                      const Args&... args ) 
                      : ANN<Optimizer>::ANN( layer_info, nonlintype, args...) 
{
  Train(inputs, labels);
}


template<class Optimizer>
template<class... Args>
ANN<Optimizer>::ANN ( const arma::mat& inputs,
                    const arma::mat& labels,
                    const size_t& archtype,
                    const size_t& nonlintype,
                    const Args&... args ) 
                    : layer_info_( ), nonlintype_(nonlintype),
                      archtype_(archtype), opt_(args...)
{
  switch (archtype_)
  {
    case 1:
      model_.Add<mlpack::Linear>(inputs.n_rows);
      break;
    case 2:
      model_.Add<mlpack::Linear>(inputs.n_rows);
      switch (nonlintype_)
      {
        case 1:
          model_.Add<mlpack::ReLU>();
          break;
        case 2:
          model_.Add<mlpack::HardSigmoid>();
          break;
        case 3:
          model_.Add<mlpack::BaseLayer<mlpack::TanhFunction,arma::mat>>();
          break;
        case 4:
          model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::mat>>();
          break;
        default:
          model_.Add<mlpack::ReLU>();
      }
      break;
    case 3:
      model_.Add<mlpack::Linear>(inputs.n_rows);
      switch (nonlintype_)
      {
        case 1:
          model_.Add<mlpack::ReLU>();
          break;
        case 2:
          model_.Add<mlpack::HardSigmoid>();
          break;
        case 3:
          model_.Add<mlpack::BaseLayer<mlpack::TanhFunction,arma::mat>>();
          break;
        case 4:
          model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::mat>>();
          break;
        default:
          model_.Add<mlpack::ReLU>();
      }
      model_.Add<mlpack::Linear>(labels.n_rows);
      break;
    default:
      model_.Add<mlpack::Linear>(inputs.n_rows);
  }
  Train(inputs, labels);
}

template<class Optimizer>
void ANN<Optimizer>::Train ( const arma::mat& inputs,
                             const arma::mat& labels )
{
  model_.Train(inputs, labels, opt_);
}

template<class Optimizer>
const arma::mat& ANN<Optimizer>::Parameters ( ) const
{
  return model_.Parameters();
}

template<class Optimizer>
arma::mat& ANN<Optimizer>::Parameters ( ) 
{
  return model_.Parameters();
}

template<class Optimizer>
void ANN<Optimizer>::TrainEarlyStop ( const arma::mat& inputs,
                                      const arma::mat& labels,
                                      const size_t& patience )
{
  //model_.Add<mlpack::Linear>(inputs.n_rows);
  //model_.Add<mlpack::ReLU>();

  //arma::Row<size_t>::iterator it = layer_info_.begin();
  //arma::Row<size_t>::iterator it_end = layer_info_.end();

  //for (; it!=it_end; it++)
  //{
  //  model_.Add<mlpack::Linear>(size_t(*it));
  //  model_.Add<mlpack::ReLU>();
  //}
  //model_.Add<mlpack::Linear>(labels.n_rows);

  //auto func = [&](const arma::mat& inputs )
  //{
  //  arma::mat pred;
  //  model_.Predict(val_inputs, pred);
  //  return mlpack::SquaredEuclideanDistance::Evaluate(pred, val_labels)
  //                                        /pred.n_cols;
  //};

  //ens::EarlyStopAtMinLoss stop(func, 20);

  //model_.Train(inputs, labels, opt_, stop);
}

template<class Optimizer>
void ANN<Optimizer>::Predict ( const arma::mat& inputs,
                              arma::mat& labels )
{
  model_.Predict(inputs, labels);
}

template<class Optimizer>
double ANN<Optimizer>::ComputeError( const arma::mat& inputs,
                                    const arma::mat& labels ) 
{
  arma::mat temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  return mlpack::SquaredEuclideanDistance::Evaluate(temp, labels)
                                                             / n_points;
}

template<class Optimizer>
void ANN<Optimizer>::StepSize ( const double& lr )
{
  opt_.StepSize() = lr;
}

template<class Optimizer>
void ANN<Optimizer>::BatchSize ( const size_t& bs )
{
  opt_.BatchSize() = bs;
}                            

template<class Optimizer>
void ANN<Optimizer>::MaxIterations ( const size_t& epochs )
{
  opt_.MaxIterations() = epochs;
}                            

template<class Optimizer>
void ANN<Optimizer>::Save ( const std::string& filename ) 
{
  mlpack::data::Save(filename, "model_", model_);
}

template<class Optimizer>
void ANN<Optimizer>::Load ( const std::string& filename ) 
{
  mlpack::data::Load(filename, "model_", model_);
}

//=============================================================================
// NN
//=============================================================================
template<class Optimizer>
template<class... Args>
NN<Optimizer>::NN ( const arma::Row<size_t>& layer_info,
                    const Args&... args ) 
                    : layer_info_(layer_info), opt_(args...)
{ }
template<class Optimizer>
template<class... Args>
NN<Optimizer>::NN ( const arma::mat& inputs,
                    const arma::mat& labels,
                    const arma::Row<size_t>& layer_info,
                    const Args&... args ) 
                    : layer_info_(layer_info), opt_(args...)
{

  Train(inputs, labels);
}

template<class Optimizer>
template<class... Args>
NN<Optimizer>::NN ( const arma::mat& inputs,
                    const arma::rowvec& labels,
                    const arma::Row<size_t>& layer_info,
                    const Args&... args ) 
                    : layer_info_(layer_info), opt_(args...)
{
  arma::mat labs = arma::conv_to<arma::mat>::from(labels);
  Train(inputs, labs);
}
template<class Optimizer>
void NN<Optimizer>::Train ( const arma::mat& inputs,
                            const arma::mat& labels )
{
  model_.Add<mlpack::Linear>(inputs.n_rows);
  model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::mat>>();

  arma::Row<size_t>::iterator it = layer_info_.begin();
  arma::Row<size_t>::iterator it_end = layer_info_.end();

  for (; it!=it_end; it++)
  {
    model_.Add<mlpack::Linear>(size_t(*it));
    model_.Add<mlpack::BaseLayer<mlpack::SoftsignFunction,arma::mat>>();
  }


  model_.Add<mlpack::Linear>(labels.n_rows);
  model_.Train(inputs, labels, opt_);
}

template<class Optimizer>
void NN<Optimizer>::Predict ( const arma::mat& inputs,
                              arma::mat& labels )
{
  model_.Predict(inputs, labels);
}

template<class Optimizer>
double NN<Optimizer>::ComputeError( const arma::mat& inputs,
                                    const arma::mat& labels ) 
{
  arma::mat temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  return mlpack::SquaredEuclideanDistance::Evaluate(temp, labels)
                                                             / n_points;
}

template<class Optimizer>
void NN<Optimizer>::StepSize ( const double& lr )
{
  opt_.StepSize() = lr;
}

template<class Optimizer>
void NN<Optimizer>::BatchSize ( const size_t& bs )
{
  opt_.BatchSize() = bs;
}                            

template<class Optimizer>
void NN<Optimizer>::MaxIterations ( const size_t& epochs )
{
  opt_.MaxIterations() = epochs;
}                            

template<class Optimizer>
void NN<Optimizer>::Save ( const std::string& filename ) 
{
  mlpack::data::Save(filename, "model_", model_);
}

template<class Optimizer>
void NN<Optimizer>::Load ( const std::string& filename ) 
{
  mlpack::data::Load(filename, "model_", model_);
}
} // regression
} // algo

#endif 
