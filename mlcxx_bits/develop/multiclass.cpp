/**
 * @file multiclass.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to create a multi-class classifier 
 */

#include <headers.h>

template<class MODEL, class T=DTYPE>
class MultiClass
{
public:

  MultiClass (  ) = default;
  MultiClass ( std::string type ) : type_(type) { }
  template<class... Args>
  MultiClass ( const arma::Mat<T>& inputs, const arma::Row<size_t>& labels,
               const std::string type,
               const Args&... args ) : type_(type)
  {
    unq_ = arma::unique(labels).eval();
    nclass_ = unq_.n_elem;
    Train(inputs, labels, args...);
  }

  template<class... Args>
  MultiClass ( const arma::Mat<T>& inputs, const arma::Row<size_t>& labels,
               const Args&... args ) 
    : type_("SoftOnevsAll")
  { 
    unq_ = arma::unique(labels).eval();
    nclass_ = unq_.n_elem;
    Train(inputs, labels, args...);
  } 

  template<class... Args>
  void Train ( const arma::Mat<T>& inputs, const arma::Row<size_t>& labels,
               const Args&... args ) 
  { 
    if (type_ == "SoftOnevsAll" || type_ == "HardOnevsAll")
      OneVsAll(inputs, labels, args...);
    else
      PRINT_ERR("MultiClass::Not Implemented Yet!");
  } 

public:
  void Classify ( const arma::Mat<T>& inputs, arma::Row<size_t>& preds )
  {
    if (type_ == "HardOnevsAll")
    {
      arma::Mat<T> collect(nclass_,inputs.n_cols);
      /* #pragma omp parallel for */
      for(size_t i=0;i<nclass_;i++)
      {
        double add = (unq_(0)!=0) ? unq_(i) :unq_(i)+1;
        arma::Row<size_t> temp;
        models_[i].Classify(inputs,temp);
        collect.row(i) = arma::conv_to<arma::Row<T>>::from(temp*add);
      }
      preds = arma::conv_to<arma::Row<size_t>>::from(
                                            arma::round(arma::mean(collect,0)));
    }
    else if (type_ == "SoftOnevsAll")
    {
      arma::Mat<T> collect(nclass_,inputs.n_cols);
      /* #pragma omp parallel for */
      for(size_t i=0;i<nclass_;i++)
      {
        arma::Row<size_t> temp;
        arma::Mat<T> probs;
        models_[i].Classify(inputs,temp,probs);
        collect.row(i) = probs.row(1);
      }
      preds = unq_.cols(arma::index_max(collect,0));
    }
  }

  T ComputeError ( const arma::Mat<T>& inputs, const arma::Row<size_t>& labels )
  {
    arma::Row<size_t> predictions;
    Classify(inputs,predictions);
    arma::Row<size_t> temp =  predictions - labels; 
    return (arma::accu(temp != 0))/T(predictions.n_elem);
  }

  T ComputeAccuracy ( const arma::Mat<T>& inputs, const arma::Row<size_t>& labels )
  {
    return 1.-ComputeError(inputs,labels);
  }
private:

  template<class... Args>
  void OneVsAll ( const arma::Mat<T>& inputs, const arma::Row<size_t>& labels,
                  const Args&... args ) 
  { 
    models_.resize(unq_.n_elem);
    /* #pragma omp parallel for */
    for(size_t i=0;i<nclass_;i++)
    {
      auto binlabels = arma::conv_to<arma::Row<size_t>>::from(labels==unq_(i));
      MODEL model(inputs,binlabels,args...);
      models_[i] = model;
    }
  }

  std::vector<MODEL> models_;
  size_t nclass_;
  arma::Row<size_t> unq_;
  std::string type_;
};


int main() 
{
  utils::data::classification::Dataset dataset;
  dataset.Load("datasets/iris.csv",true,true);
  PRINT_VAR(dataset.labels_);

  MultiClass<mlpack::LogisticRegression<arma::mat>> model(dataset.inputs_,
                                                    dataset.labels_,1.);
  /* MultiClass<algo::classification::KernelSVM<>> model(dataset.inputs_, */
                                                          /* dataset.labels_,100.,0.01); */
  arma::Row<size_t> preds;
  /* model.Classify(dataset.inputs_,preds); */
  PRINT(model.ComputeAccuracy(dataset.inputs_,dataset.labels_));


  
  return 0;
}
