/**
 * @file data.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef DATA_H
#define DATA_H


#include "dataset.h"
#include "manip.h"
#include "sample.h"

namespace data {

//-----------------------------------------------------------------------------
// Transform
//-----------------------------------------------------------------------------
template<class T = mlpack::data::StandardScaler,
         class D = data::Dataset<arma::Row<DTYPE>>>
class Transformer
{
private:
  T inp_; // Transformer for inputs
  T lab_; // Transformer for labels
public:
  /**
   * @brief Construct and fit transformers for both inputs and labels.
   * @param data Dataset to fit the transformers on.
   */
  Transformer( const D& dataset )
  {
    inp_.Fit(arma::conv_to<arma::mat>::from(dataset.inputs_));
    if constexpr (std::is_same<D, data::Dataset<arma::Row<DTYPE>>>::value ||
        std::is_same<D, data::Dataset<arma::Mat<DTYPE>>>::value ||
        std::is_same<D, data::oml::Dataset<DTYPE>>::value )
      lab_.Fit(arma::conv_to<arma::rowvec>::from(dataset.labels_));
  };

  /**
   * @brief Transform only the inputs of a dataset.
   * @param data Dataset to transform.
   * @return Transformed dataset (inputs only).
   */
  D TransInp( const D& dataset )
  {
    D tdataset = dataset;
    inp_.Transform( dataset.inputs_, tdataset.inputs_);
    return tdataset;
  };

  /**
   * @brief Transform only the labels of a dataset.
   * @param data Dataset to transform.
   * @return Transformed dataset (labels only).
   */
  D TransLab ( const D& dataset )
  {
    if constexpr (std::is_same<D, data::Dataset<arma::Row<DTYPE>>>::value ||
        std::is_same<D, data::Dataset<arma::Mat<DTYPE>>>::value ||
        std::is_same<D, data::oml::Dataset<DTYPE>>::value )
    {
      WARNING("Unfortunately I will not let you transform the labels if you \
          donot have a regression dataset!");
      return dataset; 
    }
    else
    {
      D tdataset = dataset;
      lab_.Transform( dataset.labels_, tdataset.labels_);
      return tdataset;
    }
  };

  /**
   * @brief Transform both inputs and labels of a dataset.
   * @param data Dataset to transform.
   * @return Fully transformed dataset.
   */
  D Trans ( const D& dataset )
  {
    D tdataset = TransInp(dataset);
    if constexpr (std::is_same<D, data::Dataset<arma::Row<DTYPE>>>::value ||
        std::is_same<D, data::Dataset<arma::Mat<DTYPE>>>::value ||
        std::is_same<D, data::oml::Dataset<DTYPE>>::value )
      tdataset = TransLab(tdataset);
    return tdataset;
  };

  /**
   * @brief Inverse transform both inputs 
   * @param data Dataset to inverse transform.
   * @return Original-scale dataset.
   */

  D InvTransInp( const D& dataset )
  {
    D tdataset = dataset;
    inp_.InverseTransform( dataset.inputs_, tdataset.inputs_);
    return tdataset;
  };

  /**
   * @brief Inverse transform only the labels.
   * @param data Dataset to inverse transform.
   * @return Dataset with original-scale labels.
   */
  D InvTransLab ( const D& dataset )
  {
    D tdataset = dataset;
    lab_.InverseTransform( dataset.labels_, tdataset.labels_);
    return tdataset;
  };

  /**
   * @brief Inverse transform both inputs and labels.
   * @param data Dataset to inverse transform.
   * @return Original-scale dataset.
   */
  D InvTrans ( const D& dataset )
  {
    D tdataset = InvTransInp(dataset);
    if constexpr (std::is_same<D, data::Dataset<arma::Row<DTYPE>>>::value ||
        std::is_same<D, data::Dataset<arma::Mat<DTYPE>>>::value ||
        std::is_same<D, data::oml::Dataset<DTYPE>>::value )
      tdataset = InvTransLab(tdataset);
    return tdataset;
  };
};

//-----------------------------------------------------------------------------
// Gram
//-----------------------------------------------------------------------------
template<class KERNEL, class T = DTYPE>
struct Gram
{
    /// Default constructor.
    Gram() {}

    /**
     * @brief Construct and initialize kernel with arbitrary arguments.
     * @tparam Ts Argument types for the kernel constructor.
     * @param args Arguments to forward to the kernel constructor.
     */
    template<typename... Ts>
    Gram(Ts&&... args) : kernel_(args...) {}

    // Kernel instance used for computing Gram matrices.
    KERNEL kernel_;

    /**
     * @brief Compute Gram matrix for row-major ordered data.
     * @param input1 First dataset (rows are samples).
     * @param input2 Second dataset (rows are samples).
     * @return Gram matrix of size input1.n_rows × input2.n_rows.
     */
    arma::Mat<T> GetMatrix2(const arma::Mat<T>& input1,
                            const arma::Mat<T>& input2) const
    {
        arma::Mat<T> matrix(input1.n_rows, input2.n_rows);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < int(input1.n_rows); i++)
            for (int j = 0; j < int(input2.n_rows); j++)
                matrix(i,j) = kernel_.Evaluate(input1.row(i).eval(),
                                               input2.row(j).eval());

        return matrix;
    }

    /**
     * @brief Compute Gram matrix for column-major ordered data.
     * @param input1 First dataset (columns are samples).
     * @param input2 Second dataset (columns are samples).
     * @return Gram matrix of size input1.n_cols × input2.n_cols.
     */
    arma::Mat<T> GetMatrix(const arma::Mat<T>& input1,
                           const arma::Mat<T>& input2) const
    {
        arma::Mat<T> matrix(input1.n_cols, input2.n_cols);
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < int(input1.n_cols); i++)
            for (int j = 0; j < int(input2.n_cols); j++)
                matrix(i,j) = kernel_.Evaluate(input1.col(i).eval(),
                                               input2.col(j).eval());

        return matrix;
    }

    /**
     * @brief Compute an approximate Gram matrix using the Nyström method.
     * 
     * Selects k random landmark points, computes their kernel matrix W,
     * and uses it to approximate the full kernel matrix:
     * K_approx = C * pinv(W) * C^T
     * 
     * @param input1 First dataset (columns are samples).
     * @param input2 Second dataset (columns are samples).
     * @param k      Number of landmark points to sample.
     * @return Approximated Gram matrix.
     */
    arma::Mat<T> GetApprox(const arma::Mat<T>& input1,
                                  const arma::Mat<T>& input2,
                                  size_t k) const
    {
        size_t n_samples = input1.n_rows;
        arma::uvec indices = arma::randi<arma::uvec>(k, arma::distr_param(0, n_samples - 1));
        arma::Mat<T> landmarks = input1.cols(indices);

        arma::Mat<T> W = this->GetMatrix(landmarks, landmarks);
        arma::Mat<T> C = this->GetMatrix(input1, landmarks);
        arma::Mat<T> W_pinv = arma::pinv(W);

        return C * W_pinv * C.t();
    }

    /**
     * @brief Compute Gram matrix of a dataset with itself (column-major).
     * @param input1 Dataset (columns are samples).
     * @return Symmetric Gram matrix of size input1.n_cols × input1.n_cols.
     */
    arma::Mat<T> GetMatrix(const arma::Mat<T>& input1) const
    {
        return GetMatrix(input1, input1); 
    }
    /**
     * Serialize the model.
     */
    template<typename Archive>
    void serialize ( Archive& ar, const unsigned int )
    {
      ar ( cereal::make_nvp("kernel",kernel_) );
    }
};

//-----------------------------------------------------------------------------
// Report : This just summerizes some general information about the dataset
//-----------------------------------------------------------------------------
template<class Dataset,class O=DTYPE>
void report( const Dataset& dataset )
{
    PRINT("### DATASET INFORMATION ###");
    PRINT("features : " << dataset.dimension_ );
    PRINT("size : " << dataset.size_ );

    PRINT("### FEATURE INFORMATION ###");
    PRINT("Mean :  \n" << arma::mean(dataset.inputs_,1) );
    PRINT("Median :  \n" << arma::median(dataset.inputs_,1) );
    PRINT("Variance :  \n" << arma::var(dataset.inputs_.t()) );
    PRINT("Min :  \n" << arma::min(dataset.inputs_,1) );
    PRINT("Max :  \n" << arma::max(dataset.inputs_,1) );
    PRINT("Covariance : \n" << arma::cov(dataset.inputs_.t()) );

    PRINT("### LABEL INFORMATION ###");
    PRINT("Unique :  \n" << arma::unique(dataset.labels_) );
    PRINT("Counts :  \n" << arma::hist(dataset.labels_,arma::unique(dataset.labels_)) );
}

//-----------------------------------------------------------------------
// Load : wrapper for mlpack::Load
//-----------------------------------------------------------------------
template<class T, class O=DTYPE>
arma::Mat<O> Load ( const T& filename,
                    const bool& transpose,
                    const bool& count = false )
{
  arma::Mat<O> matrix;
  mlpack::data::DatasetInfo info;
  if ( count )
  {
    mlpack::data::Load(filename,matrix,info,true,transpose);
  }
  else
    mlpack::data::Load(filename,matrix,true,transpose);
  return matrix;
}
//-----------------------------------------------------------------------------
// Save : Easy saving to a file even with directory creation
//-----------------------------------------------------------------------------
template<class T>
void Save ( const std::filesystem::path& filename,
            const T& data,
            const bool transpose=true )
{
  T temp;

  if (transpose)
    temp = data.t();
  else
    temp = data;

  std::string ext = filename.extension();

  std::filesystem::create_directories(filename.parent_path());

  if (ext == "csv")
  {
    temp.save(filename,arma::csv_ascii);
  }
  else if (ext == "bin")
  {
    temp.save(filename,arma::arma_binary);
  }
  else if (ext == "arma")
  {
    temp.save(filename,arma::arma_ascii);
  }
  else if (ext == "txt")
  {
    temp.save(filename,arma::raw_ascii);
  }
  else
    throw std::runtime_error("Not Implemented save extension!");

}

};
#endif
