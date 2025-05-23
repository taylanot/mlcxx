/**
 * @file dataset.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef DATASET_H
#define DATASET_H

namespace data {

//=============================================================================
// Dataset
//=============================================================================
template<class LABEL=arma::Row<DTYPE>,class T=DTYPE>
class Dataset 
{
public:
  size_t size_; // size of the dataset
  size_t dimension_;  // dimension of the dataset
  // If an optional value for number of classes for classification problems
  std::optional<size_t> num_class_; 
  std::optional<size_t> seed_; // seed for reproduction purposes

  arma::Mat<T> inputs_;
  LABEL labels_;

  /* Dataset Empty Constructor */
  Dataset ( ) { };

  /* Dataset initializer
   *
   * @param dim : dimension of the dataset
   *
   */
  Dataset ( const size_t dim, const size_t seed = SEED );

  /* Dataset initializer
   *
   * @param dim       : dimension of the dataset
   * @param num_class : number of classes
   *
   */
  Dataset ( const size_t dim, 
            const size_t num_class,
            const size_t seed = SEED );

  /* Dataset initializer
   *
   * @param inputs : input of the dataset
   * @param labels : output of the dataset
   *
   */
  Dataset ( const arma::Mat<T>& inputs,
            const LABEL& labels );

  /* Dataset update with the given inputs and labels
   *
   * @param inputs : input of the dataset
   * @param labels : output of the dataset
   *
   */
  void Update ( const arma::Mat<T>& inputs, const LABEL& labels ); 

  void Update ( const LABEL& labels ); 

  /* Generate linear data for regression with Gaussian noise assumption
   *
   * @param N         : number of samples
   * @param noise_std : standard deviation of the Gaussian noise
   *
   */
  void Linear ( const size_t N = 10, const T noise_std=T(1.));

  /* Generate sinusoidal data for regression with Gaussian noise assumption
   *
   * @param N         : number of samples
   * @param noise_std : standard deviation of the Gaussian noise
   *
   */
  void Sine ( const size_t N = 10, const T noise_std=T(1.));

  /* Generate banana dataset for classification
   *
   * @param N     : number of samples for each class
   * @param delta : distance between bananas
   *
   */
  void Banana ( const size_t N = 10, const T delta=0. );

  /* Generate dipping dataset for classification 
   * Loog, M., & Duin, R. P. W. (2012). The dipping phenomenon. 
   *
   * @param N           : number of samples for each class
   * @param r           : radius of the covering circle
   * @param noise_std   : noise of the circle
   *
   */
  void Dipping ( const size_t N, const T r=1, const T noise_std=0.1 );

  /* Generate dipping dataset for classification 
   * Loog, M., & Duin, R. P. W. (2012). The dipping phenomenon. 
   *
   * @param N     : number of samples for each class
   * @param means : means of the Gaussian blobs
   * @param stds  : standard deviations of the Gaussian blobs
   *
   */
  void Gaussian ( const size_t N,
                  const arma::Row<T>& means,
                  const arma::Row<T>& stds );

  /* Serliazation with cereal for the class. */
  template <class Archive>
  void serialize(Archive& ar) 
  {
    ar( CEREAL_NVP(size_),
        CEREAL_NVP(inputs_),
        CEREAL_NVP(labels_),
        CEREAL_NVP(num_class_),
        CEREAL_NVP(dimension_) );
  }

  void Save( const std::string& filename );

private:
  void _update_info ( );

};

namespace regression {

//=============================================================================
// Dataset
//=============================================================================
template<class T=DTYPE>
class Dataset 
{
public:
  size_t size_;
  size_t dimension_;

  arma::Mat<T> inputs_;
  arma::Row<T> labels_;

  Dataset ( ) { };

  Dataset ( const size_t& D,
            const size_t& N );

  Dataset ( const size_t& D,
            const size_t& N,
            const arma::Col<T> mean,
            const arma::Mat<T> cov );

  void Set ( const size_t& D,
             const size_t& N );

  void Update ( const arma::Mat<T>& input, const arma::Mat<T>& labels  ); 

  void Outlier_ ( const size_t& Nout );

  void Generate ( const double& scale,
                  const double& phase,
                  const std::string& type ) ;

  void Generate ( const double& scale,
                  const double& phase,
                  const std::string& type,
                  const double& noise_std );

  void Generate ( const std::string& type,
                  const double& noise_std );

  void Generate ( const arma::Row<T>& type,
                  const double& noise_std );


  void Generate ( const std::string& type );

  void Noise ( const double& noise_std );

  void Save ( const std::string& filename );

  void Load ( const std::string& filename,
              const size_t& Din,
              const size_t& Dout,
              const bool transpose = true,
              const bool count = false );

  void Load ( const std::string& filename,
              const arma::uvec& ins,
              const arma::uvec& outs, 
              const bool transpose = true,
              const bool count = false );

  arma::Row<T> GetLabels( const size_t id );

  /* Serliazation with cereal for the class. */
  template <class Archive>
  void serialize(Archive& ar) 
  {
    ar( CEREAL_NVP(size_),
        CEREAL_NVP(inputs_),
        CEREAL_NVP(labels_),
        CEREAL_NVP(dimension_) );
  }


private:
  arma::Col<T> mean_;
  arma::Mat<T> cov_;

  void _update_info ( );

};

} // regression 
  
namespace functional
{

template<class T=DTYPE>
struct Dataset 
{
  size_t size_;
  size_t dimension_;
  size_t nfuncs_;
  arma::Mat<T> weights_;

  arma::Mat<T> inputs_;
  arma::Mat<T> labels_;

  Dataset ( ) { };
  Dataset ( const size_t& D,
            const size_t& N,
            const size_t& M );

  void Generate ( const std::string& type );

  void Generate ( const std::string& type,
                  const double& noise_std );

  void Generate ( const std::string& type,
                  const arma::Row<T>& noise_std );

  void Noise ( const double& noise_std );

  void Noise ( const arma::Row<T>& noise_std );

  void Save ( const std::string& filename );

  void Load ( const std::string& filename,
              const size_t& Din,
              const size_t& Dout, 
              const bool& transpose = true,
              const bool& count = false );

  void Normalize ( );

  void UnNormalize ( );

};

//=============================================================================
// SineGen
//=============================================================================
template<class T=DTYPE>
struct SineGen
{
  size_t size_;
  size_t dimension_;

  arma::Row<T> a_;
  arma::Row<T> p_;

  SineGen ( ) { };
  SineGen ( const size_t& M );

  arma::Mat<T> Predict ( const arma::Mat<T>& inputs,
                         const std::string& type  = "Phase" ) const;

  arma::Mat<T> Predict ( const arma::Mat<T>& inputs,
                         const std::string& type,
                         const double& eps ) const;

  arma::Mat<T> Mean ( const arma::Mat<T>& inputs,
                      const std::string& type = "Phase" ) const;

  arma::Mat<T> Mean ( const arma::Mat<T>& inputs,
                      const std::string& type,
                      const double& eps ) const;

  size_t GetM ( );

};

} //functional

namespace classification {

//=============================================================================
// Dataset
//=============================================================================
template<class T=DTYPE>
struct Dataset 
{
  size_t size_;
  size_t dimension_;
  size_t num_class_;

  arma::Mat<T> inputs_;
  arma::Row<size_t> labels_;

  Dataset ( ) { } ;

  Dataset ( const arma::Mat<T>& inputs,
            const arma::Row<size_t>& labels );

  Dataset ( const size_t& D,
            const size_t& N,
            const size_t& Nc );

  void Set ( const size_t& D,
             const size_t& N,
             const size_t& Nc );


  void Generate ( const std::string& type );

  void _banana ( const double& delta );

  void _dipping ( const double& r,
                  const double& noise_std );

  void _2classgauss ( const arma::Col<T>& mean1,
                      const arma::Col<T>& mean2,
                      const double& eps,
                      const double& delta );

  void _imbalance2classgauss ( const double& perc );

  void Save ( const std::string& filename );

  void Load ( const std::string& filename,
              const bool last = true,
              const bool transpose = true ,
              const bool count = false );

  /* Serliazation with cereal for the class. */
  template <class Archive>
  void serialize(Archive& ar) 
  {
    ar( CEREAL_NVP(size_),
        CEREAL_NVP(num_class_),
        CEREAL_NVP(inputs_),
        CEREAL_NVP(labels_),
        CEREAL_NVP(dimension_) );
  }

};

} // namespace classification

namespace oml {

//=============================================================================
// Dataset
//=============================================================================
template<class LTYPE=size_t, class T=DTYPE>
class Dataset 
{
public:
  size_t id_;
  size_t size_;
  size_t dimension_;
  std::optional<size_t> num_class_;
  std::filesystem::path path_; 

  arma::Mat<T> inputs_;
  arma::Row<LTYPE> labels_;

  Dataset ( ) { } ;

  Dataset ( const size_t& id, const std::filesystem::path& path );

  Dataset ( const size_t& id );

  void Update ( const arma::Mat<T>& input, const arma::Row<LTYPE>& labels  ); 

  /* Serliazation with cereal for the class. */
  template <class Archive>
  void serialize(Archive& ar) 
  {
    ar( CEREAL_NVP(size_),
        CEREAL_NVP(id_),
        CEREAL_NVP(path_.string()),
        CEREAL_NVP(num_class_),
        CEREAL_NVP(meta_url_),
        CEREAL_NVP(down_url_),
        CEREAL_NVP(file_),
        CEREAL_NVP(metafile_),
        CEREAL_NVP(inputs_),
        CEREAL_NVP(labels_),
        CEREAL_NVP(dimension_) );
  }

  /* Save the object to a BinaryFile
   *
   * @param filename : binary file name
   */
  void Save ( const std::string& filename );
  static std::shared_ptr<Dataset<LTYPE,T>> Load ( const std::string& filename );

private:

  std::filesystem::path filepath_ = path_ / "datasets";
  std::filesystem::path metapath_ = path_ / "meta";

  bool _download (  ); 
  void _update_info(  ); 

  void _load ( );

  bool _iscateg(const arma::Row<T>& row); 
  arma::Row<size_t> _convcateg(const arma::Row<T>& row);
  arma::Row<size_t> _procrow(const arma::Row<T>& row);

  std::string _gettargetname (const std::string& metadata ); 
  std::string _getdownurl (const std::string& metadata ); 

  int _findlabel ( const std::string& targetname ); 

  std::string _fetchmetadata( ); 
  std::string _readmetadata( ); 

  std::string meta_url_;
  std::string down_url_;

  std::string file_;
  std::string metafile_;



};

} // namesapce oml


} // namespace data


#include "dataset_impl.h"

#endif
