/**
 * @file dataset.h
 * @author Ozgur Taylan Turan
 *
 * Here lies the definitions of all the Dataset containers for regression,
 * classification, functional and openml related containers.
 */

#ifndef DATASET_H
#define DATASET_H

namespace data {
//-----------------------------------------------------------------------------
// data:: Dataset: This is a genearl Dataset container where you can decide at 
// compile time. A classification or regression dataset. 
//-----------------------------------------------------------------------------
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
  void Linear ( const size_t N=10, const T noise_std=T(1.));

  /* Generate sinusoidal data for regression with Gaussian noise assumption
   *
   * @param N         : number of samples
   * @param noise_std : standard deviation of the Gaussian noise
   *
   */
  void Sine ( const size_t N=10, const T noise_std=T(1.));

  /* Generate banana dataset for classification
   *
   * @param N     : number of samples for each class
   * @param delta : distance between bananas
   *
   */
  void Banana ( const size_t N=10, const T delta=0. );

  /* Generate dipping dataset for classification 
   * Loog, M., & Duin, R. P. W. (2012). The dipping phenomenon. 
   *
   * @param N           : number of samples for each class
   * @param r           : radius of the covering circle
   * @param noise_std   : noise of the circle
   *
   */
  void Dipping ( const size_t N=10, const T r=1, const T noise_std=0.1 );

  /* Generate dipping dataset for classification 
   * Loog, M., & Duin, R. P. W. (2012). The dipping phenomenon. 
   *
   * @param N     : number of samples for each class
   * @param means : means of the Gaussian blobs
   * @param stds  : standard deviations of the Gaussian blobs
   *
   */
  void Gaussian ( const size_t N=10,
                  const arma::Row<T>& means = {-1,1} );

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

namespace oml {

//-----------------------------------------------------------------------------
// oml::Dataset -> Downloads data from OpenML with given id
//-----------------------------------------------------------------------------

template<class LTYPE = size_t, class T = DTYPE>
class Dataset
{
public:
  size_t id_;                       // Dataset ID
  size_t size_;                     // Number of samples
  size_t dimension_;                // Feature dimension
  std::optional<size_t> num_class_; // Number of classes (if categorical)
  std::filesystem::path path_;      // Dataset path

  arma::Mat<T> inputs_;             // Input data
  arma::Row<LTYPE> labels_;         // Labels

  Dataset() { };
  Dataset(const size_t& id, const std::filesystem::path& path);
  Dataset(const size_t& id);

  // Update dataset content
  void Update(const arma::Mat<T>& input, const arma::Row<LTYPE>& labels);

  /* Serialization with cereal */
  template <class Archive>
  void serialize(Archive& ar)
  {
    ar(CEREAL_NVP(size_),
       CEREAL_NVP(id_),
       CEREAL_NVP(path_.string()),
       CEREAL_NVP(num_class_),
       CEREAL_NVP(meta_url_),
       CEREAL_NVP(down_url_),
       CEREAL_NVP(file_),
       CEREAL_NVP(metafile_),
       CEREAL_NVP(inputs_),
       CEREAL_NVP(labels_),
       CEREAL_NVP(dimension_));
  }

  // Save/load dataset in binary format
  void Save(const std::string& filename);
  static std::shared_ptr<Dataset<LTYPE, T>> Load(const std::string& filename);

private:
  std::filesystem::path filepath_ = path_ / "datasets"; // Data folder
  std::filesystem::path metapath_ = path_ / "meta";     // Metadata folder

  bool _download();                  // Download dataset
  void _update_info();                // Update metadata
  void _load();                       // Load from disk

  bool _iscateg(const arma::Row<T>& row);     // Check if row is categorical
  arma::Row<size_t> _convcateg(const arma::Row<T>& row); // Convert to categories
  arma::Row<size_t> _procrow(const arma::Row<T>& row);   // Process row

  std::string _gettargetname(const std::string& metadata); // Extract target name
  std::string _getdownurl(const std::string& metadata);    // Extract download URL
  int _findlabel(const std::string& targetname);           // Find label index

  std::string _fetchmetadata();    // Fetch metadata from source
  std::string _readmetadata();     // Read metadata from file

  std::string meta_url_;           // Metadata URL
  std::string down_url_;           // Download URL
  std::string file_;               // Dataset file name
  std::string metafile_;           // Metadata file name
};


} // namesapce oml
  
namespace functional
{
template<class T = DTYPE>
struct Dataset 
{
  size_t size_;        // Number of samples in the dataset
  size_t dimension_;   // Number of features (input dimension)
  size_t nfuncs_;      // Number of functions or outputs
  arma::Mat<T> weights_; // Optional sample or feature weights

  arma::Mat<T> inputs_; // Input feature matrix
  arma::Mat<T> labels_; // Output/label matrix

  /* Empty constructor */
  Dataset() { }

  /* Dataset initializer
   *
   * @param D : number of features (input dimension)
   * @param N : number of samples
   * @param M : number of outputs (functions)
   */
  Dataset(const size_t& D,
          const size_t& N,
          const size_t& M);

  /* Generate synthetic dataset
   *
   * @param type : type of dataset to generate
   */
  void Generate(const std::string& type);

  /* Generate synthetic dataset with Gaussian noise
   *
   * @param type      : type of dataset to generate
   * @param noise_std : standard deviation of noise
   */
  void Generate(const std::string& type,
                const double& noise_std);

  /* Generate synthetic dataset with feature-wise Gaussian noise
   *
   * @param type      : type of dataset to generate
   * @param noise_std : per-feature noise standard deviations
   */
  void Generate(const std::string& type,
                const arma::Row<T>& noise_std);

  /* Add Gaussian noise to dataset
   *
   * @param noise_std : standard deviation of noise
   */
  void Noise(const double& noise_std);

  /* Add feature-wise Gaussian noise to dataset
   *
   * @param noise_std : per-feature noise standard deviations
   */
  void Noise(const arma::Row<T>& noise_std);

  /* Save dataset to file
   *
   * @param filename : path to save file
   */
  void Save(const std::string& filename);

  /* Load dataset from file
   *
   * @param filename  : path to file
   * @param Din       : input dimension
   * @param Dout      : output dimension
   * @param transpose : whether to transpose loaded data (default: true)
   * @param count     : whether to count number of samples (default: false)
   */
  void Load(const std::string& filename,
            const size_t& Din,
            const size_t& Dout, 
            const bool& transpose = true,
            const bool& count = false);

  /* Normalize dataset (zero mean, unit variance) */
  void Normalize();

  /* Undo normalization on dataset */
  void UnNormalize();
};


//=============================================================================
// SineGen
//=============================================================================
template<class T = DTYPE>
struct SineGen
{
  size_t size_;        ///< Number of generated samples
  size_t dimension_;   ///< Number of sine functions (output dimension)

  arma::Row<T> a_;     ///< Amplitude parameters for each function
  arma::Row<T> p_;     ///< Phase parameters for each function

  /* Empty constructor */
  SineGen() { }

  /* Initialize sine generator
   *
   * @param M : number of sine functions
   */
  SineGen(const size_t& M);

  /* Predict sine outputs for given inputs
   *
   * @param inputs : input matrix
   * @param type   : prediction type ("Phase" by default)
   * @return       : predicted outputs
   */
  arma::Mat<T> Predict(const arma::Mat<T>& inputs,
                       const std::string& type = "Phase") const;

  /* Predict sine outputs with perturbation
   *
   * @param inputs : input matrix
   * @param type   : prediction type
   * @param eps    : perturbation parameter
   * @return       : predicted outputs
   */
  arma::Mat<T> Predict(const arma::Mat<T>& inputs,
                       const std::string& type,
                       const double& eps) const;

  /* Compute mean prediction
   *
   * @param inputs : input matrix
   * @param type   : mean type ("Phase" by default)
   * @return       : mean outputs
   */
  arma::Mat<T> Mean(const arma::Mat<T>& inputs,
                    const std::string& type = "Phase") const;

  /* Compute mean prediction with perturbation
   *
   * @param inputs : input matrix
   * @param type   : mean type
   * @param eps    : perturbation parameter
   * @return       : mean outputs
   */
  arma::Mat<T> Mean(const arma::Mat<T>& inputs,
                    const std::string& type,
                    const double& eps) const;

  /* Get number of sine functions */
  size_t GetM();
};

} // namespace functional


} // namespace data


#include "dataset_impl.h"

#endif
