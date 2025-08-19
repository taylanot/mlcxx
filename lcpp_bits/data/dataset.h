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

  /* Generate banana dataset for classification a.k.a the moon dataset in 
   * sklearn.
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

  /* Create Gaussian blobs with number of blobs equal to the means provided
   * all the blobs have spherical covariance.
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

  bool _iscateg(const arma::Row<T>& row); // Check if row is categorical
  arma::Row<size_t> _convcateg(const arma::Row<T>& row); // Convert to categs
  arma::Row<size_t> _procrow(const arma::Row<T>& row);   // Process row

  std::string _gettargetname(const std::string& metadata);// target name
  std::string _getdownurl(const std::string& metadata);   // download URL
  int _findlabel(const std::string& targetname);          // Find label index

  std::string _fetchmetadata();    // Fetch metadata from source
  std::string _readmetadata();     // Read metadata from file

  std::string meta_url_;           // Metadata URL
  std::string down_url_;           // Download URL
  std::string file_;               // Dataset file name
  std::string metafile_;           // Metadata file name
};

//-----------------------------------------------------------------------------
// Collect : This is for collection of datasets through OpenML servers
//-----------------------------------------------------------------------------
template<class T=size_t>
class Collect
{
public:
  /*
   * Const
   * @param id  : id of the study
   */ 
  Collect ( const size_t& id );

  /*
   * @param ids  : ids of datasets
   */ 
  Collect ( const arma::Row<size_t>& ids );

  /*
   * @param id    : id of the study
   * @param paht  : path to save the collection
   */ 
  Collect ( const size_t& id, const std::filesystem::path& path );

  Dataset<T> GetNext (  ); 

  Dataset<T> GetID ( const size_t& id ); 

  size_t GetSize (  ) const {return size_;}
  size_t GetCounter (  ) const {return counter_;}
  arma::Row<size_t> GetKeys ( ) const {return keys_;}

private:
  size_t id_;
  size_t size_;
  size_t counter_ = 0;

  std::string url_;

  arma::Row<size_t> _getkeys ( ); 

  arma::Row<size_t> keys_;

  std::filesystem::path path_; 
  std::filesystem::path filespath_ = path_ / "collect";
  std::filesystem::path metapath_ = path_ / "collect";
  std::filesystem::path metafile_ = metapath_ / (std::to_string(id_)+".meta");

};

} // namesapce oml
  
} // namespace data


#include "dataset_impl.h"

#endif
