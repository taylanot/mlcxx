/**
 * @file collect.h
 * @author Ozgur Taylan Turan
 *
 * COllection of datasets
 */

#ifndef COLLECT_H
#define COLLECT_H

namespace data {
namespace classification {
namespace oml {

//=============================================================================
// Collect
//=============================================================================
template<class T=DTYPE>
class Collect
{
public:
  /*
   * @param id  : id of the study
   */ 
  Collect ( const size_t& id );

  /*
   * @param ids  : ids of datasets
   */ 
  Collect ( const arma::Row<size_t>& ids );

  /*
   * @param id    : id of the study
   * @param paht  : path to save the study information
   */ 
  Collect ( const size_t& id, const std::filesystem::path& path );

  Dataset<T> GetNext (  ); 

  Dataset<T> GetID ( const size_t& id ); 

  /* size_t GetSize (  ) const {return size_;} */
  /* size_t GetCounter (  ) const {return counter_;} */
  /* arma::Row<size_t> GetKeys ( ) const {return keys_;} */

/* private: */
  size_t id_;
  size_t size_;
  size_t counter_ = 0;

  std::string url_;

  arma::Row<size_t> _getkeys ( ); 

  arma::Row<size_t> keys_;

  std::filesystem::path path_; 
  std::filesystem::path filepath_ = path_ / "openml/collect";
  std::filesystem::path metafile_ = filepath_ / (std::to_string(id_)+".arff");

};


} // namesapce oml
} // namesapce classification
} // namesapce data

#include "collect_impl.h"

#endif


