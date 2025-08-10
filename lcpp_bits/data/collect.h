/**
 * @file collect.h
 * @author Ozgur Taylan Turan
 *
 * 
 */

#ifndef COLLECT_H
#define COLLECT_H

namespace data {
namespace oml {

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
} // namesapce data

#include "collect_impl.h"

#endif


