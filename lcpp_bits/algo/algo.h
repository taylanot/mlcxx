/**
 * @file algo.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef ALGO_H
#define ALGO_H


#include "regressors/regressors.h"
#include "classifiers/classifiers.h"
#include "dimred/dimred.h"
#include "nn.h"

namespace algo 
{

  /*
   * Save your models easily
   *
   * @param filePath -> where to save
   * @param model -> which model object
   */
  template<class MODEL>
  void save( const std::filesystem::path& filePath,
             const MODEL& model)
  {
    std::ofstream ofs(filePath, std::ios::binary);
    if (!ofs)
      throw std::runtime_error("Cannot open file for writing: " + filePath.string());

    {
      cereal::BinaryOutputArchive archive(ofs);
      archive(cereal::make_nvp("model", model));
    } // archive is destroyed here, ensuring flush
  }

  /*
   * Load your models easily
   *
   * @param filePath -> where to save
   * @param model -> which model object
   */
  template<class MODEL>
  std::shared_ptr<MODEL> load( const std::filesystem::path &filePath )
  {
    std::ifstream ifs(filePath, std::ios::binary);
    if (!ifs)
      throw std::runtime_error("Cannot open file for reading: " + filePath.string());

    auto loaded = std::make_shared<MODEL>();
    {
      cereal::BinaryInputArchive archive(ifs);
      archive(cereal::make_nvp("model", *loaded));
    } // archive destroyed here

    return loaded;
  }
} //namespace algo
#endif
