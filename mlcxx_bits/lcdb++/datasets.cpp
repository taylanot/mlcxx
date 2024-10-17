/**
 * @file dataset.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for collecting the datasets needed for generating lcdb++ 
 * database
 *
 */

#define DTYPE double  

#include <headers.h>
#include "lcdb++/config.h"


int main(int argc, char** argv) 
{
  using Suite = data::classification::oml::Collect<>;

  arma::Row<size_t> suiteids = {283,271,240,218,379,445,298,425,393,225};
  /* arma::Row<size_t> suiteids = {283}; */
  arma::Row<size_t> which;

  for (size_t i = 0; i < suiteids.n_elem; i++)
  {
    Suite suite(suiteids(i), lcdb::path);
    for (size_t j = 0; j < suite.GetSize(); j++)
    {
      try
      {
        auto dataset = suite.GetNext(); // Try loading the dataset

        // Check if dataset fits the criteria
        if (dataset.size_ <= lcdb::nlim)
        {
          which.resize(which.n_elem + 1);
          which(which.n_elem - 1) = suite.GetKeys()(j);
        }
      }
      catch (const std::exception& e)
      {
        // Log the error and skip this dataset
        ERR("Error loading dataset: " << e.what() << ".Skipping this dataset.");
        continue; // Skip to the next dataset
      }
    }
  }

  LOG("Datasets that fit my criteria are " << arma::unique(which));
  return 0;
}
