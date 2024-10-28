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


void print_info ( const arma::Row<size_t>& ids )
{
  using Dataset = data::classification::oml::Dataset<>;
  for (size_t id: ids)
  {
    Dataset dataset(id, lcdb::path);
    LOG("Dimension  : " << dataset.dimension_);
    LOG("Size       : " << dataset.size_);
  }
}


int main(int argc, char** argv) 
{
  //arma::Row<size_t> datasets = {"1 2 8 10 11 13 15 18 22 23 25 29 31 37 39 41 43 48 49 50 51 53 54 55 56 61 62 171 181 185 186 187 188 194 195 196 199 200 203 204 207 210 213 222 223 224 229 230 231 232 334 463 470 475 685 934 1049 1050 1059 1063 1064 1068 1071 1121 1462 1464 1467 1480 1482 1494 1504 1510 1513 1552 40496 40975 40981 40982"};
  /* arma::Row<size_t> datasets = { 3,6,11,12,14,15,16,18,22,23,28,29,31,32,37,38,44,46, */
  /*                              50,54,151,182,188,300,307,458,469, */
  /*                              554,1049,1050,1053,1063,1067,1068,1461,1462, */
  /*                              1464,1468,1475,1478,1480,1485,1486,1487,1489, */
  /*                              1494,1497,1501,1510,1590,4134,4534,4538,6332, */
  /*                              23381,23517,40499,40668,40670,40701,40923,40927,40966, */
  /*                              40975,40978,40979,40982,40983,40984,40994,40996,41027}; */
  
  arma::Row<size_t> datasets = { 11,15,23,37,1063,4534,151,41027,32,4538,1475};
  print_info(datasets);
  /* using Suite = data::classification::oml::Collect<>; */

  /* arma::Row<size_t> suiteids = {283,271,240,218,379,445,298,425,393,225}; */
  /* /1* arma::Row<size_t> suiteids = {283}; *1/ */
  /* arma::Row<size_t> which; */

  /* for (size_t i = 0; i < suiteids.n_elem; i++) */
  /* { */
  /*   Suite suite(suiteids(i), lcdb::path); */
  /*   for (size_t j = 0; j < suite.GetSize(); j++) */
  /*   { */
  /*     try */
  /*     { */
  /*       auto dataset = suite.GetNext(); // Try loading the dataset */

  /*       // Check if dataset fits the criteria */
  /*       if (dataset.size_ <= lcdb::nlim && dataset.dimension_<lcdb::flim) */
  /*       { */
  /*         which.resize(which.n_elem + 1); */
  /*         which(which.n_elem - 1) = suite.GetKeys()(j); */
  /*       } */
  /*     } */
  /*     catch (const std::exception& e) */
  /*     { */
  /*       // Log the error and skip this dataset */
  /*       ERR("Error loading dataset: " << e.what() << ".Skipping this dataset."); */
  /*       continue; // Skip to the next dataset */
  /*     } */
  /*   } */
  /* } */

  /* LOG("Datasets that fit my criteria are " << arma::unique(which)); */

  return 0;
}
