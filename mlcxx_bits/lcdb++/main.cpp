/**
 * @file main.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for lcdb++ generation
 *
 */

#define DTYPE double  

#include <headers.h>
#include "lcdb++/config.h"

template<class MODEL,class LOSS>
int rand_( const size_t id,
           const std::string& algo, const std::string& loss,
           const size_t seed, const size_t nreps,
           const std::filesystem::path& path ) 
{
  data::classification::oml::Dataset dataset(id,lcdb::path);

  mlpack::RandomSeed(seed);
  /* const arma::irowvec Ns = arma::regspace<arma::irowvec> */
  /*                                             (1,1,size_t(dataset.size_*0.9)); */

  src::LCurve<MODEL,LOSS> lcurve(lcdb::Ns,nreps,true,true);
  lcurve.RandomSet(dataset,
                   arma::unique(dataset.labels_).eval().n_elem);
  lcurve.GetResults().save(path/(loss+".csv"),arma::csv_ascii);
 
  return 0;
}

template<class MODEL,class LOSS,class... Args>
int hptrand_( const size_t id,
              const std::string& algo, const std::string& loss,
              const size_t seed, const size_t nreps,
              const std::filesystem::path& path, Args... args ) 
{
 
  data::classification::oml::Dataset dataset(id,lcdb::path);

  mlpack::RandomSeed(seed);
  /* const arma::irowvec Ns = arma::regspace<arma::irowvec> */
  /*                                             (10,1,size_t(dataset.size_*0.9)); */

  src::LCurveHPT<MODEL,LOSS> lcurve(lcdb::hptNs,nreps,lcdb::vsize,true,true);
  lcurve.RandomSet(dataset,
                   mlpack::Fixed(arma::unique(dataset.labels_).eval().n_elem),
                   args...);
  lcurve.GetResults().save(path/(loss+".csv"),arma::csv_ascii);
 
  return 0;
}

template<class MODEL,class LOSS>
int boot_( const size_t id,
           const std::string& algo, const std::string& loss,
           const size_t seed, const size_t nreps,
           const std::filesystem::path& path ) 
{
  data::classification::oml::Dataset dataset(id,lcdb::path);

  mlpack::RandomSeed(seed);
  /* const arma::irowvec Ns = arma::regspace<arma::irowvec> */
  /*                                             (1,1,size_t(dataset.size_*0.9)); */

  src::LCurve<MODEL,LOSS> lcurve(lcdb::Ns,nreps,true,true);
  lcurve.Bootstrap(dataset,
                   arma::unique(dataset.labels_).eval().n_elem);
  lcurve.GetResults().save(path/(loss+".csv"),arma::csv_ascii);
 
  return 0;
}

template<class MODEL,class LOSS,class... Args>
int hptboot_( const size_t id,
              const std::string& algo, const std::string& loss,
              const size_t seed, const size_t nreps,
              const std::filesystem::path& path, Args... args ) 
{
 
  data::classification::oml::Dataset dataset(id,lcdb::path);

  mlpack::RandomSeed(seed);
  /* const arma::irowvec Ns = arma::regspace<arma::irowvec> */
  /*                                             (10,1,size_t(dataset.size_*0.9)); */

  src::LCurveHPT<MODEL,LOSS> lcurve(lcdb::hptNs,nreps,lcdb::vsize,true,true);
  lcurve.Bootstrap(dataset,
                   mlpack::Fixed(arma::unique(dataset.labels_).eval().n_elem),
                   args...);
  lcurve.GetResults().save(path/(loss+".csv"),arma::csv_ascii);
 
  return 0;
}

template<class MODEL,class LOSS>
int add_( const size_t id,
          const std::string& algo, const std::string& loss,
          const size_t seed, const size_t nreps,
           const std::filesystem::path& path ) 
{
  data::classification::oml::Dataset dataset(id,lcdb::path);

  mlpack::RandomSeed(seed);

  /* const arma::irowvec Ns = arma::regspace<arma::irowvec> */
    /* (1,1,size_t(dataset.size_*0.9)); */

  src::LCurve<MODEL,LOSS> lcurve(lcdb::Ns,nreps,true,true);
  lcurve.Additive(dataset,
                  arma::unique(dataset.labels_).eval().n_elem);
  lcurve.GetResults().save(path/(loss+".csv"),arma::csv_ascii);
 
  return 0;
}

template<class MODEL,class LOSS,class... Args>
int hptadd_( const size_t id,
              const std::string& algo, const std::string& loss,
              const size_t seed, const size_t nreps,
              const std::filesystem::path& path, Args... args ) 
{
 
  data::classification::oml::Dataset dataset(id,lcdb::path);

  mlpack::RandomSeed(seed);

  /* const arma::irowvec Ns = arma::regspace<arma::irowvec> */
  /*   (10,1,size_t(dataset.size_*0.9)); */

  src::LCurveHPT<MODEL,LOSS> lcurve(lcdb::hptNs,nreps,lcdb::vsize, true,true);
  lcurve.Additive(dataset,
                   mlpack::Fixed(arma::unique(dataset.labels_).eval().n_elem),
                   args...);
  lcurve.GetResults().save(path/(loss+".csv"),arma::csv_ascii);
 
  return 0;
}

template<class MODEL,class LOSS>
int split_( const size_t id,
           const std::string& algo, const std::string& loss,
           const size_t seed, const size_t nreps,
           const std::filesystem::path& path ) 
{
  data::classification::oml::Dataset dataset(id,lcdb::path);

  mlpack::RandomSeed(seed);
  data::classification::oml::Dataset trainset,testset;

  data::StratifiedSplit(dataset,trainset,testset,lcdb::splitsize);

  /* const arma::irowvec Ns = arma::regspace<arma::irowvec> */
  /*   (1,1,size_t(trainset.size_*0.9)); */

  src::LCurve<MODEL,LOSS> lcurve(lcdb::Ns,nreps,true,true);
  lcurve.Split(trainset,testset,arma::unique(dataset.labels_).eval().n_elem);
  lcurve.GetResults().save(path/(loss+".csv"),arma::csv_ascii);
 
  return 0;
}

template<class MODEL,class LOSS,class... Args>
int hptsplit_( const size_t id,
              const std::string& algo, const std::string& loss,
              const size_t seed, const size_t nreps,
              const std::filesystem::path& path, Args... args ) 
{
 
  data::classification::oml::Dataset dataset(id,lcdb::path);

  mlpack::RandomSeed(seed);
  data::classification::oml::Dataset trainset,testset;

  data::StratifiedSplit(dataset,trainset,testset,lcdb::splitsize);

  const arma::irowvec Ns = arma::regspace<arma::irowvec>
    (10,1,size_t(trainset.size_*0.9));

  src::LCurveHPT<MODEL,LOSS> lcurve(Ns,nreps,lcdb::vsize, true,true);
  lcurve.Split(trainset,testset,
               mlpack::Fixed(arma::unique(dataset.labels_).eval().n_elem),
               args...);
  lcurve.GetResults().save(path/(loss+".csv"),arma::csv_ascii);
 
  return 0;
}

void rands( size_t id, const std::string& algo, const std::string& loss,
           size_t seed, size_t nreps,  bool hpt )
{
    std::filesystem::path path;
    if (!hpt)
      path = lcdb::path/"rands"/"ntune";
    else
      path = lcdb::path/"rands"/"tune";

    path = path/std::to_string(id)/algo/std::to_string(seed);
    std::filesystem::create_directories(path);

    using cont = arma::Row<DTYPE>;
    /* using disc = arma::Row<size_t>; */

    using Func = std::function<void(const size_t id,
                                    const std::string&,
                                    const std::string&,
                                    const size_t,
                                    const size_t,
                                    const std::filesystem::path&)>;

    using contFunc = std::function<void(const size_t id,
                                        const std::string&,
                                        const std::string&,
                                        const size_t,
                                        const size_t,
                                        const std::filesystem::path&, 
                                        const cont&)>;

    // Mapping of algo and loss types
    std::unordered_map<std::string,
                       std::unordered_map<std::string,Func>> run = 
    {
      {"lreg", {{"acc", rand_<lcdb::LREG, lcdb::Acc>},
                {"crs", rand_<lcdb::LREG, lcdb::Crs>},
                {"bri", rand_<lcdb::LREG, lcdb::Bri>},
                {"auc", rand_<lcdb::LREG, lcdb::Auc>}}},
      {"nmc",  {{"acc", rand_<lcdb::NMC, lcdb::Acc>},
                {"crs", rand_<lcdb::NMC, lcdb::Crs>},
                {"bri", rand_<lcdb::NMC, lcdb::Bri>},
                {"auc", rand_<lcdb::NMC, lcdb::Auc>}}},
      {"nnc",  {{"acc", rand_<lcdb::NNC, lcdb::Acc>},
                {"crs", rand_<lcdb::NNC, lcdb::Crs>},
                {"bri", rand_<lcdb::NNC, lcdb::Bri>},
                {"auc", rand_<lcdb::NNC, lcdb::Auc>}}},
      {"ldc",  {{"acc", rand_<lcdb::LDC, lcdb::Acc>},
                {"crs", rand_<lcdb::LDC, lcdb::Crs>},
                {"bri", rand_<lcdb::LDC, lcdb::Bri>},
                {"auc", rand_<lcdb::LDC, lcdb::Auc>}}},
      {"qdc",  {{"acc", rand_<lcdb::QDC, lcdb::Acc>},
                {"crs", rand_<lcdb::QDC, lcdb::Crs>},
                {"bri", rand_<lcdb::QDC, lcdb::Bri>},
                {"auc", rand_<lcdb::QDC, lcdb::Auc>}}},
      {"lsvc", {{"acc", rand_<lcdb::LSVC, lcdb::Acc>},
                {"crs", rand_<lcdb::LSVC, lcdb::Crs>},
                {"bri", rand_<lcdb::LSVC, lcdb::Bri>},
                {"auc", rand_<lcdb::LSVC, lcdb::Auc>}}},
      {"esvc", {{"acc", rand_<lcdb::ESVC, lcdb::Acc>},
                {"crs", rand_<lcdb::ESVC, lcdb::Crs>},
                {"bri", rand_<lcdb::ESVC, lcdb::Bri>},
                {"auc", rand_<lcdb::ESVC, lcdb::Auc>}}},
      {"gsvc", {{"acc", rand_<lcdb::GSVC, lcdb::Acc>},
                {"crs", rand_<lcdb::GSVC, lcdb::Crs>},
                {"bri", rand_<lcdb::GSVC, lcdb::Bri>},
                {"auc", rand_<lcdb::GSVC, lcdb::Auc>}}},
      {"adab", {{"acc", rand_<lcdb::ADAB, lcdb::Acc>},
                {"crs", rand_<lcdb::ADAB, lcdb::Crs>},
                {"bri", rand_<lcdb::ADAB, lcdb::Bri>},
                {"auc", rand_<lcdb::ADAB, lcdb::Auc>}}},
      {"rfor", {{"acc", rand_<lcdb::RFOR, lcdb::Acc>},
                {"crs", rand_<lcdb::RFOR, lcdb::Crs>},
                {"bri", rand_<lcdb::RFOR, lcdb::Bri>},
                {"auc", rand_<lcdb::RFOR, lcdb::Auc>}}},
      {"dt",   {{"acc", rand_<lcdb::DT, lcdb::Acc>},
                {"crs", rand_<lcdb::DT, lcdb::Crs>},
                {"bri", rand_<lcdb::DT, lcdb::Bri>},
                {"auc", rand_<lcdb::DT, lcdb::Auc>}}},
      {"nb",   {{"acc", rand_<lcdb::NB, lcdb::Acc>},
                {"crs", rand_<lcdb::NB, lcdb::Crs>},
                {"bri", rand_<lcdb::NB, lcdb::Bri>},
                {"auc", rand_<lcdb::NB, lcdb::Auc>}}}
    };

       std::unordered_map<std::string,
                       std::unordered_map<std::string,contFunc>> contrun =

    {
      {"lreg", {{"acc", hptrand_<lcdb::LREG, lcdb::Acc,cont>},
                {"crs", hptrand_<lcdb::LREG, lcdb::Crs,cont>},
                {"bri", hptrand_<lcdb::LREG, lcdb::Bri,cont>},
                {"auc", hptrand_<lcdb::LREG, lcdb::Auc,cont>}}},
      {"ldc",  {{"acc", hptrand_<lcdb::LDC, lcdb::Acc,cont>},
                {"crs", hptrand_<lcdb::LDC, lcdb::Crs,cont>},
                {"bri", hptrand_<lcdb::LDC, lcdb::Bri,cont>},
                {"auc", hptrand_<lcdb::LDC, lcdb::Auc,cont>}}},
      {"qdc",  {{"acc", hptrand_<lcdb::QDC, lcdb::Acc,cont>},
                {"crs", hptrand_<lcdb::QDC, lcdb::Crs,cont>},
                {"bri", hptrand_<lcdb::QDC, lcdb::Bri,cont>},
                {"auc", hptrand_<lcdb::QDC, lcdb::Auc,cont>}}},
      {"lsvc", {{"acc", hptrand_<lcdb::LSVC, lcdb::Acc,cont>},
                {"crs", hptrand_<lcdb::LSVC, lcdb::Crs,cont>},
                {"bri", hptrand_<lcdb::LSVC, lcdb::Bri,cont>},
                {"auc", hptrand_<lcdb::LSVC, lcdb::Auc,cont>}}},
      {"esvc", {{"acc", hptrand_<lcdb::ESVC, lcdb::Acc,cont>},
                {"crs", hptrand_<lcdb::ESVC, lcdb::Crs,cont>},
                {"bri", hptrand_<lcdb::ESVC, lcdb::Bri,cont>},
                {"auc", hptrand_<lcdb::ESVC, lcdb::Auc,cont>}}},
      {"gsvc", {{"acc", hptrand_<lcdb::GSVC, lcdb::Acc,cont>},
                {"crs", hptrand_<lcdb::GSVC, lcdb::Crs,cont>},
                {"bri", hptrand_<lcdb::GSVC, lcdb::Bri,cont>},
                {"auc", hptrand_<lcdb::GSVC, lcdb::Auc,cont>}}}
    };


    // Check if algo exists in the map
    if (run.find(algo) != run.end() && !hpt) 
    {
      auto loss_map = run[algo];
      if (loss_map.find(loss) != loss_map.end()) 
          // Call the corresponding function
          loss_map[loss](id, algo, loss, seed, nreps, path);
      else 
          ERR("Not defined loss argument!");
    }
    else if (contrun.find(algo) != contrun.end() && hpt)
    {
      auto loss_map = contrun[algo];
      if (loss_map.find(loss) != loss_map.end()) 
          loss_map[loss](id, algo, loss, seed, nreps, path, lcdb::lambdas);
      else 
          ERR("Not defined loss argument!");
    }
    else 
        ERR("Not defined algo argument!");
}

void boot( size_t id, const std::string& algo, const std::string& loss,
           size_t seed, size_t nreps,  bool hpt )
{
    std::filesystem::path path;
    if (!hpt)
      path = lcdb::path/"boot"/"ntune";
    else
      path = lcdb::path/"boot"/"tune";

    path = path/std::to_string(id)/algo/std::to_string(seed);
    std::filesystem::create_directories(path);

    using cont = arma::Row<DTYPE>;
    /* using disc = arma::Row<size_t>; */

    using Func = std::function<void(const size_t id,
                                    const std::string&,
                                    const std::string&,
                                    const size_t,
                                    const size_t,
                                    const std::filesystem::path&)>;

    using contFunc = std::function<void(const size_t id,
                                        const std::string&,
                                        const std::string&,
                                        const size_t,
                                        const size_t,
                                        const std::filesystem::path&, 
                                        const cont&)>;

    // Mapping of algo and loss types
    std::unordered_map<std::string,
                       std::unordered_map<std::string,Func>> run = 
    {
      {"lreg", {{"acc", boot_<lcdb::LREG, lcdb::Acc>},
                {"crs", boot_<lcdb::LREG, lcdb::Crs>},
                {"bri", boot_<lcdb::LREG, lcdb::Bri>},
                {"auc", boot_<lcdb::LREG, lcdb::Auc>}}},
      {"nmc",  {{"acc", boot_<lcdb::NMC, lcdb::Acc>},
                {"crs", boot_<lcdb::NMC, lcdb::Crs>},
                {"bri", boot_<lcdb::NMC, lcdb::Bri>},
                {"auc", boot_<lcdb::NMC, lcdb::Auc>}}},
      {"nnc",  {{"acc", boot_<lcdb::NNC, lcdb::Acc>},
                {"crs", boot_<lcdb::NNC, lcdb::Crs>},
                {"bri", boot_<lcdb::NNC, lcdb::Bri>},
                {"auc", boot_<lcdb::NNC, lcdb::Auc>}}},
      {"ldc",  {{"acc", boot_<lcdb::LDC, lcdb::Acc>},
                {"crs", boot_<lcdb::LDC, lcdb::Crs>},
                {"bri", boot_<lcdb::LDC, lcdb::Bri>},
                {"auc", boot_<lcdb::LDC, lcdb::Auc>}}},
      {"qdc",  {{"acc", boot_<lcdb::QDC, lcdb::Acc>},
                {"crs", boot_<lcdb::QDC, lcdb::Crs>},
                {"bri", boot_<lcdb::QDC, lcdb::Bri>},
                {"auc", boot_<lcdb::QDC, lcdb::Auc>}}},
      {"lsvc", {{"acc", boot_<lcdb::LSVC, lcdb::Acc>},
                {"crs", boot_<lcdb::LSVC, lcdb::Crs>},
                {"bri", boot_<lcdb::LSVC, lcdb::Bri>},
                {"auc", boot_<lcdb::LSVC, lcdb::Auc>}}},
      {"esvc", {{"acc", boot_<lcdb::ESVC, lcdb::Acc>},
                {"crs", boot_<lcdb::ESVC, lcdb::Crs>},
                {"bri", boot_<lcdb::ESVC, lcdb::Bri>},
                {"auc", boot_<lcdb::ESVC, lcdb::Auc>}}},
      {"gsvc", {{"acc", boot_<lcdb::GSVC, lcdb::Acc>},
                {"crs", boot_<lcdb::GSVC, lcdb::Crs>},
                {"bri", boot_<lcdb::GSVC, lcdb::Bri>},
                {"auc", boot_<lcdb::GSVC, lcdb::Auc>}}},
      {"adab", {{"acc", boot_<lcdb::ADAB, lcdb::Acc>},
                {"crs", boot_<lcdb::ADAB, lcdb::Crs>},
                {"bri", boot_<lcdb::ADAB, lcdb::Bri>},
                {"auc", boot_<lcdb::ADAB, lcdb::Auc>}}},
      {"rfor", {{"acc", boot_<lcdb::RFOR, lcdb::Acc>},
                {"crs", boot_<lcdb::RFOR, lcdb::Crs>},
                {"bri", boot_<lcdb::RFOR, lcdb::Bri>},
                {"auc", boot_<lcdb::RFOR, lcdb::Auc>}}},
      {"dt",   {{"acc", boot_<lcdb::DT, lcdb::Acc>},
                {"crs", boot_<lcdb::DT, lcdb::Crs>},
                {"bri", boot_<lcdb::DT, lcdb::Bri>},
                {"auc", boot_<lcdb::DT, lcdb::Auc>}}},
      {"nb",   {{"acc", boot_<lcdb::NB, lcdb::Acc>},
                {"crs", boot_<lcdb::NB, lcdb::Crs>},
                {"bri", boot_<lcdb::NB, lcdb::Bri>},
                {"auc", boot_<lcdb::NB, lcdb::Auc>}}}
    };

       std::unordered_map<std::string,
                       std::unordered_map<std::string,contFunc>> contrun =

    {
      {"lreg", {{"acc", hptboot_<lcdb::LREG, lcdb::Acc,cont>},
                {"crs", hptboot_<lcdb::LREG, lcdb::Crs,cont>},
                {"bri", hptboot_<lcdb::LREG, lcdb::Bri,cont>},
                {"auc", hptboot_<lcdb::LREG, lcdb::Auc,cont>}}},
      {"ldc",  {{"acc", hptboot_<lcdb::LDC, lcdb::Acc,cont>},
                {"crs", hptboot_<lcdb::LDC, lcdb::Crs,cont>},
                {"bri", hptboot_<lcdb::LDC, lcdb::Bri,cont>},
                {"auc", hptboot_<lcdb::LDC, lcdb::Auc,cont>}}},
      {"qdc",  {{"acc", hptboot_<lcdb::QDC, lcdb::Acc,cont>},
                {"crs", hptboot_<lcdb::QDC, lcdb::Crs,cont>},
                {"bri", hptboot_<lcdb::QDC, lcdb::Bri,cont>},
                {"auc", hptboot_<lcdb::QDC, lcdb::Auc,cont>}}},
      {"lsvc", {{"acc", hptboot_<lcdb::LSVC, lcdb::Acc,cont>},
                {"crs", hptboot_<lcdb::LSVC, lcdb::Crs,cont>},
                {"bri", hptboot_<lcdb::LSVC, lcdb::Bri,cont>},
                {"auc", hptboot_<lcdb::LSVC, lcdb::Auc,cont>}}},
      {"esvc", {{"acc", hptboot_<lcdb::ESVC, lcdb::Acc,cont>},
                {"crs", hptboot_<lcdb::ESVC, lcdb::Crs,cont>},
                {"bri", hptboot_<lcdb::ESVC, lcdb::Bri,cont>},
                {"auc", hptboot_<lcdb::ESVC, lcdb::Auc,cont>}}},
      {"gsvc", {{"acc", hptboot_<lcdb::GSVC, lcdb::Acc,cont>},
                {"crs", hptboot_<lcdb::GSVC, lcdb::Crs,cont>},
                {"bri", hptboot_<lcdb::GSVC, lcdb::Bri,cont>},
                {"auc", hptboot_<lcdb::GSVC, lcdb::Auc,cont>}}}
    };


    // Check if algo exists in the map
    if (run.find(algo) != run.end() && !hpt) 
    {
      auto loss_map = run[algo];
      if (loss_map.find(loss) != loss_map.end()) 
          // Call the corresponding function
          loss_map[loss](id, algo, loss, seed, nreps, path);
      else 
          ERR("Not defined loss argument!");
    }
    else if (contrun.find(algo) != contrun.end() && hpt)
    {
      auto loss_map = contrun[algo];
      if (loss_map.find(loss) != loss_map.end()) 
          loss_map[loss](id, algo, loss, seed, nreps, path, lcdb::lambdas);
      else 
          ERR("Not defined loss argument!");
    }
    else 
        ERR("Not defined algo argument!");
}

void split ( size_t id, const std::string& algo, const std::string& loss,
             size_t seed, size_t nreps,  bool hpt )
{
    std::filesystem::path path;
    if (!hpt)
      path = lcdb::path/"split"/"ntune";
    else
      path = lcdb::path/"split"/"tune";

    path = path/std::to_string(id)/algo/std::to_string(seed);
    std::filesystem::create_directories(path);

    using cont = arma::Row<DTYPE>;
    /* using disc = arma::Row<size_t>; */

    using Func = std::function<void(const size_t id,
                                    const std::string&,
                                    const std::string&,
                                    const size_t,
                                    const size_t,
                                    const std::filesystem::path&)>;

    using contFunc = std::function<void(const size_t id,
                                        const std::string&,
                                        const std::string&,
                                        const size_t,
                                        const size_t,
                                        const std::filesystem::path&, 
                                        const cont&)>;

    // Mapping of algo and loss types
    std::unordered_map<std::string,
                       std::unordered_map<std::string,Func>> run = 
    {
      {"lreg", {{"acc", split_<lcdb::LREG, lcdb::Acc>},
                {"crs", split_<lcdb::LREG, lcdb::Crs>},
                {"bri", split_<lcdb::LREG, lcdb::Bri>},
                {"auc", split_<lcdb::LREG, lcdb::Auc>}}},
      {"nmc",  {{"acc", split_<lcdb::NMC, lcdb::Acc>},
                {"crs", split_<lcdb::NMC, lcdb::Crs>},
                {"bri", split_<lcdb::NMC, lcdb::Bri>},
                {"auc", split_<lcdb::NMC, lcdb::Auc>}}},
      {"nnc",  {{"acc", split_<lcdb::NNC, lcdb::Acc>},
                {"crs", split_<lcdb::NNC, lcdb::Crs>},
                {"bri", split_<lcdb::NNC, lcdb::Bri>},
                {"auc", split_<lcdb::NNC, lcdb::Auc>}}},
      {"ldc",  {{"acc", split_<lcdb::LDC, lcdb::Acc>},
                {"crs", split_<lcdb::LDC, lcdb::Crs>},
                {"bri", split_<lcdb::LDC, lcdb::Bri>},
                {"auc", split_<lcdb::LDC, lcdb::Auc>}}},
      {"qdc",  {{"acc", split_<lcdb::QDC, lcdb::Acc>},
                {"crs", split_<lcdb::QDC, lcdb::Crs>},
                {"bri", split_<lcdb::QDC, lcdb::Bri>},
                {"auc", split_<lcdb::QDC, lcdb::Auc>}}},
      {"lsvc", {{"acc", split_<lcdb::LSVC, lcdb::Acc>},
                {"crs", split_<lcdb::LSVC, lcdb::Crs>},
                {"bri", split_<lcdb::LSVC, lcdb::Bri>},
                {"auc", split_<lcdb::LSVC, lcdb::Auc>}}},
      {"esvc", {{"acc", split_<lcdb::ESVC, lcdb::Acc>},
                {"crs", split_<lcdb::ESVC, lcdb::Crs>},
                {"bri", split_<lcdb::ESVC, lcdb::Bri>},
                {"auc", split_<lcdb::ESVC, lcdb::Auc>}}},
      {"gsvc", {{"acc", split_<lcdb::GSVC, lcdb::Acc>},
                {"crs", split_<lcdb::GSVC, lcdb::Crs>},
                {"bri", split_<lcdb::GSVC, lcdb::Bri>},
                {"auc", split_<lcdb::GSVC, lcdb::Auc>}}},
      {"adab", {{"acc", split_<lcdb::ADAB, lcdb::Acc>},
                {"crs", split_<lcdb::ADAB, lcdb::Crs>},
                {"bri", split_<lcdb::ADAB, lcdb::Bri>},
                {"auc", split_<lcdb::ADAB, lcdb::Auc>}}},
      {"rfor", {{"acc", split_<lcdb::RFOR, lcdb::Acc>},
                {"crs", split_<lcdb::RFOR, lcdb::Crs>},
                {"bri", split_<lcdb::RFOR, lcdb::Bri>},
                {"auc", split_<lcdb::RFOR, lcdb::Auc>}}},
      {"dt",   {{"acc", split_<lcdb::DT, lcdb::Acc>},
                {"crs", split_<lcdb::DT, lcdb::Crs>},
                {"bri", split_<lcdb::DT, lcdb::Bri>},
                {"auc", split_<lcdb::DT, lcdb::Auc>}}},
      {"nb",   {{"acc", split_<lcdb::NB, lcdb::Acc>},
                {"crs", split_<lcdb::NB, lcdb::Crs>},
                {"bri", split_<lcdb::NB, lcdb::Bri>},
                {"auc", split_<lcdb::NB, lcdb::Auc>}}}
    };

       std::unordered_map<std::string,
                       std::unordered_map<std::string,contFunc>> contrun =

    {
      {"lreg", {{"acc", hptsplit_<lcdb::LREG, lcdb::Acc,cont>},
                {"crs", hptsplit_<lcdb::LREG, lcdb::Crs,cont>},
                {"bri", hptsplit_<lcdb::LREG, lcdb::Bri,cont>},
                {"auc", hptsplit_<lcdb::LREG, lcdb::Auc,cont>}}},
      {"ldc",  {{"acc", hptsplit_<lcdb::LDC, lcdb::Acc,cont>},
                {"crs", hptsplit_<lcdb::LDC, lcdb::Crs,cont>},
                {"bri", hptsplit_<lcdb::LDC, lcdb::Bri,cont>},
                {"auc", hptsplit_<lcdb::LDC, lcdb::Auc,cont>}}},
      {"qdc",  {{"acc", hptsplit_<lcdb::QDC, lcdb::Acc,cont>},
                {"crs", hptsplit_<lcdb::QDC, lcdb::Crs,cont>},
                {"bri", hptsplit_<lcdb::QDC, lcdb::Bri,cont>},
                {"auc", hptsplit_<lcdb::QDC, lcdb::Auc,cont>}}},
      {"lsvc", {{"acc", hptsplit_<lcdb::LSVC, lcdb::Acc,cont>},
                {"crs", hptsplit_<lcdb::LSVC, lcdb::Crs,cont>},
                {"bri", hptsplit_<lcdb::LSVC, lcdb::Bri,cont>},
                {"auc", hptsplit_<lcdb::LSVC, lcdb::Auc,cont>}}},
      {"esvc", {{"acc", hptsplit_<lcdb::ESVC, lcdb::Acc,cont>},
                {"crs", hptsplit_<lcdb::ESVC, lcdb::Crs,cont>},
                {"bri", hptsplit_<lcdb::ESVC, lcdb::Bri,cont>},
                {"auc", hptsplit_<lcdb::ESVC, lcdb::Auc,cont>}}},
      {"gsvc", {{"acc", hptsplit_<lcdb::GSVC, lcdb::Acc,cont>},
                {"crs", hptsplit_<lcdb::GSVC, lcdb::Crs,cont>},
                {"bri", hptsplit_<lcdb::GSVC, lcdb::Bri,cont>},
                {"auc", hptsplit_<lcdb::GSVC, lcdb::Auc,cont>}}}
    };


    // Check if algo exists in the map
    if (run.find(algo) != run.end() && !hpt) 
    {
      auto loss_map = run[algo];
      if (loss_map.find(loss) != loss_map.end()) 
          // Call the corresponding function
          loss_map[loss](id, algo, loss, seed, nreps, path);
      else 
          ERR("Not defined loss argument!");
    }
    else if (contrun.find(algo) != contrun.end() && hpt)
    {
      auto loss_map = contrun[algo];
      if (loss_map.find(loss) != loss_map.end()) 
          loss_map[loss](id, algo, loss, seed, nreps, path, lcdb::lambdas);
      else 
          ERR("Not defined loss argument!");
    }
    else 
        ERR("Not defined algo argument!");
}

void add ( size_t id, const std::string& algo, const std::string& loss,
           size_t seed, size_t nreps,  bool hpt )
{
    std::filesystem::path path;
    if (!hpt)
      path = lcdb::path/"add"/"ntune";
    else
      path = lcdb::path/"add"/"tune";

    path = path/std::to_string(id)/algo/std::to_string(seed);
    std::filesystem::create_directories(path);

    using cont = arma::Row<DTYPE>;
    /* using disc = arma::Row<size_t>; */

    using Func = std::function<void(const size_t id,
                                    const std::string&,
                                    const std::string&,
                                    const size_t,
                                    const size_t,
                                    const std::filesystem::path&)>;

    using contFunc = std::function<void(const size_t id,
                                        const std::string&,
                                        const std::string&,
                                        const size_t,
                                        const size_t,
                                        const std::filesystem::path&, 
                                        const cont&)>;

    // Mapping of algo and loss types
    std::unordered_map<std::string,
                       std::unordered_map<std::string,Func>> run = 
    {
      {"lreg", {{"acc", add_<lcdb::LREG, lcdb::Acc>},
                {"crs", add_<lcdb::LREG, lcdb::Crs>},
                {"bri", add_<lcdb::LREG, lcdb::Bri>},
                {"auc", add_<lcdb::LREG, lcdb::Auc>}}},
      {"nmc",  {{"acc", add_<lcdb::NMC, lcdb::Acc>},
                {"crs", add_<lcdb::NMC, lcdb::Crs>},
                {"bri", add_<lcdb::NMC, lcdb::Bri>},
                {"auc", add_<lcdb::NMC, lcdb::Auc>}}},
      {"nnc",  {{"acc", add_<lcdb::NNC, lcdb::Acc>},
                {"crs", add_<lcdb::NNC, lcdb::Crs>},
                {"bri", add_<lcdb::NNC, lcdb::Bri>},
                {"auc", add_<lcdb::NNC, lcdb::Auc>}}},
      {"ldc",  {{"acc", add_<lcdb::LDC, lcdb::Acc>},
                {"crs", add_<lcdb::LDC, lcdb::Crs>},
                {"bri", add_<lcdb::LDC, lcdb::Bri>},
                {"auc", add_<lcdb::LDC, lcdb::Auc>}}},
      {"qdc",  {{"acc", add_<lcdb::QDC, lcdb::Acc>},
                {"crs", add_<lcdb::QDC, lcdb::Crs>},
                {"bri", add_<lcdb::QDC, lcdb::Bri>},
                {"auc", add_<lcdb::QDC, lcdb::Auc>}}},
      {"lsvc", {{"acc", add_<lcdb::LSVC, lcdb::Acc>},
                {"crs", add_<lcdb::LSVC, lcdb::Crs>},
                {"bri", add_<lcdb::LSVC, lcdb::Bri>},
                {"auc", add_<lcdb::LSVC, lcdb::Auc>}}},
      {"esvc", {{"acc", add_<lcdb::ESVC, lcdb::Acc>},
                {"crs", add_<lcdb::ESVC, lcdb::Crs>},
                {"bri", add_<lcdb::ESVC, lcdb::Bri>},
                {"auc", add_<lcdb::ESVC, lcdb::Auc>}}},
      {"gsvc", {{"acc", add_<lcdb::GSVC, lcdb::Acc>},
                {"crs", add_<lcdb::GSVC, lcdb::Crs>},
                {"bri", add_<lcdb::GSVC, lcdb::Bri>},
                {"auc", add_<lcdb::GSVC, lcdb::Auc>}}},
      {"adab", {{"acc", add_<lcdb::ADAB, lcdb::Acc>},
                {"crs", add_<lcdb::ADAB, lcdb::Crs>},
                {"bri", add_<lcdb::ADAB, lcdb::Bri>},
                {"auc", add_<lcdb::ADAB, lcdb::Auc>}}},
      {"rfor", {{"acc", add_<lcdb::RFOR, lcdb::Acc>},
                {"crs", add_<lcdb::RFOR, lcdb::Crs>},
                {"bri", add_<lcdb::RFOR, lcdb::Bri>},
                {"auc", add_<lcdb::RFOR, lcdb::Auc>}}},
      {"dt",   {{"acc", add_<lcdb::DT, lcdb::Acc>},
                {"crs", add_<lcdb::DT, lcdb::Crs>},
                {"bri", add_<lcdb::DT, lcdb::Bri>},
                {"auc", add_<lcdb::DT, lcdb::Auc>}}},
      {"nb",   {{"acc", add_<lcdb::NB, lcdb::Acc>},
                {"crs", add_<lcdb::NB, lcdb::Crs>},
                {"bri", add_<lcdb::NB, lcdb::Bri>},
                {"auc", add_<lcdb::NB, lcdb::Auc>}}}
    };

       std::unordered_map<std::string,
                       std::unordered_map<std::string,contFunc>> contrun =

    {
      {"lreg", {{"acc", hptadd_<lcdb::LREG, lcdb::Acc,cont>},
                {"crs", hptadd_<lcdb::LREG, lcdb::Crs,cont>},
                {"bri", hptadd_<lcdb::LREG, lcdb::Bri,cont>},
                {"auc", hptadd_<lcdb::LREG, lcdb::Auc,cont>}}},
      {"ldc",  {{"acc", hptadd_<lcdb::LDC, lcdb::Acc,cont>},
                {"crs", hptadd_<lcdb::LDC, lcdb::Crs,cont>},
                {"bri", hptadd_<lcdb::LDC, lcdb::Bri,cont>},
                {"auc", hptadd_<lcdb::LDC, lcdb::Auc,cont>}}},
      {"qdc",  {{"acc", hptadd_<lcdb::QDC, lcdb::Acc,cont>},
                {"crs", hptadd_<lcdb::QDC, lcdb::Crs,cont>},
                {"bri", hptadd_<lcdb::QDC, lcdb::Bri,cont>},
                {"auc", hptadd_<lcdb::QDC, lcdb::Auc,cont>}}},
      {"lsvc", {{"acc", hptadd_<lcdb::LSVC, lcdb::Acc,cont>},
                {"crs", hptadd_<lcdb::LSVC, lcdb::Crs,cont>},
                {"bri", hptadd_<lcdb::LSVC, lcdb::Bri,cont>},
                {"auc", hptadd_<lcdb::LSVC, lcdb::Auc,cont>}}},
      {"esvc", {{"acc", hptadd_<lcdb::ESVC, lcdb::Acc,cont>},
                {"crs", hptadd_<lcdb::ESVC, lcdb::Crs,cont>},
                {"bri", hptadd_<lcdb::ESVC, lcdb::Bri,cont>},
                {"auc", hptadd_<lcdb::ESVC, lcdb::Auc,cont>}}},
      {"gsvc", {{"acc", hptadd_<lcdb::GSVC, lcdb::Acc,cont>},
                {"crs", hptadd_<lcdb::GSVC, lcdb::Crs,cont>},
                {"bri", hptadd_<lcdb::GSVC, lcdb::Bri,cont>},
                {"auc", hptadd_<lcdb::GSVC, lcdb::Auc,cont>}}}
    };


    // Check if algo exists in the map
    if (run.find(algo) != run.end() && !hpt) 
    {
      auto loss_map = run[algo];
      if (loss_map.find(loss) != loss_map.end()) 
          // Call the corresponding function
          loss_map[loss](id, algo, loss, seed, nreps, path);
      else 
          ERR("Not defined loss argument!");
    }
    else if (contrun.find(algo) != contrun.end() && hpt)
    {
      auto loss_map = contrun[algo];
      if (loss_map.find(loss) != loss_map.end()) 
          loss_map[loss](id, algo, loss, seed, nreps, path, lcdb::lambdas);
      else 
          ERR("Not defined loss argument!");
    }
    else 
        ERR("Not defined algo argument!");
}


int main(int argc, char* argv[]) 
{
  // Define default values
  const size_t DEFAULT_ID = 11;
  const std::string DEFAULT_ALGO = "nmc";
  const std::string DEFAULT_LOSS = "acc";
  const std::string DEFAULT_TYPE = "rands";
  const size_t DEFAULT_SEED = 24; // Kobeeee
  const size_t DEFAULT_NREPS = 1;
  const bool DEFAULT_HPT = false;

  // Initialize parameters with default values
  size_t id = DEFAULT_ID;
  std::string algo = DEFAULT_ALGO;
  std::string loss = DEFAULT_LOSS;
  std::string type = DEFAULT_TYPE;
  size_t seed = DEFAULT_SEED;
  size_t nreps = DEFAULT_NREPS;
  bool hpt = DEFAULT_HPT;

  arma::wall_clock timer;
  timer.tic();

  int i=1;
  try 
  {
      // Map to associate flags with actions
      std::map<std::string, std::function<void()>> flag_map = 
      {
        {"-id",    [&]() { id = std::strtoul(argv[++i], nullptr, 10); }},
        {"-i",     [&]() { id = std::strtoul(argv[++i], nullptr, 10); }},
        {"-algo",  [&]() { algo = argv[++i]; }},
        {"-a",     [&]() { algo = argv[++i]; }},
        {"-loss",  [&]() { loss = argv[++i]; }},
        {"-l",     [&]() { loss = argv[++i]; }},
        {"-type",  [&]() { type = argv[++i]; }},
        {"-t",     [&]() { type = argv[++i]; }},
        {"-seed",  [&]() { seed = std::strtoul(argv[++i], nullptr, 10); }},
        {"-s",     [&]() { seed = std::strtoul(argv[++i], nullptr, 10); }},
        {"-nreps", [&]() { nreps = std::strtoul(argv[++i], nullptr, 10); }},
        {"-n",     [&]() { nreps = std::strtoul(argv[++i], nullptr, 10); }},
        {"-hpt",   [&]() { 
            std::string hpt_str = argv[++i];
            std::transform(hpt_str.begin(), hpt_str.end(),
                           hpt_str.begin(),::tolower);

            if (hpt_str == "true" || hpt_str == "1") hpt = true;
            else if (hpt_str == "false" || hpt_str == "0") hpt = false;
            else 
              ERR("Invalid value for -hpt. Use 'true', 'false', '1', or '0'.");
        }},
        {"-h",   [&]() { 
            std::string hpt_str = argv[++i];
            std::transform(hpt_str.begin(), hpt_str.end(),
                           hpt_str.begin(), ::tolower);
            if (hpt_str == "true" || hpt_str == "1") hpt = true;
            else if (hpt_str == "false" || hpt_str == "0") hpt = false;
            else 
              ERR("Invalid value for -hpt. Use 'true', 'false', '1', or '0'.");
        }},
    };

      // Process arguments
      for (i = 1; i < argc; ++i) 
      {
        std::string arg = argv[i];
        if (flag_map.find(arg) != flag_map.end()) 
        {
          flag_map[arg]();  // Call the associated action for the flag
        } 
        else 
          ERR("Unknown flag: " + arg);
      }

      // Output configuration used
      LOG("Configuration used for the run...\n");
      LOG("OpenML dataset id  -> " << id << "\n");
      LOG("Algorithm          -> " << algo << "\n");
      LOG("Loss               -> " << loss << "\n");
      LOG("Type of curve gen  -> " << type << "\n");
      LOG("Seed               -> " << seed << "\n");
      LOG("Number of reps     -> " << nreps << "\n");
      LOG("HPT enabled        -> " << (hpt ? "Yes" : "No") << "\n");

      // Call the function to run the experiment
      if (type == "rands")
          rands(id, algo, loss, seed, nreps, hpt);
      else if (type == "add")
          add(id, algo, loss, seed, nreps, hpt);
      else if (type == "split")
          split(id, algo, loss, seed, nreps, hpt);
      else if (type == "boot")
          boot(id, algo, loss, seed, nreps, hpt);
      else 
          ERR("Invalid type for the experiment.\n");

      PRINT_TIME(timer.toc());
      return 0;
  } 
  catch (const std::invalid_argument& e)
  {
      ERR( "Error: " << e.what() );
      PRINT( "Usage: " << argv[0] << " [flags]\n");
      PRINT( "Available flags:");
      PRINT( "  -id <number>, -i <number>     : OpenML dataset ID (default: " 
            << DEFAULT_ID << ")");
      PRINT( "  -algo <string>, -a <string>   : Algorithm name (default: " 
          << DEFAULT_ALGO << ")");
      PRINT( "  -loss <string>, -l <string>   : Loss function (default: " 
          << DEFAULT_LOSS << ")");
      PRINT( "  -type <string>, -t <string>   : Type of curve generation" 
          << "(rands/add/split) (default: " << DEFAULT_TYPE << ")");
      PRINT("  -seed <number>, -s <number>   : Seed value (default: " 
          << DEFAULT_SEED << ")");
      PRINT("  -nreps <number>, -n <number>  : Number of repetitions (default: "
          << DEFAULT_NREPS << ")");
      PRINT("  -hpt <bool>                   : Enable hyperparameter tuning" 
          << "(true/false) (default: " << (DEFAULT_HPT ? "true" : "false") 
          << ")");
      PRINT_TIME(timer.toc());
      return 1;
  } 
  catch (const std::out_of_range& e) 
  {
      ERR("Error: Number out of range. " << e.what());
      PRINT_TIME(timer.toc());
      return 1;
  } 

}
