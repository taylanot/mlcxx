/**
 * @file main.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for lcdb++ experiments
 *
 */

#define DTYPE double  

#include <headers.h>
#include "lcdb++/config.h"

template<class MODEL,class LOSS>
int boot_( const size_t id,
           const std::string& algo, const std::string& loss,
           const size_t seed, const size_t nreps,
           const std::filesystem::path& path ) 
{
  data::classification::oml::Dataset dataset(id);

  mlpack::RandomSeed(seed);
  const arma::irowvec Ns = arma::regspace<arma::irowvec>
                                              (1,1,size_t(dataset.size_*0.9));

  src::LCurve<MODEL,LOSS> lcurve(Ns,nreps,true,true);
  lcurve.Bootstrap(dataset.inputs_,dataset.labels_,
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
 
  data::classification::oml::Dataset dataset(id);

  mlpack::RandomSeed(seed);
  const arma::irowvec Ns = arma::regspace<arma::irowvec>
                                              (10,1,size_t(dataset.size_*0.9));

  src::LCurveHPT<MODEL,LOSS> lcurve(Ns,nreps,lcdb::vsize, true,true);
  lcurve.Bootstrap(dataset.inputs_,dataset.labels_,
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
  data::classification::oml::Dataset dataset(id);

  mlpack::RandomSeed(seed);

  const arma::irowvec Ns = arma::regspace<arma::irowvec>
    (1,1,size_t(dataset.size_*0.9));

  src::LCurve<MODEL,LOSS> lcurve(Ns,nreps,true,true);
  lcurve.Additive(dataset.inputs_,dataset.labels_,
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
 
  data::classification::oml::Dataset dataset(id);

  mlpack::RandomSeed(seed);

  const arma::irowvec Ns = arma::regspace<arma::irowvec>
    (10,1,size_t(dataset.size_*0.9));


  src::LCurveHPT<MODEL,LOSS> lcurve(Ns,nreps,lcdb::vsize, true,true);
  lcurve.Additive(dataset.inputs_,dataset.labels_,
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
  data::classification::oml::Dataset dataset(id);

  mlpack::RandomSeed(seed);
  data::classification::oml::Dataset trainset,testset;

  data::StratifiedSplit(dataset,trainset,testset,lcdb::splitsize);

  const arma::irowvec Ns = arma::regspace<arma::irowvec>
    (1,1,size_t(trainset.size_*0.9));

  src::LCurve<MODEL,LOSS> lcurve(Ns,nreps,true,true);
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
 
  data::classification::oml::Dataset dataset(id);

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

void boot( size_t id, const std::string& algo, const std::string& loss,
           size_t seed, size_t nreps,  bool hpt )
{
    std::filesystem::path path;
    if (!hpt)
      path = EXP_PATH/"lcdb++"/"boot"/"ntune";
    else
      path = EXP_PATH/"lcdb++"/"boot"/"tune";

    path = path/std::to_string(id)/algo/std::to_string(seed);
    std::filesystem::create_directories(path);

    using cont = arma::Row<DTYPE>;
    /* using disc = arma::Row<size_t>; */

    using BootFunc = std::function<void(const size_t id,
                                        const std::string&,
                                        const std::string&,
                                        const size_t,
                                        const size_t,
                                        const std::filesystem::path&)>;

    using contBootFunc = std::function<void(const size_t id,
                                            const std::string&,
                                            const std::string&,
                                            const size_t,
                                            const size_t,
                                            const std::filesystem::path&, 
                                            const cont&)>;

    /* using discBootFunc = std::function<void(const size_t id, */
    /*                                         const std::string&, */
    /*                                         const std::string&, */
    /*                                         const size_t, */
    /*                                         const size_t, */
    /*                                         const std::filesystem::path&, */ 
    /*                                         const disc&)>; */

    // Mapping of algo and loss types
    std::unordered_map<std::string,
                       std::unordered_map<std::string,BootFunc>> run = 
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
                       std::unordered_map<std::string,contBootFunc>> contrun =

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
          // Call the corresponding boot function
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
      path = EXP_PATH/"lcdb++"/"split"/"ntune";
    else
      path = EXP_PATH/"lcdb++"/"split"/"tune";

    path = path/std::to_string(id)/algo/std::to_string(seed);
    std::filesystem::create_directories(path);

    using cont = arma::Row<DTYPE>;
    /* using disc = arma::Row<size_t>; */

    using BootFunc = std::function<void(const size_t id,
                                        const std::string&,
                                        const std::string&,
                                        const size_t,
                                        const size_t,
                                        const std::filesystem::path&)>;

    using contBootFunc = std::function<void(const size_t id,
                                            const std::string&,
                                            const std::string&,
                                            const size_t,
                                            const size_t,
                                            const std::filesystem::path&, 
                                            const cont&)>;

    /* using discBootFunc = std::function<void(const size_t id, */
    /*                                         const std::string&, */
    /*                                         const std::string&, */
    /*                                         const size_t, */
    /*                                         const size_t, */
    /*                                         const std::filesystem::path&, */ 
    /*                                         const disc&)>; */

    // Mapping of algo and loss types
    std::unordered_map<std::string,
                       std::unordered_map<std::string,BootFunc>> run = 
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
                       std::unordered_map<std::string,contBootFunc>> contrun =

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
          // Call the corresponding boot function
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
      path = EXP_PATH/"lcdb++"/"add"/"ntune";
    else
      path = EXP_PATH/"lcdb++"/"add"/"tune";

    path = path/std::to_string(id)/algo/std::to_string(seed);
    std::filesystem::create_directories(path);

    using cont = arma::Row<DTYPE>;
    /* using disc = arma::Row<size_t>; */

    using BootFunc = std::function<void(const size_t id,
                                        const std::string&,
                                        const std::string&,
                                        const size_t,
                                        const size_t,
                                        const std::filesystem::path&)>;

    using contBootFunc = std::function<void(const size_t id,
                                            const std::string&,
                                            const std::string&,
                                            const size_t,
                                            const size_t,
                                            const std::filesystem::path&, 
                                            const cont&)>;

    /* using discBootFunc = std::function<void(const size_t id, */
    /*                                         const std::string&, */
    /*                                         const std::string&, */
    /*                                         const size_t, */
    /*                                         const size_t, */
    /*                                         const std::filesystem::path&, */ 
    /*                                         const disc&)>; */

    // Mapping of algo and loss types
    std::unordered_map<std::string,
                       std::unordered_map<std::string,BootFunc>> run = 
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
                       std::unordered_map<std::string,contBootFunc>> contrun =

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
          // Call the corresponding boot function
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
    size_t default_id = 11;
    std::string default_algo = "nmc";
    std::string default_loss = "acc";
    std::string default_type = "boot";
    size_t default_seed = SEED ;
    size_t default_nreps = 1;
    bool default_hpt = false;

    // Initialize parameters with default values
    size_t id = default_id;
    std::string algo = default_algo;
    std::string loss = default_loss;
    std::string type = default_type;
    size_t seed = default_seed;
    size_t nreps = default_nreps;
    bool hpt = default_hpt;

  arma::wall_clock timer;
  timer.tic();

  int args_to_process = std::min(argc - 1, 7);

  try 
  {
      if (args_to_process >= 1) 
          id = std::strtoul(argv[1], nullptr, 10);  
      if (args_to_process >= 2) 
          algo = argv[2]; 
      if (args_to_process >= 3) 
          loss = argv[3]; 
      if (args_to_process >= 4) 
          type = argv[4]; 
      if (args_to_process >= 5) 
          seed = std::strtoul(argv[5], nullptr, 10); 
      if (args_to_process >= 6) 
          nreps = std::strtoul(argv[6], nullptr, 10); 
      if (args_to_process >= 7) 
      {
        std::string hpt_str = argv[7];
        // Convert to lower case for case-insensitive comparison
        std::transform(hpt_str.begin(), hpt_str.end(),
                                        hpt_str.begin(), ::tolower);
        if (hpt_str == "true" || hpt_str == "1") 
          hpt = true;
        else if (hpt_str == "false" || hpt_str == "0") 
          hpt = false;
        else 
          throw std::invalid_argument(
              "Invalid value for hpt. Use 'true', 'false', '1', or '0'.");
      }

      // Optionally, you can notify the user about using default values
      if (argc - 1 < 7) 
      {
        LOG("Some arguments were not provided."
            << "Using default values where applicable.\n");
      }

      LOG("Configuration used for the run...\n");
      LOG("OpenML dataset id  -> " << id << "\n");
      LOG("Algorithm          -> " << algo << "\n");
      LOG("Loss               -> " << loss << "\n");
      LOG("Type of curve gen  -> " << type << "\n");
      LOG("Seed               -> " << seed << "\n");
      LOG("Number of reps     -> " << nreps << "\n");
      LOG("HPT enabled        -> " << (hpt ? "Yes" : "No")<< "\n");
      
      
      // Call the function to run the experiment

      if (type == "boot")
        boot(id,algo,loss,seed,nreps,hpt);
      else if (type == "add")
        add(id,algo,loss,seed,nreps,hpt);
      else if (type == "split")
        split(id,algo,loss,seed,nreps,hpt);
      else 
        ERR("Invalid type for the experiment.");

      PRINT_TIME(timer.toc());
      return 0;
  } 

  catch (const std::invalid_argument& e)
  {
    std::cerr << "Error: Invalid argument provided. " << e.what() << "\n";
    std::cerr << "Usage: " << argv[0] 
                          << " [<id>] [<algo>] [<seed>] [<nreps>] [<hpt>]\n";
    std::cerr << "Default values will be used"
              << " for any missing or invalid arguments.\n";
    PRINT_TIME(timer.toc());
    return 1;
  } 
  catch (const std::out_of_range& e) 
  {
    std::cerr << "Error: Number out of range. " << e.what() << "\n";
    PRINT_TIME(timer.toc());
    return 1;
  } 
  catch (...) 
  {
    std::cerr << "An unexpected error occurred while parsing arguments.\n";
    PRINT_TIME(timer.toc());
    return 1;
  }

}

