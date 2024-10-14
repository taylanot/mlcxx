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
                                              (1,1,size_t(dataset.size_*0.9));

  src::LCurveHPT<MODEL,LOSS> lcurve(Ns,nreps,lcdb::vsize, true,true);
  lcurve.Bootstrap(dataset.inputs_,dataset.labels_,
                   arma::unique(dataset.labels_).eval().n_elem, args...);
  lcurve.GetResults().save(path/(loss+".csv"),arma::csv_ascii);
 
  return 0;
}
void boot( size_t id,
           const std::string& algo, const std::string& loss,
           size_t seed, size_t nreps, bool hpt ) 
{
  std::filesystem::path path;
  if (!hpt)
    path = EXP_PATH/"lcdb++"/"boot"/"ntune";
  else
    path = EXP_PATH/"lcdb++"/"boot"/"tune";

  path = path/std::to_string(id)/algo/std::to_string(seed);
  std::filesystem::create_directories(path);
 
  if (!algo.compare("lreg"))
  {
    if (!loss.compare("acc"))
      boot_<lcdb::LREG,lcdb::Acc> (id,algo,loss,seed,nreps,path);
    else if (!loss.compare("crs"))
      boot_<lcdb::LREG,lcdb::Crs> (id,algo,loss,seed,nreps,path);
    else
      ERR("Not defined loss argument!");
  }

  else if (!algo.compare("nmc"))
  {
    if (!loss.compare("acc"))
      boot_<lcdb::NMC,lcdb::Acc> (id,algo,loss,seed,nreps,path);
    else if (!loss.compare("crs"))
      boot_<lcdb::NMC,lcdb::Crs> (id,algo,loss,seed,nreps,path);
    else
      ERR("Not defined loss argument!");
  
  }

  else if (!algo.compare("ldc"))
  {
    if (!loss.compare("acc"))
      boot_<lcdb::LDC,lcdb::Acc> (id,algo,loss,seed,nreps,path);
    else if (!loss.compare("crs"))
      boot_<lcdb::LDC,lcdb::Crs> (id,algo,loss,seed,nreps,path);
    else
      ERR("Not defined loss argument!");
  
  }

  else if (!algo.compare("qdc"))
  {
    if (!loss.compare("acc"))
      boot_<lcdb::QDC,lcdb::Acc> (id,algo,loss,seed,nreps,path);
    else if (!loss.compare("crs"))
      boot_<lcdb::QDC,lcdb::Crs> (id,algo,loss,seed,nreps,path);
    else
      ERR("Not defined loss argument!");
  
  }

  else if (!algo.compare("nb"))
  {
    if (!loss.compare("acc"))
      boot_<lcdb::NB,lcdb::Acc> (id,algo,loss,seed,nreps,path);
    else if (!loss.compare("crs"))
      boot_<lcdb::NB,lcdb::Crs> (id,algo,loss,seed,nreps,path);
    else
      ERR("Not defined loss argument!");
  
  }

  else if (!algo.compare("lsvc"))
  {
    if (!loss.compare("acc"))
      boot_<lcdb::LSVC,lcdb::Acc> (id,algo,loss,seed,nreps,path);
    else if (!loss.compare("crs"))
      boot_<lcdb::LSVC,lcdb::Crs> (id,algo,loss,seed,nreps,path);
    else
      ERR("Not defined loss argument!");
  }

  else if (!algo.compare("gsvc"))
  {
    if (!loss.compare("acc"))
      boot_<lcdb::GSVC,lcdb::Acc> (id,algo,loss,seed,nreps,path);
    else if (!loss.compare("crs"))
      boot_<lcdb::GSVC,lcdb::Crs> (id,algo,loss,seed,nreps,path);
    else
      ERR("Not defined loss argument!");
  }

  else if (!algo.compare("esvc"))
  {
    if (!loss.compare("acc"))
      boot_<lcdb::ESVC,lcdb::Acc> (id,algo,loss,seed,nreps,path);
    else if (!loss.compare("crs"))
      boot_<lcdb::ESVC,lcdb::Crs> (id,algo,loss,seed,nreps,path);
    else
      ERR("Not defined loss argument!");
  }

  else 
    ERR("Not defined algo argument!");   
}

template<class MODEL,class LOSS>
int split_( size_t id,
            const std::string& algo, const std::string& loss,
            size_t seed, size_t nreps, bool hpt ) 
{
  std::filesystem::path path;
  if (!hpt)
    path = EXP_PATH/"lcdb++"/"split"/"ntune";
  else
    path = EXP_PATH/"lcdb++"/"split"/"tune";
    
  path = path/std::to_string(id)/algo/std::to_string(seed);
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);
  
  data::classification::oml::Dataset dataset(id);

  mlpack::RandomSeed(seed);
  data::classification::oml::Dataset trainset,testset;

  data::StratifiedSplit(dataset,trainset,testset,0.2);

  const arma::irowvec Ns = arma::regspace<arma::irowvec>
    (1,1,size_t(trainset.size_*0.9));

  src::LCurve<MODEL,LOSS> lcurve(Ns,nreps,true,false,true);
  lcurve.Split(trainset,testset,arma::unique(dataset.labels_).eval().n_elem);
  lcurve.test_errors_.save(path/(loss+".csv"),arma::csv_ascii);
 
  return 0;
}

template<class MODEL,class LOSS>
int add_( size_t id,
          const std::string& algo, const std::string& loss,
          size_t seed, size_t nreps, bool hpt ) 
{
  std::filesystem::path path;
  if (!hpt)
    path = EXP_PATH/"lcdb++"/"add"/"ntune";

  else
    path = EXP_PATH/"lcdb++"/"add"/"tune";
    
  path = path/std::to_string(id)/algo/std::to_string(seed);
  std::filesystem::create_directories(path);
  /* std::filesystem::current_path(path); */
  
  data::classification::oml::Dataset dataset(id);

  mlpack::RandomSeed(seed);
  data::classification::oml::Dataset trainset,testset;

  data::StratifiedSplit(dataset,trainset,testset,0.2);

  const arma::irowvec Ns = arma::regspace<arma::irowvec>
    (100,1,size_t(trainset.size_*0.9));

  src::LCurve<MODEL,LOSS> lcurve(Ns,nreps,true,false,true);
  lcurve.Additive(trainset,testset,arma::unique(dataset.labels_).eval().n_elem);
  lcurve.test_errors_.save(path/(loss+".csv"),arma::csv_ascii);
 
  return 0;
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

      LOG("Configuration Used for the run...\n");
      LOG("OpenML dataset id  -> " << id << "\n");
      LOG("Algorithm          -> " << algo << "\n");
      LOG("Loss               -> " << loss << "\n");
      LOG("Type of curve gen  -> " << type << "\n");
      LOG("Seed               -> " << seed << "\n");
      LOG("Number of reps     -> " << nreps << "\n");
      LOG("HPT enabled        -> " << (hpt ? "Yes" : "No")<< "\n");
      
      
      // Call the function to run the experiment

        boot(id,algo,loss,seed,nreps,hpt);
        /* ERR("Invalid type for the experiment."); */

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

