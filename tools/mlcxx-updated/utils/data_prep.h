/**
 * @file data_prep.h
 * @author Ozgur Taylan Turan
 *
 * Function that prepares data from input file arguments
 *
 *
 */

#ifndef DATA_PREP_H
#define DATA_PREP_H

  
auto DataPrepare(jem::util::Properties& dataProps)
{
  
  jem::String type = "regression"; jem::String genfunc; 
  jem::String filename;  

  bool generate = dataProps.find(genfunc,   "genfunc");
  bool read     = dataProps.find(filename,  "filename");
  utils::data::regression::Dataset trainset;
  utils::data::regression::Dataset testset;

  int D = 1; int Ntrn = 10; int Ntst = 1000;
  double a = 1.; double p = 0.; double eps = 0.1;

  if ((type == "regression") && generate)
  { 
    dataProps.find(D,     "D");
    dataProps.find(Ntrn,  "Ntrn");
    dataProps.find(Ntst,  "Ntst");
    dataProps.find(eps,   "eps");
    dataProps.find(a,     "scale");
    dataProps.find(p,     "phase");
    dataProps.find(type,  "type");

    std::cout << "Generating Data..." << std::endl;
    std::cout << "  - D     : " << D  << "\n"
              << "  - Ntrn  : " << Ntrn << "\n" 
              << "  - Ntst  : " << Ntst << "\n" 
              << "  - Scale : " << a << "\n" 
              << "  - Phase : " << p << "\n" 
              << "  - Noise : " << eps << "\n" 
              << "  - From  : " << utils::to_string(genfunc) << std::endl;

    trainset.Set(D, Ntrn);
    testset.Set(D, Ntst);

    trainset.Generate(a, p, utils::to_string(genfunc), eps);
    testset.Generate(a, p, utils::to_string(genfunc), eps);
    
    return std::make_tuple(trainset, testset);
  }
  else if (read)
  {
    utils::data::regression::Dataset trainset(D, Ntrn);
    utils::data::regression::Dataset testset(D, Ntst);

    trainset.Generate(a, p, utils::to_string(genfunc), eps);
    testset.Generate(a, p, utils::to_string(genfunc), eps);
    

    return std::make_tuple(trainset, testset);
  }

  return std::make_tuple(trainset, testset);
}

#endif
