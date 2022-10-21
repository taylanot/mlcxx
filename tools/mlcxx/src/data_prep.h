
  
template<class T>
auto DataPrepare(const T& props)
{
  jem::util::Properties dataProps = props.findProps(DATA_PROP);
  jem::String type = "regression"; jem::String genfunc; 
  jem::String filename;  

  bool generate = dataProps.find(genfunc,   "genfunc");
  bool read     = dataProps.find(filename,  "filename");

  if ((type == "regression") && generate)
  { 
    int D = 1; int Ntrn = 10; int Ntst = 1000;

    double a = 1.; double p = 0.; double eps = 0.1;

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

    utils::data::regression::Dataset trainset(D, Ntrn, eps);
    utils::data::regression::Dataset testset(D, Ntst, eps);

    trainset.Generate(a, p, utils::to_string(genfunc));
    testset.Generate(a, p, utils::to_string(genfunc));
    
    return std::make_tuple(trainset, testset);
  }

}
