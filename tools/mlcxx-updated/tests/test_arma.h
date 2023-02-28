//-----------------------------------------------------------------------
//   test
//-----------------------------------------------------------------------
//#include <boost/preprocessor/repetition/repeat.hpp>

//#define DECL(z, n, text) text ## n = n;

void test_smpkr()
{
  int D, Ntrn, Ntst; D=1; Ntrn=4; Ntst=1000;
  double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
  utils::data::regression::Dataset trainset(D, Ntrn, eps);
  utils::data::regression::Dataset testset(D, Ntst, eps);

  trainset.Generate(a, p, "Sine");
  testset.Generate(a, p, "Sine");
  auto inputs = trainset.inputs;
  auto labels = arma::conv_to<arma::rowvec>::from(trainset.labels);
  algo::regression::SemiParamKernelRidge<mlpack::GaussianKernel,
                                         utils::data::regression::SineGen> 
                      model(inputs,labels, 0.5, size_t(5), 1.);

  arma::rowvec pred_labels;
  arma::mat test_inp = arma::sort(testset.inputs);
  model.Predict(test_inp, pred_labels);
  arma::mat test_out = arma::conv_to<arma::mat>::from(pred_labels);
  utils::Save("pred.csv",test_inp,test_out);
  

  //utils::data::regression::SineGen funcs(3);

  //arma::mat psi = funcs.Predict(inputs, "Phase");
  //std::cout << psi << std::endl;
  
  //arma::rowvec beta;
  //for(int i=0; i<10; i++)
  //{
  //  beta = arma::ones<arma::rowvec>(10)*i;
  //  std::cout << beta << std::endl;
  //}
  //  std::cout << beta << std::endl;
}

//-----------------------------------------------------------------------
//   test_combine
//-----------------------------------------------------------------------

void test_combine()
{
  //int D, Ntrn, Ntst; D=1; Ntrn=5000; Ntst=10000;
  //double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
  //utils::data::regression::Dataset trainset(D, Ntrn, eps);
  //utils::data::regression::Dataset testset(D, Ntst, eps);

  //trainset.Generate(a, p, "Sine");
  //testset.Generate(a, p, "Sine");
  //auto inputs = trainset.inputs;
  //auto inputs2 = testset.inputs;
  //auto cov = inputs.t() * inputs;
  //auto cov2 = inputs2.t() * inputs2;

  //jem::util::Timer timer;

  //timer.start();
  //auto inverse = arma::inv(cov); 
  //auto multip = cov.t() * cov;
  //std::cout << "***[ Total Elapsed Time : "  << timer.toDouble() << " ]***";
  //timer.start();
  //auto inverse2 = arma::inv(cov2); 
  //std::cout << "***[ Total Elapsed Time : "  << timer.toDouble() << " ]***";
  
  int N = 2; int M = 3;
  arma::mat K(N, N, arma::fill::ones);
  arma::mat psi(N,M, arma::fill::ones);
  arma::mat A = arma::join_rows(K, 2*psi); 
  arma::mat B = arma::join_cols(arma::join_rows(K, arma::zeros(N,M)),
                                                   arma::zeros(M,N+M)); 
  std::cout << A << std::endl;
  std::cout << B << std::endl;
}

//-----------------------------------------------------------------------
//   test_functional
//-----------------------------------------------------------------------

void test_functional()
{
  int D, Ntrn, Ntst, M, eps; D=1; Ntrn=100; Ntst=20; M=10; eps=0.;
  utils::data::regression::Dataset funcset(D, Ntrn, eps);
  utils::data::regression::Dataset testfuncset(D, Ntst, eps);
  funcset.Generate(M,"SineFunctional");
  testfuncset.Generate(M,"SineFunctional");

  //arma::mat labels = {{1,2,3,4,5},{2,4,6,8,10}};
  //arma::mat inputs = {{1,2,3,4,5}};
  
  arma::mat inputs = funcset.inputs;
  arma::mat testinputs = testfuncset.inputs;
  arma::mat labels = funcset.labels;

  arma::mat smooth_labels;
  arma::mat pred_inputs = testinputs;

  //arma::mat pred_inputs = arma::linspace(-10,10,1000);
  
  //arma::vec bandwidths = arma::linspace(0.001,0.5,100);
  
  smooth_labels = utils::functional::kernelsmoothing
                    <mlpack::GaussianKernel>
                      (inputs, labels, pred_inputs, 0.1);

  auto results = utils::functional::ufpca(pred_inputs, smooth_labels,0.99);
  
  utils::Save("inputs.csv",pred_inputs,std::get<1>(results)) ;
  arma::mat ali,ali1;
  ali1.load("inputs.csv",arma::csv_ascii);
  ali = utils::Load("inputs.csv");
}

//-----------------------------------------------------------------------
//   test_lc
//-----------------------------------------------------------------------

void test_lc()
{
  int D, Ntrn, Ntst; D=1; Ntrn=1000; Ntst=1000;
  double a, p, eps; a = 10.0; p = 0.; eps = 0.1;
  utils::data::regression::Dataset dataset(D, Ntrn, eps);

  dataset.Generate(a, p, "Sine");
  auto inputs = dataset.inputs;
  auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels);

  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,50);
  int repeat = 1000; double cv_valid = 0.2;

  LCurve<algo::regression::KernelRidge<mlpack::GaussianKernel>,
         mlpack::MSE> lcurve(Ns,repeat);
  std::string filename = "KR-1.csv";

  auto stats = lcurve.Generate(filename, inputs, labels, 0., 1.);

  arma::mat train_curves = lcurve.train_errors_;
  arma::mat test_curves = lcurve.train_errors_;
  
  //std::cout << train_curves << std::endl;
  //std::cout << test_curves << std::endl;
}

//-----------------------------------------------------------------------
// lrlc
//-----------------------------------------------------------------------

void lrlc(bool tune)
{
  int D, N; D=50; N = 1000;
  double a, p, eps; a = 1.0; p = 0.; eps = 1.;
  utils::data::regression::Dataset dataset(D, N, eps);

  dataset.Generate(a, p, "Linear");
  auto inputs = dataset.inputs;
  auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels);

  arma::irowvec Ns = arma::regspace<arma::irowvec>(5,1,100);
  int repeat = 1000; double cv_valid = 0.2;
  
  std::string filename, rawfilename, dir, rawdir;

  arma::rowvec lambdas = arma::linspace<arma::rowvec>(0.,1.,100);
  //arma::rowvec lambdas = arma::linspace<arma::rowvec>(0.,10.,100);
  if (!tune)
  {
    dir = "50D-extra/LR-LearningCurves/notune";
    rawdir = "50D-extra/LR-LearningCurves/notune/raw";
    utils::create_dirs(dir);
    utils::create_dirs(rawdir);
    for(int i=0; i < lambdas.n_cols; i++)
    {
    LCurve<mlpack::LinearRegression,
           mlpack::MSE> lcurve(Ns,repeat);

    filename =  dir+"/LR-"+std::to_string(i)+".csv";
    rawfilename =  rawdir+"/LR-"+std::to_string(i)+".csv";
    lcurve.Generate(filename, inputs, labels, double(lambdas(i)));

    utils::Save(rawfilename,lcurve.train_errors_, lcurve.test_errors_);
    }
  }
  else
  {
    dir = "50D-extra/LR-LearningCurves/tune";
    rawdir = "50D-extra/LR-LearningCurves/tune/raw";
    utils::create_dirs(dir);
    utils::create_dirs(rawdir);
    LCurve_HPT<mlpack::LinearRegression,
           mlpack::MSE,
           mlpack::SimpleCV> lcurve(Ns,repeat,cv_valid);

    filename =  dir+"/LR-tuned.csv";
    rawfilename =  rawdir+"/LR-tuned.csv";

    lcurve.Generate(filename, inputs, labels, lambdas);

    utils::Save(rawfilename,lcurve.train_errors_, lcurve.test_errors_);

  }
}

//-----------------------------------------------------------------------
//  genlrlc
//-----------------------------------------------------------------------

void genlrlc()
{
  lrlc(false);
  lrlc(true);
}

//-----------------------------------------------------------------------
// DataAnalytics
//-----------------------------------------------------------------------
void DataAnalytics(arma::mat data) 
{
  arma::inplace_trans(data);

  std::cout << "mean    : " << arma::mean(data) << std::endl;
  std::cout << "median  : " << arma::median(data) << std::endl;
  std::cout << "stddev  : " << arma::stddev(data) << std::endl;
  std::cout << "correl  : " << "\n"  << arma::cor(data) << std::endl;
  std::cout << "max     : " << arma::max(data) << std::endl;
  std::cout << "min     : " << arma::min(data) << std::endl;
}

//-----------------------------------------------------------------------
// read_data
//-----------------------------------------------------------------------
void read_data()
{
  arma::mat matrix;
  mlpack::data::DatasetInfo info;
  mlpack::data::Load("datasets/iris.csv", matrix, info,false,true);
  DataAnalytics(matrix);
}
