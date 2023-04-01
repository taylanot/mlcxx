/**
 * @file main.cpp
 * @author Ozgur Taylan Turan
 *
 * Main file of mlcxx where you do not have to do anything...
 */

#include <headers.h>

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------
//class FOO
//{
//  public:
//  
//  FOO(arma::mat& X, arma::vec& y) : X(X), y(y) { }
//  
//  double Evaluate (const arma::mat& theta)
//  {
//    const arma::vec res = Residual(theta);
//    return arma::dot(res,res);
//  }
//  void Gradient ( const arma::mat& theta, arma::mat& gradient )
//  {
//    const arma::vec res = Residual(theta);
//    const arma::vec dfdp1 = arma::exp(-X.t() * theta(1,0));
//    const arma::mat dfdp2 = -theta(0,0)*arma::exp(-X.t() * theta(1,0))%X.t();
//    const arma::vec dfdp3(X.n_cols, arma::fill::ones);
//    gradient = arma::join_cols(dfdp1.t()*res, dfdp2.t()*res,dfdp3.t()*res);
//    gradient *= 2;
//  }
//  //double EvaluateWithGradient(const arma::mat& theta, arma::mat& gradient)
//  //{
//  //  const arma::vec tmp = X.t() * theta - y;
//  //  gradient = 2 * X * tmp;
//  //  return arma::dot(tmp,tmp);
//  //}
//  arma::vec Residual(const arma::mat& theta)
//  {
//    return (theta(0,0)*arma::exp(-X.t() * theta(1,0))+theta(2,0)) - y;
//  }
//  
//  private:
//  
//  const arma::mat& X;
//  const arma::vec& y;
//};

const int  SEED = 24 ; // KOBEEEE

int main ( int argc, char** argv )
{
  {
    int D, N;
    double a, p, eps;
    std::string type;

    D = 1; N = 4; type = "Sine";
    a = 1.0; p = 0.; eps = 0.0;
    utils::data::regression::Dataset dataset(D, N);

    dataset.Generate(a, p, type, eps);

    auto inputs = dataset.inputs_;
    auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

    algo::regression::GP<mlpack::GaussianKernel> model(inputs,labels, 0.,0.1);
    arma::mat labprior,labposterior;
    arma::mat inp = arma::regspace<arma::mat>(0,1,N);
    size_t k=2;
    model.SamplePrior(k,inp.t(),labprior);
    model.SamplePosterior(k,inp.t(),labposterior);
    //PRINT(labprior);
    //PRINT(labposterior);
  }
  //{ 
  //  int n_dims   = 1;
  //  int params   = 1; int n_points = 100; 

  //  arma::mat X(n_dims, n_points, arma::fill::randu);
  //  X += arma::randn(n_dims, n_points)*0.5;
  //  arma::vec p = {1,1,0};
  //  arma::vec y = p(0)*arma::exp(-X.t()*p(1))+p(2); //* 2 + 3;
  //  
  //  
  //  FOO obj(X, y);
  //  
  //  // create a Limited-memory BFGS optimizer object with default parameters
  //  ens::L_BFGS opt;
  //  opt.MaxIterations() = 10000;

  //  arma::vec theta(3);
  //  for (int i=0; i < 10; i++)
  //  {
  //  // initial point (uniform random)
  //  
  //  theta.randn()*20;
  //  
  //  opt.Optimize(obj, theta, ens::Report());
  //  } 
  //  // theta now contains the optimized parameters
  //  theta.print("theta:");
  //} 
  //arma::rowvec ali = arma::regspace<arma::rowvec>(0,1,10);
  //PRINT(ali.t() * ali);
  //{
  //  arma::rowvec a = {1,2,3};
  //  arma::mat veli = (a.t() * a);
  //  PRINT( veli );
  //}
  //{
  //{
    //arma::vec a = {-10};
    //arma::vec b = {5};
    ////double distance = mlpack::SquaredEuclideanDistance::Evaluate(a,b);
    ////PRINT_VAR(distance);
    //mlpack::EuclideanDistance metric;
    //PRINT(metric.Evaluate(a,b));
  //arma::vec ali = {0,0,0,1,2,3,4,5};
  //ali.elem(find(ali==0)) = 10.;
  //PRINT(ali);
  //}
  //{
  //    int D = 2; int N = 10000; int Nc = 2;

  //    utils::data::classification::Dataset data(D, N, Nc);
  //    double r = 10; double eps =0.01;
  //    data._dipping(r, eps);
  //    data.Save("classification.csv");

  //}
  //{
  //  auto res = create_saw(100 , 0.1);
  //  PRINT(std::get<0>(res));
  //}
  //{
  //  arma::rowvec Ns = arma::regspace<arma::rowvec>(5,1,100);
  //  auto res = create_saw(Ns, 0.1);
  //  PRINT(res);
  //}
  doctest::Context context;

  // !!! THIS IS JUST AN EXAMPLE SHOWING HOW DEFAULTS/OVERRIDES ARE SET !!!

  // defaults
  //context.addFilter("test-case-exclude", "*math*"); 
  context.setOption("abort-after", 5);
  context.setOption("order-by", "file");

  context.applyCommandLine(argc, argv);

  // overrides
  context.setOption("no-breaks", true);

  int res = context.run(); 

  if(context.shouldExit()) 
      return res;          
  
  jem::String log_file;

  jem::util::Timer timer;

  if ( argc > 1 ) 
  {
    std::filesystem::path path(argv[1]); 

    int seed = SEED;

    timer.start();

    mlpack::RandomSeed(seed);
   // arma::arma_rng::set_seed(seed);
    if ( path.extension() == ".pro" )
    {

      jem::util::Properties props;
      props.parseFile(argv[1]);

      props.find(seed, "seed");

      mlpack::RandomSeed(seed);

      input_run(props);
    }
    else
    {
      func_run(argv[1]);
    }

    std::cout.put('\n');
    PRINT_SEED( seed ); 
    std::cout.put('\n');
    std::cout.put('\n');
    PRINT_TIME( timer.toDouble() );
    std::cout.put('\n');
  }

  return 0; 
}
