/**
 * @file test_algo.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_ALGO_H 
#define TEST_ALGO_H

TEST_SUITE("FUNCTIONALPCA") {
    TEST_CASE("UFPCA")
  {
    int D, Ntrn, Ntst, M, eps;
    D=1; Ntrn=100; Ntst=20; M=10; eps=0.;
    utils::data::regression::Dataset funcset(D, Ntrn);
    utils::data::regression::Dataset testfuncset(D, Ntst);

    funcset.Generate(M,"SineFunctional", eps);
    testfuncset.Generate(M,"SineFunctional", eps);
    
    arma::mat inputs = funcset.inputs_;
    arma::mat testinputs = testfuncset.inputs_;
    arma::mat labels = funcset.labels_;

    arma::mat smooth_labels;
    arma::mat pred_inputs = testinputs;
    
    smooth_labels = algo::functional::kernelsmoothing
                      <mlpack::GaussianKernel>
                        (inputs, labels, pred_inputs, 0.1);

    SUBCASE("KERNELSMOOTHING")
    {
      CHECK ( smooth_labels.n_rows == M );
      CHECK ( smooth_labels.n_cols == Ntst );
    }
    SUBCASE("PCA")
    {
      size_t npc = 2;
      auto results = algo::functional::ufpca(pred_inputs, smooth_labels,npc);
      CHECK (std::get<0>(results).n_rows == npc );
      CHECK (std::get<1>(results).n_rows == npc );
      CHECK (std::get<1>(results).n_cols == Ntst );
    }
  }
}
TEST_SUITE("SEMIPARAMETRICKERNELRIDGE") {
  TEST_CASE("PROBLEMS")
  {
    int D, N; double tol = 1e-6;
    double a, p, eps;
    std::string type;

    SUBCASE("1D-Sine")
    {
      D = 1; N = 10; type = "Sine";
      a = 1.0; p = 0.; eps = 0.0;
      utils::data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);      

      algo::regression::SemiParamKernelRidge<mlpack::GaussianKernel,
                                             utils::data::regression::SineGen> 
                          model(inputs,labels, 0.5, size_t(5), 1.);
      double error = model.ComputeError(inputs, labels);
      CHECK ( error <= tol );
    }  
  }
}


TEST_SUITE("KERNELRIDGE") {
  TEST_CASE("PROBLEMS")
  {
    int D, N; double tol = 1e-6;
    double a, p, eps;
    std::string type;

    SUBCASE("1D-Sine")
    {
      D = 1; N = 10; type = "Sine";
      a = 1.0; p = 0.; eps = 0.0;
      utils::data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

      algo::regression::KernelRidge<mlpack::GaussianKernel>
                                                 model(inputs,labels, 0.,0.1);
      double error = model.ComputeError(inputs, labels);
      CHECK ( error <= tol );
    }
    SUBCASE("2D-Sine")
    {
      D = 2; N = 10; type = "Sine";
      a = 1.0; p = 0.; eps = 0.0;
      utils::data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

      algo::regression::KernelRidge<mlpack::GaussianKernel>
                                                model(inputs,labels, 0.,0.1);
      double error = model.ComputeError(inputs, labels);
      CHECK ( error <= tol );
    }
  }
}
TEST_SUITE("GAUSSIANPROCESSREGRESSION") {

  int D, N; double tol = 1e-6;
  double a, p, eps;
  std::string type;
  size_t Ngrid = 10;
  TEST_CASE("PROBLEMS")
  {
        SUBCASE("1D-Sine")
    {
      D = 1; N = 10; type = "Sine";
      a = 1.0; p = 0.; eps = 0.0;
      utils::data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

      algo::regression::GaussianProcess<mlpack::GaussianKernel> model(inputs,labels, 0.,0.1);
      double error = model.ComputeError(inputs, labels);
      double ll = model.LogLikelihood(inputs, labels);
      CHECK ( error <= tol );
      CHECK ( ll <= 0. );
    }
    SUBCASE("2D-Sine")
    {
      D = 2; N = 10; type = "Sine";
      a = 1.0; p = 0.; eps = 0.0;
      utils::data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

      algo::regression::GaussianProcess<mlpack::GaussianKernel>
                                                 model(inputs,labels, 0.,0.1);
      double error = model.ComputeError(inputs, labels);
      double ll = model.LogLikelihood(inputs, labels);
      CHECK ( error <= tol );
      CHECK ( ll <= 0. );
    }
  }
  TEST_CASE("SAMPLING")
  {
    SUBCASE("1D")
    {
      D = 1; N = 4; type = "Sine";
      a = 1.0; p = 0.; eps = 0.0;
      size_t k=2;
      utils::data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

      algo::regression::GaussianProcess<mlpack::GaussianKernel> 
                                                  model(inputs,labels, 0.,0.1);
      arma::mat labprior,labposterior;
      arma::mat inp = arma::randu(D,Ngrid);
      model.SamplePrior(k,inp,labprior);
      model.SamplePosterior(k,inp,labposterior);
      CHECK ( labprior.n_cols == Ngrid );
      CHECK ( labprior.n_rows== k );
      CHECK ( labposterior.n_cols == Ngrid );
      CHECK ( labposterior.n_rows== k );
    }
    SUBCASE("2D")
    {
      D = 2; N = 10; type = "Sine";
      a = 1.0; p = 0.; eps = 0.0;
      size_t k=2;
      utils::data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

      algo::regression::GaussianProcess<mlpack::GaussianKernel> 
                                                  model(inputs,labels, 0.,0.1);
      arma::mat labprior,labposterior;
      arma::mat inp = arma::randu(D,Ngrid);
      model.SamplePrior(k,inp,labprior);
      model.SamplePosterior(k,inp,labposterior);
      CHECK ( labprior.n_cols == Ngrid );
      CHECK ( labprior.n_rows== k );
      CHECK ( labposterior.n_cols == Ngrid );
      CHECK ( labposterior.n_rows== k );
    }
  }
}

TEST_SUITE("NEARESTMEANCLASSIFIER") {
  TEST_CASE("PROBLEMS")
  {
    int D, N, Nc; //double tol = 1e-2;
    std::string type;

    SUBCASE("1D-Simple")
    {
      D = 1; N = 10000; Nc = 2; type = "Simple";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_);
      double acc = model.ComputeAccuracy(data.inputs_, data.labels_);
      CHECK ( acc == 100.  );
    }
    SUBCASE("2D-Simple")
    {
      D = 2; N = 10000; Nc = 2; type = "Simple";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_);
      double acc = model.ComputeAccuracy(data.inputs_, data.labels_);
      CHECK ( acc == 100. );
    }
    SUBCASE("1D-Hard")
    {
      D = 1; N = 10000; Nc = 2; type = "Hard";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_);
      double acc = model.ComputeAccuracy(data.inputs_, data.labels_);
      CHECK ( acc >= 0.9 );
    }
    SUBCASE("2D-Hard")
    {
      D = 2; N = 10000; Nc = 2; type = "Hard";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_);
      double acc = model.ComputeAccuracy(data.inputs_, data.labels_);
      CHECK ( acc >= 0.9 );
    }
  }
}

//TEST_SUITE("FISHERCLASSIFIER") {
//  TEST_CASE("PROBLEMS")
//  {
//    int D, N, Nc; //double tol = 1e-2;
//    std::string type;
//
//    SUBCASE("1D-Simple")
//    {
//      D = 1; N = 10000; Nc = 2; type = "Simple";
//
//      utils::data::classification::Dataset data(D, N, Nc);
//      data.Generate(type);
//      algo::classification::FISHERC model(data.inputs_, data.labels_);
//      double error = model.ComputeError(data.inputs_, data.labels_);
//      CHECK ( error <= 0. );
//    }
//    SUBCASE("2D-Simple")
//    {
//      D = 2; N = 10000; Nc = 2; type = "Simple";
//
//      utils::data::classification::Dataset data(D, N, Nc);
//      data.Generate(type);
//      algo::classification::FISHERC model(data.inputs_, data.labels_);
//      double error = model.ComputeError(data.inputs_, data.labels_);
//      CHECK ( error <= 0. );
//    }
//    SUBCASE("1D-Hard")
//    {
//      D = 1; N = 10000; Nc = 2; type = "Hard";
//
//      utils::data::classification::Dataset data(D, N, Nc);
//      data.Generate(type);
//      algo::classification::FISHERC model(data.inputs_, data.labels_);
//      double error = model.ComputeError(data.inputs_, data.labels_);
//      CHECK ( error <= 0.1 );
//    }
//    SUBCASE("2D-Hard")
//    {
//      D = 2; N = 10000; Nc = 2; type = "Hard";
//
//      utils::data::classification::Dataset data(D, N, Nc);
//      data.Generate(type);
//      algo::classification::FISHERC model(data.inputs_, data.labels_);
//      double error = model.ComputeError(data.inputs_, data.labels_);
//      CHECK ( error <= 0.1 );
//    }
//  }
//}

#endif
