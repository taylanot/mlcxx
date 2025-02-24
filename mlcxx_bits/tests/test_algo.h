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
    data::functional::Dataset funcset(D, Ntrn,M);
    data::functional::Dataset testfuncset(D, Ntst,M);

    funcset.Generate("Sine", eps);
    testfuncset.Generate("Sine", eps);
    
    arma::Mat<DTYPE> inputs = funcset.inputs_;
    arma::Mat<DTYPE> testinputs = testfuncset.inputs_;
    arma::Mat<DTYPE> labels = funcset.labels_;

    arma::Mat<DTYPE> smooth_labels;
    arma::Mat<DTYPE> pred_inputs = testinputs;
    
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
      data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);      

      algo::regression::SemiParamKernelRidge<mlpack::GaussianKernel,
                                             data::functional::SineGen<>> 
                          model(inputs,labels, 0.5, size_t(5), 1.);
      DTYPE error = model.ComputeError(inputs, labels);
      CHECK ( error <= tol );
    }  
  }
}

/* TEST_SUITE("ANN") { */
/*   TEST_CASE("PROBLEMS") */
/*   { */
/*     int D, N; DTYPE tol = 1e-3; */
/*     double a, p, eps; */
/*     std::string type; */
/*     typedef mlpack::FFN<mlpack::MeanSquaredError> NetworkType; */
/*     NetworkType network; */
/*     network.Add<mlpack::Linear>(1); */

/*     /1* algo::ANN<NetworkType> model(&network); *1/ */

/*     SUBCASE("1D-Linear") */
/*     { */
/*       D = 1; N = 10; type = "Linear"; */
/*       a = 1.0; p = 0.; eps = 0.0; */
/*       data::regression::Dataset dataset(D, N); */
/*       dataset.Generate(a, p, type, eps); */

/*       auto inputs = dataset.inputs_; */
/*       auto labels = dataset.labels_; */

/*       algo::ANN<NetworkType> model(dataset.inputs_, */
/*                                      dataset.labels_,&network); */

/*       arma::mat preds; */
/*       model.Predict(inputs, preds); */

/*       CHECK ( model.ComputeError(inputs,labels) <= tol ); */
/*     } */

/*     SUBCASE("2D-Linear") */
/*     { */
/*       D = 2; N = 40; type = "Linear"; */
/*       a = 1.0; p = 0.; eps = 0.0; */
/*       data::regression::Dataset dataset(D, N); */
/*       dataset.Generate(a, p, type, eps); */

/*       auto inputs = dataset.inputs_; */
/*       auto labels = dataset.labels_; */

/*       algo::ANN<NetworkType> model(dataset.inputs_, */
/*                                      dataset.labels_,&network); */


/*       CHECK ( model.ComputeError(inputs,labels) <= tol ); */
/*     } */
/*     SUBCASE("Optimizer") */
/*     { */
/*       D = 1; N = 10; type = "Linear"; */
/*       a = 1.0; p = 0.; eps = 0.0; */
/*       data::regression::Dataset dataset(D, N); */
/*       dataset.Generate(a, p, type, eps); */

/*       auto inputs = dataset.inputs_; */
/*       auto labels = dataset.labels_; */

/*       algo::ANN<NetworkType,ens::Adam> model(dataset.inputs_, */
/*                                              dataset.labels_,&network, false, */
/*                                              0.001,32); */

/*       CHECK ( model.ComputeError(inputs,labels) <= tol ); */
/*     } */
/*   } */
/* } */

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
      data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);

      algo::regression::KernelRidge<mlpack::GaussianKernel>
                                                 model(inputs,labels, 0.,0.1);
      DTYPE error = model.ComputeError(inputs, labels);
      CHECK ( error <= tol );
    }
    SUBCASE("2D-Sine")
    {
      D = 2; N = 10; type = "Sine";
      a = 1.0; p = 0.; eps = 0.0;
      data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);

      algo::regression::KernelRidge<mlpack::GaussianKernel>
                                                model(inputs,labels, 0.,0.1);
      DTYPE error = model.ComputeError(inputs, labels);
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
      data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);

      algo::regression::GaussianProcess<mlpack::GaussianKernel> model(inputs,labels, 0.,0.1);
      DTYPE error = model.ComputeError(inputs, labels);
      double ll = model.LogLikelihood(inputs, labels);
      CHECK ( error <= tol );
      CHECK ( ll <= 0. );
    }
    SUBCASE("2D-Sine")
    {
      D = 2; N = 10; type = "Sine";
      a = 1.0; p = 0.; eps = 0.0;
      data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);

      algo::regression::GaussianProcess<mlpack::GaussianKernel>
                                                 model(inputs,labels, 0.,0.1);
      DTYPE error = model.ComputeError(inputs, labels);
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
      data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);

      algo::regression::GaussianProcess<mlpack::GaussianKernel> 
                                                  model(inputs,labels, 0.,0.1);
      arma::Mat<DTYPE> labprior,labposterior;
      arma::Mat<DTYPE> inp = arma::randu<arma::Mat<DTYPE>>(D,Ngrid);
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
      data::regression::Dataset dataset(D, N);

      dataset.Generate(a, p, type, eps);

      auto inputs = dataset.inputs_;
      auto labels = arma::conv_to<arma::Row<DTYPE>>::from(dataset.labels_);

      algo::regression::GaussianProcess<mlpack::GaussianKernel> 
                                                  model(inputs,labels, 0.,0.1);
      arma::Mat<DTYPE> labprior,labposterior;
      arma::Mat<DTYPE> inp = arma::randu<arma::Mat<DTYPE>>(D,Ngrid);
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

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_,data.labels_,Nc);
      double acc = model.ComputeAccuracy(data.inputs_, data.labels_);
      CHECK ( acc == 100.  );
    }
    SUBCASE("1D-Simple-Shrink")
    {
      D = 1; N = 5; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_,Nc,1000);
      algo::classification::NMC model_(data.inputs_, data.labels_,Nc);
      arma::Mat<DTYPE> param1 = model.Parameters(); 
      arma::Mat<DTYPE> param2 = model_.Parameters();
      CHECK ( param1(0,0) != param2(0,0));
      CHECK ( param1(0,1) != param2(0,1));
    }
    SUBCASE("2D-Simple-Shrink")
    {
      D = 2; N = 5; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_,Nc,1000);
      algo::classification::NMC model_(data.inputs_, data.labels_,Nc);
      arma::Mat<DTYPE> param1 = model.Parameters(); 
      arma::Mat<DTYPE> param2 = model_.Parameters();
      CHECK ( param1(0,0) != param2(0,0));
      CHECK ( param1(0,1) != param2(0,1));
    }
    SUBCASE("2D-Simple")
    {
      D = 2; N = 10000; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_,Nc);
      double acc = model.ComputeAccuracy(data.inputs_, data.labels_);
      CHECK ( acc == 100. );
    }
    SUBCASE("1D-Hard")
    {
      D = 1; N = 10000; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_,Nc);
      double acc = model.ComputeAccuracy(data.inputs_, data.labels_);
      CHECK ( acc >= 0.9 );
    }
    SUBCASE("2D-Hard")
    {
      D = 2; N = 10000; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_,Nc);
      double acc = model.ComputeAccuracy(data.inputs_, data.labels_);
      CHECK ( acc >= 0.9 );
    }
  }
}

/* TEST_SUITE("MULTICLASS") { */
/*   TEST_CASE("PROBLEMS") */
/*   { */
/*     SUBCASE("SVM") */
/*     { */
      
/*     } */

/*     SUBCASE("LREG") */
/*     { */
      
/*     } */
/*   } */
/* } */

TEST_SUITE("LDC") {
  TEST_CASE("PROBLEMS")
  {
    int D, N, Nc; //double tol = 1e-2;
    std::string type;

    SUBCASE("1D-Simple")
    {
      D = 1; N = 10000; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::LDC model(data.inputs_, data.labels_,Nc);
      DTYPE error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("2D-Simple")
    {
      D = 2; N = 10000; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::LDC model(data.inputs_, data.labels_,Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("1D-Hard")
    {
      D = 1; N = 10000; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::LDC model(data.inputs_, data.labels_,Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.1 );
    }
    SUBCASE("2D-Hard")
    {
      D = 2; N = 10000; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::LDC model(data.inputs_, data.labels_,Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.1 );
    }
  }
}

TEST_SUITE("QDC") {
  TEST_CASE("PROBLEMS")
  {
    int D, N, Nc; //double tol = 1e-2;
    std::string type;

    SUBCASE("1D-Simple")
    {
      D = 1; N = 10000; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::QDC model(data.inputs_, data.labels_,Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("2D-Simple")
    {
      D = 2; N = 10000; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::QDC model(data.inputs_, data.labels_,Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("1D-Hard")
    {
      D = 1; N = 10000; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::QDC model(data.inputs_, data.labels_,Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.1 );
    }
    SUBCASE("2D-Hard")
    {
      D = 2; N = 10000; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::QDC model(data.inputs_, data.labels_,Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.1 );
    }
    SUBCASE("BANANA")
    {
      D = 2; N = 10000; Nc = 2; type = "Banana";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::QDC model(data.inputs_, data.labels_,Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.2 );
    }
  }
}

TEST_SUITE("NNC") {
  TEST_CASE("PROBLEMS")
  {
    int D, N, Nc; //double tol = 1e-2;
    std::string type;

    SUBCASE("1D-Simple")
    {
      D = 1; N = 100; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NNC model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }

    SUBCASE("2D-Simple")
    {
      D = 2; N = 100; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NNC model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("1D-Hard")
    {
      D = 1; N = 100; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NNC model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.5 );
    }
    SUBCASE("2D-Hard")
    {
      D = 2; N = 100; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NNC model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.5 );
    }
    SUBCASE("BANANA")
    {
      D = 2; N = 100; Nc = 2; type = "Banana";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NNC model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.5 );
    }
  }
}

TEST_SUITE("SVM") {
  TEST_CASE("PROBLEMS")
  {
    int D, N, Nc; //double tol = 1e-2;
    std::string type;

    SUBCASE("1D-Simple")
    {
      D = 1; N = 100; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::SVM model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }

    SUBCASE("2D-Simple")
    {
      D = 2; N = 100; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::SVM model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("1D-Hard")
    {
      D = 1; N = 100; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::SVM model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.5 );
    }
    SUBCASE("2D-Hard")
    {
      D = 2; N = 100; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::SVM model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.5 );
    }
    SUBCASE("BANANA")
    {
      D = 2; N = 100; Nc = 2; type = "Banana";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::SVM<mlpack::GaussianKernel>
        model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.5 );
    }
  }
}

TEST_SUITE("LOGISTICREG") {
  TEST_CASE("PROBLEMS")
  {
    int D, N, Nc; //double tol = 1e-2;
    std::string type;

    SUBCASE("1D-Simple")
    {
      D = 1; N = 100; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::LogisticRegression 
        model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }

    SUBCASE("2D-Simple")
    {
      D = 2; N = 100; Nc = 2; type = "Simple";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::LogisticRegression 
        model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("1D-Hard")
    {
      D = 1; N = 100; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::LogisticRegression 
        model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.5 );
    }
    SUBCASE("2D-Hard")
    {
      D = 2; N = 100; Nc = 2; type = "Hard";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::LogisticRegression 
        model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.5 );
    }
    SUBCASE("BANANA")
    {
      D = 2; N = 100; Nc = 2; type = "Banana";

      data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::LogisticRegression 
        model(data.inputs_, data.labels_, Nc);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.5 );
    }
  }
}
#endif
