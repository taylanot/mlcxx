/**
 * @file test_algo.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_ALGO_H 
#define TEST_ALGO_H

TEST_SUITE("GPR") {
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

      algo::regression::GP<mlpack::GaussianKernel> model(inputs,labels, 0.,0.1);
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

      algo::regression::GP<mlpack::GaussianKernel> model(inputs,labels, 0.,0.1);
      double error = model.ComputeError(inputs, labels);
      double ll = model.LogLikelihood(inputs, labels);
      CHECK ( error <= tol );
      CHECK ( ll <= 0. );
    }
  }
}

TEST_SUITE("NMC") {
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
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("2D-Simple")
    {
      D = 2; N = 10000; Nc = 2; type = "Simple";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("1D-Hard")
    {
      D = 1; N = 10000; Nc = 2; type = "Hard";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.1 );
    }
    SUBCASE("2D-Hard")
    {
      D = 2; N = 10000; Nc = 2; type = "Hard";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::NMC model(data.inputs_, data.labels_);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.1 );
    }
  }
}

TEST_SUITE("FISHERC") {
  TEST_CASE("PROBLEMS")
  {
    int D, N, Nc; //double tol = 1e-2;
    std::string type;

    SUBCASE("1D-Simple")
    {
      D = 1; N = 10000; Nc = 2; type = "Simple";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::FISHERC model(data.inputs_, data.labels_);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("2D-Simple")
    {
      D = 2; N = 10000; Nc = 2; type = "Simple";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::FISHERC model(data.inputs_, data.labels_);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0. );
    }
    SUBCASE("1D-Hard")
    {
      D = 1; N = 10000; Nc = 2; type = "Hard";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::FISHERC model(data.inputs_, data.labels_);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.1 );
    }
    SUBCASE("2D-Hard")
    {
      D = 2; N = 10000; Nc = 2; type = "Hard";

      utils::data::classification::Dataset data(D, N, Nc);
      data.Generate(type);
      algo::classification::FISHERC model(data.inputs_, data.labels_);
      double error = model.ComputeError(data.inputs_, data.labels_);
      CHECK ( error <= 0.1 );
    }
  }
}
TEST_SUITE("SMPKR") {
  TEST_CASE("PROBLEMS")
  {
    CHECK( 1 == 1 );
  }
}

#endif
