/**
 * @file test_utils.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_UTILS_H 
#define TEST_UTILS_H

TEST_SUITE("Dataset") {

  int D, N;
  double a, p, eps;

  double tol = 1.e-6;

  TEST_CASE("Regression-1D")
  {
    D=1; N=4;
    utils::data::regression::Dataset data(D, N);
    a = 1.0; p = 0.; eps = 0.0;

    SUBCASE("SINE")
    {
      data.Generate(a, p, "Sine", eps);

      CHECK ( data.inputs_.n_cols == N );
      CHECK ( data.inputs_.n_rows == D );

      CHECK ( data.labels_.n_cols == N );
      CHECK ( data.labels_.n_rows == 1 );

      double sum_check = std::pow(std::cos(data.inputs_(0,0)),2) 
                         + std::pow(data.labels_(0),2);

      CHECK ( sum_check - 1. < tol );

      arma::mat labels = data.labels_;
      eps = 0.1;
      data.Noise(eps);

      CHECK ( labels(0) != data.labels_(0) );
    }

    SUBCASE("LINEAR")
    {
      data.Generate(a, p, "Linear", eps);

      CHECK ( data.inputs_.n_cols == N );
      CHECK ( data.inputs_.n_rows == D );

      CHECK ( data.labels_.n_cols == N );
      CHECK ( data.labels_.n_rows == 1 );
      
      CHECK ( data.inputs_(0,0) == data.labels_(0) ); 

      arma::mat labels = data.labels_;
      eps = 0.1;
      data.Noise(eps);

      CHECK ( labels(0) != data.labels_(0) );
    }
  }

  TEST_CASE("Regression-2D")
  {
    D=2; N=4;
    utils::data::regression::Dataset data(D, N);
    a = 1.0; p = 0.; eps = 0.0;

    SUBCASE("SINE")
    {
      data.Generate(a, p, "Sine", eps);

      CHECK ( data.inputs_.n_cols == N );
      CHECK ( data.inputs_.n_rows == D );

      CHECK ( data.labels_.n_cols == N );
      CHECK ( data.labels_.n_rows == 1 );

      double sum_check = std::pow(std::cos(data.inputs_(0,0)
                                          +data.inputs_(1,0)),2) 
                                 + std::pow(data.labels_(0),2);

      CHECK ( sum_check - 1. < tol );

      arma::mat labels = data.labels_;
      eps = 0.1;
      data.Noise(eps);

      CHECK ( labels(0) != data.labels_(0) );
    }

    SUBCASE("LINEAR")
    {
      data.Generate(a, p, "Linear", eps);

      CHECK ( data.inputs_.n_cols == N );
      CHECK ( data.inputs_.n_rows == D );

      CHECK ( data.labels_.n_cols == N );
      CHECK ( data.labels_.n_rows == 1 );
      
      CHECK ( data.inputs_(0,0)+data.inputs_(1,0) == data.labels_(0) ); 

      arma::mat labels = data.labels_;
      eps = 0.1;
      data.Noise(eps);

      CHECK ( labels(0) != data.labels_(0) );
    }
  }
  TEST_CASE("CLASSIFICATION")
  {
    int Nc; tol = 1e-1;

    SUBCASE("SIMPLE-1D")
    {
      D = 1; N = 10000; Nc = 2;
      utils::data::classification::Dataset data(D, N, Nc);
      std::string type = "Simple";
      data.Generate(type);

      CHECK ( data.inputs_.n_cols == 2*N );
      CHECK ( data.inputs_.n_rows == D );
      CHECK ( data.labels_.n_cols == 2*N );
      CHECK ( data.labels_.n_rows == 1 );
      CHECK ( arma::mean(arma::mean(data.labels_)) == 0.5 );
      CHECK ( std::abs(arma::mean(arma::mean(data.inputs_))) <= tol);
      //CHECK ( arma::mean(arma::mean(data.inputs_.cols(0,N-1))) + 5 <= tol );
      //CHECK ( arma::mean(arma::mean(data.inputs_.cols(N,2*N-1))) - 5 <= tol );
      //CHECK ( arma::stddev(arma::stddev(data.inputs_.cols(N,2*N-1))) - 
      //                                                 std::sqrt(0.1) <= tol );
      //CHECK ( arma::stddev(arma::stddev(data.inputs_.cols(0,N-1))) - 
      //                                                 std::sqrt(0.1) <= tol );
    }
    SUBCASE("SIMPLE-2D")
    {
      D = 2; N = 10000; Nc = 2;
      utils::data::classification::Dataset data(D, N, Nc);
      std::string type = "Simple";
      data.Generate(type);

      CHECK ( data.inputs_.n_cols == 2*N );
      CHECK ( data.inputs_.n_rows == D );
      CHECK ( data.labels_.n_cols == 2*N );
      CHECK ( data.labels_.n_rows == 1 );
      CHECK ( arma::mean(arma::mean(data.labels_)) == 0.5 );
      CHECK ( std::abs(arma::mean(arma::mean(data.inputs_))) <= tol );
      //CHECK ( arma::mean(arma::mean(data.inputs_.cols(0,N-1))) + 5 <= tol );
      //CHECK ( arma::mean(arma::mean(data.inputs_.cols(N,2*N-1))) - 5 <= tol );
      //CHECK ( arma::stddev(arma::stddev(data.inputs_.cols(N,2*N-1))) - 
      //                                                std::sqrt(0.1) <= tol );
      //CHECK ( arma::stddev(arma::stddev(data.inputs_.cols(0,N-1))) -
      //                                                std::sqrt(0.1) <= tol );

    }
    SUBCASE("HARD-1D")
    {
      D = 1; N = 10000; Nc = 2;
      utils::data::classification::Dataset data(D, N, Nc);
      std::string type = "Simple";
      data.Generate(type);

      CHECK ( data.inputs_.n_cols == 2*N );
      CHECK ( data.inputs_.n_rows == D );
      CHECK ( data.labels_.n_cols == 2*N );
      CHECK ( data.labels_.n_rows == 1 );
      CHECK ( arma::mean(arma::mean(data.labels_)) == 0.5 );
      CHECK ( std::abs(arma::mean(arma::mean(data.inputs_))) <= tol);
      //CHECK ( arma::mean(arma::mean(data.inputs_.cols(0,N-1))) + 5 <= tol );
      //CHECK ( arma::mean(arma::mean(data.inputs_.cols(N,2*N-1))) - 5 <= tol );
      //CHECK ( arma::stddev(arma::stddev(data.inputs_.cols(N,2*N-1))) - 
      //                                                  std::sqrt(2.) <= tol );
      //CHECK ( arma::stddev(arma::stddev(data.inputs_.cols(0,N-1))) - 
      //                                                  std::sqrt(2.) <= tol );
    }
    SUBCASE("HARD-2D")
    {
      D = 2; N = 10000; Nc = 2;
      utils::data::classification::Dataset data(D, N, Nc);
      std::string type = "Simple";
      data.Generate(type);

      CHECK ( data.inputs_.n_cols == 2*N );
      CHECK ( data.inputs_.n_rows == D );
      CHECK ( data.labels_.n_cols == 2*N );
      CHECK ( data.labels_.n_rows == 1 );
      CHECK ( arma::mean(arma::mean(data.labels_)) == 0.5 );
      CHECK ( std::abs(arma::mean(arma::mean(data.inputs_))) <= tol );
      //CHECK ( arma::mean(arma::mean(data.inputs_.cols(0,N-1),)) + 5 <= tol );
      //CHECK ( arma::mean(arma::mean(data.inputs_.cols(N,2*N-1))) - 5 <= tol );
      //CHECK ( arma::stddev(arma::stddev(data.inputs_.cols(N,2*N-1))) -
      //                                                  std::sqrt(2.) <= tol );
      //CHECK ( arma::stddev(arma::stddev(data.inputs_.cols(0,N-1))) - 
      //                                                  std::sqrt(2.) <= tol );

 
    }
    SUBCASE("DIPPING")
    {
      D = 1; N = 10000; Nc = 2;
      utils::data::classification::Dataset data(D, N, Nc);
      std::string type = "Dipping";
      data.Generate(type);

      CHECK ( data.inputs_.n_cols == 2*N );
      CHECK ( data.inputs_.n_rows == D );
      CHECK ( data.labels_.n_cols == 2*N );
      CHECK ( data.labels_.n_rows == 1 );
      CHECK ( arma::mean(arma::mean(data.labels_)) == 0.5 );

      bool check1 = arma::any(arma::vectorise(data.inputs_) < -2.5);
      bool check2 = arma::any(arma::vectorise(data.inputs_) > 2.5);

      CHECK ( !check1 );
      CHECK ( !check2 );
    }

    SUBCASE("DELAYED-DIPPING")
    {
      D = 2; N = 10; Nc = 2;

      utils::data::classification::Dataset data(D, N, Nc);
      std::string type = "Delayed-Dipping";
      data.Generate(type);

      CHECK ( data.inputs_.n_cols == 2*N );
      CHECK ( data.inputs_.n_rows == D );
      CHECK ( data.labels_.n_cols == 2*N );
      CHECK ( data.labels_.n_rows == 1 );
      CHECK ( arma::mean(arma::mean(data.labels_)) == 0.5 );

      CHECK ( arma::min(arma::min(data.inputs_)) > -1.1 );
      CHECK ( arma::max(arma::max(data.inputs_)) < 1.1 );

      arma::mat inside = data.inputs_(arma::span(0,1),arma::span(N,2*N-1));
      double in_max = arma::max(arma::max(inside));
      double in_min = arma::min(arma::min(inside));

      CHECK ( in_max < 2. );  
      CHECK ( in_min > -2. );  
    }
    SUBCASE("DELAYED-DIPPING-ARGS")
    {
      D = 2; N = 10; Nc = 2;

      utils::data::classification::Dataset data(D, N, Nc);
      double r = 10; double eps =0.01;
      data._dipping(r, eps);

      CHECK ( data.inputs_.n_cols == 2*N );
      CHECK ( data.inputs_.n_rows == D );
      CHECK ( data.labels_.n_cols == 2*N );
      CHECK ( data.labels_.n_rows == 1 );
      CHECK ( arma::mean(arma::mean(data.labels_)) == 0.5 );

      CHECK ( arma::min(arma::min(data.inputs_)) > -10.1 );
      CHECK ( arma::max(arma::max(data.inputs_)) < 10.1 );

      arma::mat inside = data.inputs_(arma::span(0,1),arma::span(N,2*N-1));
      double in_max = arma::max(arma::max(inside));
      double in_min = arma::min(arma::min(inside));

      CHECK ( in_max < 2. );
      CHECK ( in_min > -2. );

    }
  }
}

TEST_SUITE("SINEGEN") {

  int M=3; int N=10;
  double eps;

  utils::data::regression::SineGen funcs(M);

  arma::mat inputs(1,N,arma::fill::randn);

  TEST_CASE("PHASE")
  {

    SUBCASE("NOISE=0")
    {
      arma::mat psi = funcs.Predict(inputs, "Phase");

      CHECK ( psi.n_cols == N );
      CHECK ( psi.n_rows == M );
      CHECK ( std::sin(inputs(0)+funcs.p_(0)) == psi(0,0) );
      CHECK ( std::sin(inputs(N-1)+funcs.p_(M-1)) == psi(M-1,N-1) );
    }
    SUBCASE("NOISE=0.1")
    {
      eps = 0.1;
      arma::mat psi = funcs.Predict(inputs, "Phase", eps);

      CHECK ( psi.n_cols == N );
      CHECK ( psi.n_rows == M );
      CHECK ( std::sin(inputs(0)+funcs.p_(0)) != psi(0,0) );
      CHECK ( std::sin(inputs(N-1)+funcs.p_(M-1)) != psi(M-1,N-1) );
    }
  } 

  TEST_CASE("AMPLITUDE")
  {
    SUBCASE("NOISE=0")
    {
      arma::mat psi = funcs.Predict(inputs, "Amplitude");

      CHECK ( psi.n_cols == N );
      CHECK ( psi.n_rows == M );
      CHECK ( funcs.a_(0)*std::sin(inputs(0)) == psi(0,0) );
      CHECK ( funcs.a_(M-1)*std::sin(inputs(N-1)) == psi(M-1,N-1) );
    }
    SUBCASE("NOISE=0.1")
    {
      eps = 0.1;
      arma::mat psi = funcs.Predict(inputs, "Amplitude", eps);

      CHECK ( psi.n_cols == N );
      CHECK ( psi.n_rows == M );
      CHECK ( funcs.a_(0)*std::sin(inputs(0)) != psi(0,0) );
      CHECK ( funcs.a_(M-1)*std::sin(inputs(N-1)) != psi(M-1,N-1) );
    }
  }
  TEST_CASE("PHASEAMPLITUDE")
  {
    SUBCASE("NOISE=0")
    {
      arma::mat psi = funcs.Predict(inputs, "PhaseAmplitude");

      CHECK ( psi.n_cols == N );
      CHECK ( psi.n_rows == M );
      CHECK ( funcs.a_(0)*std::sin(inputs(0)+funcs.p_(0)) == psi(0,0) );
      CHECK ( funcs.a_(M-1)*std::sin(inputs(N-1)+funcs.p_(M-1))
                                                           == psi(M-1,N-1) );
    }
    SUBCASE("NOISE=0.1")
    {
      eps = 0.1;
      arma::mat psi = funcs.Predict(inputs, "PhaseAmplitude", eps);

      CHECK ( psi.n_cols == N );
      CHECK ( psi.n_rows == M );
      CHECK ( funcs.a_(0)*std::sin(inputs(0)+funcs.p_(0)) != psi(0,0) );
      CHECK ( funcs.a_(M-1)*std::sin(inputs(N-1)+funcs.p_(0)) != psi(M-1,N-1) );
    }
  }
}

TEST_SUITE("EXTRACT_CLASSES") {
    int D; int N=10; int Nc=2; double tol = 1e-6;
    TEST_CASE("1D")
    {
      D = 1;

      utils::data::classification::Dataset data(D, N, Nc);
      std::string type = "Simple";
      data.Generate(type);

      std::tuple<arma::mat, arma::uvec> collect0;
      std::tuple<arma::mat, arma::uvec> collect1;
      collect0 = utils::extract_class(data.inputs_, data.labels_,0);
      collect1 = utils::extract_class(data.inputs_, data.labels_,1);

      arma::mat out0 = std::get<0>(collect0);
      arma::mat out1 = std::get<0>(collect1);

      arma::uvec id0 = arma::regspace<arma::uvec>(0,1,N-1);
      arma::uvec id1 = arma::regspace<arma::uvec>(N,1,2*N-1);

      CHECK ( arma::norm((data.inputs_.cols(id0)-out0)) <= tol );
      CHECK ( arma::norm(data.inputs_.cols(id1)-out1) <= tol );

      CHECK ( arma::norm(std::get<1>(collect0) - id0) <=tol );
      CHECK ( arma::norm(std::get<1>(collect1) - id1) <=tol );
    }
    TEST_CASE("2D")
    {
      D = 2;

      utils::data::classification::Dataset data(D, N, Nc);
      std::string type = "Simple";
      data.Generate(type);

      std::tuple<arma::mat, arma::uvec> collect0;
      std::tuple<arma::mat, arma::uvec> collect1;
      collect0 = utils::extract_class(data.inputs_, data.labels_,0);
      collect1 = utils::extract_class(data.inputs_, data.labels_,1);

      arma::mat out0 = std::get<0>(collect0);
      arma::mat out1 = std::get<0>(collect1);

      arma::uvec id0 = arma::regspace<arma::uvec>(0,1,N-1);
      arma::uvec id1 = arma::regspace<arma::uvec>(N,1,2*N-1);

      CHECK ( arma::norm((data.inputs_.cols(id0)-out0)) <= tol );
      CHECK ( arma::norm(data.inputs_.cols(id1)-out1) <= tol );

      CHECK ( arma::norm(std::get<1>(collect0) - id0) <=tol );
      CHECK ( arma::norm(std::get<1>(collect1) - id1) <=tol );
    }


  
}



#endif
