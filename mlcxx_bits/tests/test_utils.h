/**
 * @file test_utils.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_UTILS_H 
#define TEST_UTILS_H

TEST_SUITE("DATASET") {

  int D, N;
  double a, p, eps;

  double tol = 1.e-2;

  TEST_CASE("LOAD-REGRESSION")
  {
    utils::data::regression::Dataset dataset;
    dataset.Load("datasets/winequality-red.csv",11,1,true,false);
    CHECK ( dataset.inputs_(0,0) == 7.4 );
    CHECK ( dataset.inputs_(dataset.dimension_-1,dataset.size_-1) == 11 );
    CHECK ( dataset.labels_(0,0) == 5. );
    CHECK ( dataset.labels_(0,dataset.size_-1) == 6. );
  }

  TEST_CASE("REGRESSION-1D")
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

      //double sum_check = std::pow(std::cos(data.inputs_(0,0)),2) 
      //                   + std::pow(data.inputs_(0,0),2);

      //CHECK ( sum_check - 1. < tol );

      arma::mat labels = data.labels_;
      eps = 0.1;
      data.Noise(eps);

      CHECK ( labels(0) != data.labels_(0) );
    }
    SUBCASE("SINC")
    {
      data.Generate(a, p, "Sinc", eps);

      CHECK ( data.inputs_.n_cols == N );
      CHECK ( data.inputs_.n_rows == D );

      CHECK ( data.labels_.n_cols == N );
      CHECK ( data.labels_.n_rows == 1 );

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

  TEST_CASE("REGRESSION-2D")
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

      //double sum_check = std::pow(std::cos(data.inputs_(0,0)
      //                                    +data.inputs_(1,0)),2) 
      //                      + std::pow(data.inputs_(0,0)+data.inputs_(1,0),2);

      //CHECK ( sum_check - 1. < tol );

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

    SUBCASE("LOAD-IRIS")
    {
      utils::data::classification::Dataset dataset;
      dataset.Load("datasets/iris.csv",true,true);
      CHECK ( dataset.inputs_(0,0) == 5.1 );
      CHECK ( dataset.inputs_(dataset.dimension_-1,dataset.size_-1) == 1.8 );
      CHECK ( dataset.num_class_ == 3 );
    }

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
      CHECK ( arma::mean(arma::mean(arma::conv_to<arma::rowvec>
                                              ::from(data.labels_))) == 0.5 );
      CHECK ( std::abs(arma::mean(arma::mean(data.inputs_))) <= tol);
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
      CHECK ( arma::mean(arma::mean(arma::conv_to<arma::rowvec>
                                              ::from(data.labels_))) == 0.5 );
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
      CHECK ( arma::mean(arma::mean(arma::conv_to<arma::rowvec>
                                              ::from(data.labels_))) == 0.5 );
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
      CHECK ( arma::mean(arma::mean(arma::conv_to<arma::rowvec>
                                              ::from(data.labels_))) == 0.5 );
      CHECK ( std::abs(arma::mean(arma::mean(data.inputs_))) <= tol );
      //CHECK ( arma::mean(arma::mean(data.inputs_.cols(0,N-1),)) + 5 <= tol );
      //CHECK ( arma::mean(arma::mean(data.inputs_.cols(N,2*N-1))) - 5 <= tol );
      //CHECK ( arma::stddev(arma::stddev(data.inputs_.cols(N,2*N-1))) -
      //                                                  std::sqrt(2.) <= tol );
      //CHECK ( arma::stddev(arma::stddev(data.inputs_.cols(0,N-1))) - 
      //                                                  std::sqrt(2.) <= tol );

 
    }

    SUBCASE("BANANA")
    {
      D = 2; N = 1000; Nc = 2;
      utils::data::classification::Dataset data(D, N, Nc);
      std::string type = "Banana";
      data.Generate(type);

      CHECK ( data.inputs_.n_cols == 2*N );
      CHECK ( data.inputs_.n_rows == D );
      CHECK ( data.labels_.n_cols == 2*N );
      CHECK ( data.labels_.n_rows == 1 );
      CHECK ( arma::mean(arma::mean(arma::conv_to<arma::rowvec>
                                              ::from(data.labels_))) == 0.5 );
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
      CHECK ( arma::mean(arma::mean(arma::conv_to<arma::rowvec>
                                              ::from(data.labels_))) == 0.5 );

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
      CHECK ( arma::mean(arma::mean(arma::conv_to<arma::rowvec>
                                              ::from(data.labels_))) == 0.5 );

      CHECK ( arma::min(arma::min(data.inputs_)) > -24. );
      CHECK ( arma::max(arma::max(data.inputs_)) < 24. );

      arma::mat inside = data.inputs_(arma::span(0,1),arma::span(N,2*N-1));
      double in_max = arma::max(arma::max(inside));
      double in_min = arma::min(arma::min(inside));

      CHECK ( in_max < 4 );  
      CHECK ( in_min > -4 );  
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
      CHECK ( arma::mean(arma::mean(arma::conv_to<arma::rowvec>
                                              ::from(data.labels_))) == 0.5 );

      CHECK ( arma::min(arma::min(data.inputs_)) > -10.1 );
      CHECK ( arma::max(arma::max(data.inputs_)) < 10.1 );

      arma::mat inside = data.inputs_(arma::span(0,1),arma::span(N,2*N-1));
      double in_max = arma::max(arma::max(inside));
      double in_min = arma::min(arma::min(inside));

      CHECK ( in_max < 2. );
      CHECK ( in_min > -2. );

    }

    SUBCASE("IMBALANCE-GAUSSIAN")
    {
      D = 2; N = 10; Nc = 2; double perc = 0.2;

      utils::data::classification::Dataset data(D, N, Nc);
      data._imbalance2classgauss(perc);

      CHECK ( data.inputs_.n_cols == 2*N );
      CHECK ( data.inputs_.n_rows == D );
      CHECK ( data.labels_.n_cols == 2*N );
      CHECK ( data.labels_.n_rows == 1 );
      CHECK ( (arma::find( data.labels_ == 0 ).eval()).n_elem == 4 );
      CHECK ( (arma::find( data.labels_ == 1 ).eval()).n_elem == 16 );
    }
  }
}

TEST_SUITE("TRANSFORM") {

  double tol = 1e-6;

  TEST_CASE("REGRESSION")
  {
    utils::data::regression::Dataset data(2, 10);
    utils::data::regression::Dataset tdata,tbdata;
    data.Generate(1,0,"Linear",0);
    utils::data::regression::Transformer trans(data);
    tdata = trans.Trans(data);
    tbdata = trans.InvTrans(tdata);

    CHECK ( arma::sum(data.inputs_(0,0) - tbdata.inputs_(0,0))  <= tol );
    CHECK ( arma::sum(data.labels_(0,0) - tbdata.labels_(0,0))  <= tol );
    
  }

  TEST_CASE("CLASSIFICATION")
  {
    utils::data::classification::Dataset data(2, 10, 2);
    utils::data::classification::Dataset tdata,tbdata;
    data.Generate("Simple");
    utils::data::classification::Transformer trans(data);
    tdata = trans.Trans(data);
    tbdata = trans.InvTrans(tdata);

    CHECK ( arma::sum(data.inputs_(0,0) - tbdata.inputs_(0,0))  <= tol );
    CHECK ( arma::sum(data.labels_(0,0) - tbdata.labels_(0,0))  <= tol );
    
  }
}

TEST_SUITE("FUNCTIONAL") {

  size_t D = 1; size_t N = 10; size_t M=3;

  utils::data::functional::Dataset funcs(D,N,M);

  TEST_CASE("GENERATE")
  {

    funcs.Generate("Sine");

    CHECK ( funcs.inputs_.n_cols == N );
    CHECK ( funcs.labels_.n_cols == N );
    CHECK ( funcs.labels_.n_rows == M );
  }
  
}
TEST_SUITE("SINEGEN") {

  int M=3; int N=10;
  double eps;

  utils::data::functional::SineGen funcs(M);

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


TEST_SUITE("SYSTEM") {
  std::filesystem::path dir1 = "ali";
  std::filesystem::path dir2 = "veli";
  std::filesystem::path dir3 = "deli";
  std::filesystem::path dir4 = "celi";
  std::filesystem::path remdir1 = dir2;
  std::filesystem::path remdir2 = dir4/dir2;
  std::filesystem::path path = dir1/dir2/dir3/dir4;
    TEST_CASE("SPLIT")
    {
      std::vector<std::filesystem::path> list = utils::split_path(path);
      CHECK ( list[0] == dir1 );
      CHECK ( list[1] == dir2 );
      CHECK ( list[2] == dir3 );
      CHECK ( list[3] == dir4 );
    }
    TEST_CASE("REMOVE")
    {
      std::filesystem::path check1 = utils::remove_path(path, remdir1);
      std::filesystem::path check2 = utils::remove_path(path, remdir2);
      CHECK ( check1 == dir1/dir3/dir4 );
      CHECK ( check2 == dir1/dir3 );
    }
}

TEST_SUITE("SPLIT-REGRESSION-DATASET") {
    
    size_t D = 1; 
    size_t N = 10; size_t Ntrn = 8; double testratio= 0.2;

    utils::data::regression::Dataset dataset(D,N);
    utils::data::regression::Dataset trainset;
    utils::data::regression::Dataset testset;

    TEST_CASE("NUMBER")
    {
      dataset.Generate("Linear",0.);

      utils::data::Split(dataset,trainset,testset,Ntrn);

      arma::mat reconst_input = 
      arma::sort(arma::join_rows(trainset.inputs_,testset.inputs_),"ascend",1);

      arma::mat reconst_label = 
      arma::sort(arma::join_rows(trainset.labels_,testset.labels_),"ascend",1);

      arma::mat sortinp = arma::sort(dataset.inputs_,"ascend",1);
      arma::mat sortlab = arma::sort(dataset.labels_,"ascend",1);

      CHECK ( trainset.size_ == 8 );
      CHECK ( testset.size_ == 2 );
      CHECK ( arma::approx_equal(reconst_input, sortinp, "absdiff", 1e-3) );
      CHECK ( arma::approx_equal(reconst_label, sortlab, "absdiff", 1e-3) );
    }

    TEST_CASE("PERCENT")
    {
      dataset.Generate("Linear",0.);

      utils::data::Split(dataset,trainset,testset,testratio);

      arma::mat reconst_input = 
      arma::sort(arma::join_rows(trainset.inputs_,testset.inputs_),"ascend",1);

      arma::mat reconst_label = 
      arma::sort(arma::join_rows(trainset.labels_,testset.labels_),"ascend",1);

      arma::mat sortinp = arma::sort(dataset.inputs_,"ascend",1);
      arma::mat sortlab = arma::sort(dataset.labels_,"ascend",1);

      CHECK ( trainset.size_ == 8 );
      CHECK ( testset.size_ == 2 );
      CHECK ( arma::approx_equal(reconst_input, sortinp, "absdiff", 1e-3) );
      CHECK ( arma::approx_equal(reconst_label, sortlab, "absdiff", 1e-3) );
    }

}

TEST_SUITE("SPLIT-CLASSIFICATION-DATASET") {
    
    size_t D = 1; size_t Nc = 2;
    size_t N = 5; size_t Ntrn = 8; double testratio= 0.2;

    utils::data::classification::Dataset dataset(D,N,Nc);
    utils::data::classification::Dataset trainset;
    utils::data::classification::Dataset testset;

    TEST_CASE("NUMBER")
    {
      dataset.Generate("Simple");

      utils::data::Split(dataset,trainset,testset,Ntrn);

      arma::mat reconst_input = 
      arma::sort(arma::join_rows(trainset.inputs_,testset.inputs_),"ascend",1);

      arma::Row<size_t> reconst_label = 
      arma::sort(arma::join_rows(trainset.labels_,testset.labels_),"ascend",1);

      arma::mat sortinp = arma::sort(dataset.inputs_,"ascend",1);
      arma::Row<size_t> sortlab = arma::sort(dataset.labels_,"ascend",1);

      CHECK ( trainset.size_ == 8 );
      CHECK ( testset.size_ == 2 );
      CHECK ( arma::approx_equal(reconst_input, sortinp, "absdiff", 1e-3) );
      CHECK ( arma::approx_equal(reconst_label, sortlab, "absdiff", 1e-3) );
    }

    TEST_CASE("PERCENT")
    {
      dataset.Generate("Simple");

      utils::data::Split(dataset,trainset,testset,testratio);

      arma::mat reconst_input = 
      arma::sort(arma::join_rows(trainset.inputs_,testset.inputs_),"ascend",1);

      arma::Row<size_t>reconst_label = 
      arma::sort(arma::join_rows(trainset.labels_,testset.labels_),"ascend",1);

      arma::mat sortinp = arma::sort(dataset.inputs_,"ascend",1);
      arma::Row<size_t> sortlab = arma::sort(dataset.labels_,"ascend",1);

      CHECK ( trainset.size_ == 8 );
      CHECK ( testset.size_ == 2 );
      CHECK ( arma::approx_equal(reconst_input, sortinp, "absdiff", 1e-3) );
      CHECK ( arma::approx_equal(reconst_label, sortlab, "absdiff", 1e-3) );
    }
}


TEST_SUITE("STRATIFIED-SPLIT") {
    
    size_t D = 1; size_t Nc = 2;
    size_t N = 5; size_t Ntrn = 8;

    utils::data::classification::Dataset dataset(D,N,Nc);
    utils::data::classification::Dataset trainset;
    utils::data::classification::Dataset testset;
    
    arma::mat traininp,testinp; arma::Row<size_t> trainlab,testlab;

    TEST_CASE("DATASET")
    {
      dataset.Generate("Simple");

      utils::data::StratifiedSplit(dataset,trainset,testset,Ntrn);

      arma::mat reconst_input = 
      arma::sort(arma::join_rows(trainset.inputs_,testset.inputs_),"ascend",1);

      arma::Row<size_t> reconst_label = 
      arma::sort(arma::join_rows(trainset.labels_,testset.labels_),"ascend",1);

      arma::mat sortinp = arma::sort(dataset.inputs_,"ascend",1);
      arma::Row<size_t> sortlab = arma::sort(dataset.labels_,"ascend",1);

      CHECK ( trainset.size_ == 8 );
      CHECK ( testset.size_ == 2 );
      CHECK ( arma::approx_equal(reconst_input, sortinp, "absdiff", 1e-3) );
      CHECK ( arma::approx_equal(reconst_label, sortlab, "absdiff", 1e-3) );
    }

    TEST_CASE("EXPLICIT")
    {
      dataset.Generate("Simple");

      utils::data::StratifiedSplit(dataset.inputs_, dataset.labels_,
                                   traininp, testinp,
                                   trainlab, testlab,
                                   Ntrn);

      arma::mat reconst_input = 
      arma::sort(arma::join_rows(traininp, testinp),"ascend",1);

      arma::Row<size_t> reconst_label = 
      arma::sort(arma::join_rows(trainlab, testlab),"ascend",1);

      arma::mat sortinp = arma::sort(dataset.inputs_,"ascend",1);
      arma::Row<size_t> sortlab = arma::sort(dataset.labels_,"ascend",1);

      CHECK ( traininp.n_cols == 8 );
      CHECK ( trainlab.n_cols == 8 );
      CHECK ( testinp.n_cols == 2 );
      CHECK ( testlab.n_cols == 2 );
      CHECK ( arma::approx_equal(reconst_input, sortinp, "absdiff", 1e-3) );
      CHECK ( arma::approx_equal(reconst_label, sortlab, "absdiff", 1e-3) );
    }

}
#endif
