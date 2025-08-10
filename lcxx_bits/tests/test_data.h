/**
 * @file test_data.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_DATA_H 
#define TEST_DATA_H

TEST_SUITE("OPENML") {

  using Dataset = data::oml::Dataset<>;
  using Collect = data::oml::Collect<>;

  TEST_CASE("Dataset")
  {
    Dataset dataset(61);
    CHECK ( dataset.inputs_.n_cols == 150 );
    CHECK ( dataset.inputs_.n_rows == 4 );
    CHECK ( dataset.labels_.n_elem == 150 );
    CHECK ( arma::unique(dataset.labels_).eval().n_elem == 3 );
  }
  TEST_CASE("Collect")
  {
    Collect collect(283);
    CHECK ( collect.GetSize() == 53 );
    CHECK ( collect.GetKeys().n_elem == 53 );
    Dataset dataset = collect.GetNext();
    CHECK ( dataset.inputs_.n_cols == 345 );
    CHECK ( dataset.inputs_.n_rows == 6 );
    CHECK ( dataset.labels_.n_elem == 345);
    CHECK ( arma::unique(dataset.labels_).eval().n_elem == 16 );

  }
}

TEST_SUITE("DATASET") {

  int D, N;
  double a, p, eps;

  double tol = 1.e-2;

  /* TEST_CASE("LOAD-REGRESSION") */
  /* { */
  /*   data::regression::Dataset dataset; */
  /*   dataset.Load(DATASET_PATH/"winequality-red.csv",11,1,true,false); */
  /*   CHECK ( dataset.inputs_(0,0) - DTYPE(7.4) < tol ); */
  /*   CHECK ( dataset.inputs_(dataset.dimension_-1,dataset.size_-1) == 11 ); */
  /*   CHECK ( dataset.labels_(0,0) == DTYPE(5.) ); */
  /*   CHECK ( dataset.labels_(0,dataset.size_-1) == DTYPE(6.) ); */
  /* } */

  TEST_CASE("REGRESSION-1D")
  {
    D=1; N=4;
    data::regression::Dataset data(D, N);
    a = 1.0; p = 0.; eps = 0.1;

    SUBCASE("OUTLIER-1")
    {
      data.Generate("Outlier-1");

      CHECK ( data.inputs_.n_cols == N );
      CHECK ( data.inputs_.n_rows == D );

      CHECK ( data.labels_.n_cols == N );
      CHECK ( data.labels_.n_rows == 1 );

    }

    SUBCASE("RANDOMLINEAR")
    {
      data.Generate("RandomLinear");

      CHECK ( data.inputs_.n_cols == N );
      CHECK ( data.inputs_.n_rows == D );

      CHECK ( data.labels_.n_cols == N );
      CHECK ( data.labels_.n_rows == 1 );

    }

    SUBCASE("SINE")
    {
      data.Generate(a, p, "Sine");

      CHECK ( data.inputs_.n_cols == N );
      CHECK ( data.inputs_.n_rows == D );

      CHECK ( data.labels_.n_cols == N );
      CHECK ( data.labels_.n_rows == 1 );

      CHECK ( data.labels_(0) == arma::sin(data.inputs_).eval()(0,0) );

      arma::Mat<DTYPE> labels = data.labels_;
      data.Noise(eps);

      CHECK ( labels(0) != data.labels_(0) );
    }
    SUBCASE("SINC")
    {
      data.Generate(a, p, "Sinc");

      CHECK ( data.inputs_.n_cols == N );
      CHECK ( data.inputs_.n_rows == D );

      CHECK ( data.labels_.n_cols == N );
      CHECK ( data.labels_.n_rows == 1 );

      CHECK ( data.labels_(0) == 
          (arma::sin(data.inputs_)/data.inputs_).eval()(0,0) );

      arma::Mat<DTYPE> labels = data.labels_;
      eps = 0.1;
      data.Noise(eps);

      CHECK ( labels(0) != data.labels_(0) );
    }

    SUBCASE("LINEAR")
    {
      data.Generate(a, p, "Linear");

      CHECK ( data.inputs_.n_cols == N );
      CHECK ( data.inputs_.n_rows == D );

      CHECK ( data.labels_.n_cols == N );
      CHECK ( data.labels_.n_rows == 1 );
      
      CHECK ( data.labels_(0) == data.inputs_(0,0) );

      arma::Mat<DTYPE> labels = data.labels_;
      data.Noise(eps);

      CHECK ( labels(0) != data.labels_(0) );
    }
  }

  TEST_CASE("REGRESSION-2D")
  {
    D=2; N=4;
    data::regression::Dataset data(D, N);
    a = 1.0; p = 0.; eps = 0.0;

    SUBCASE("SINE")
    {
      data.Generate(a, p, "Sine", eps);

      CHECK ( data.inputs_.n_cols == N );
      CHECK ( data.inputs_.n_rows == D );

      CHECK ( data.labels_.n_cols == N );
      CHECK ( data.labels_.n_rows == 1 );

      arma::Mat<DTYPE> labels = data.labels_;
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

      arma::Mat<DTYPE> labels = data.labels_;
      eps = 0.1;
      data.Noise(eps);

      CHECK ( labels(0) != data.labels_(0) );
    }
  }

  TEST_CASE("CLASSIFICATION")
  {
    int Nc; tol = 1e-1;

    /* SUBCASE("LOAD-IRIS") */
    /* { */
    /*   data::classification::Dataset dataset; */
    /*   dataset.Load(DATASET_PATH/"iris.csv",true,true,true); */
    /*   CHECK ( dataset.inputs_(0,0) == DTYPE(5.1) ); */
    /*   CHECK ( dataset.inputs_(dataset.dimension_-1,dataset.size_-1) */
    /*                                                             == DTYPE(1.8) ); */
    /*   CHECK ( dataset.num_class_ == 3 ); */
    /* } */
    SUBCASE("SIMPLE-1D")
    {
      D = 1; N = 10000; Nc = 2;
      data::classification::Dataset data(D, N, Nc);
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
      data::classification::Dataset data(D, N, Nc);
      std::string type = "Simple";
      data.Generate(type);

      CHECK ( data.inputs_.n_cols == 2*N );
      CHECK ( data.inputs_.n_rows == D );
      CHECK ( data.labels_.n_cols == 2*N );
      CHECK ( data.labels_.n_rows == 1 );
      CHECK ( arma::mean(arma::mean(arma::conv_to<arma::rowvec>
                                              ::from(data.labels_))) == 0.5 );
      CHECK ( std::abs(arma::mean(arma::mean(data.inputs_))) <= tol );
    }
    SUBCASE("HARD-1D")
    {
      D = 1; N = 10000; Nc = 2;
      data::classification::Dataset data(D, N, Nc);
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
    SUBCASE("HARD-2D")
    {
      D = 2; N = 10000; Nc = 2;
      data::classification::Dataset data(D, N, Nc);
      std::string type = "Simple";
      data.Generate(type);

      CHECK ( data.inputs_.n_cols == 2*N );
      CHECK ( data.inputs_.n_rows == D );
      CHECK ( data.labels_.n_cols == 2*N );
      CHECK ( data.labels_.n_rows == 1 );
      CHECK ( arma::mean(arma::mean(arma::conv_to<arma::rowvec>
                                              ::from(data.labels_))) == 0.5 );
      CHECK ( std::abs(arma::mean(arma::mean(data.inputs_))) <= tol );
    }

    SUBCASE("BANANA")
    {
      D = 2; N = 1000; Nc = 2;
      data::classification::Dataset data(D, N, Nc);
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
      data::classification::Dataset data(D, N, Nc);
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

      data::classification::Dataset data(D, N, Nc);
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

      arma::Mat<DTYPE> inside = data.inputs_(arma::span(0,1),arma::span(N,2*N-1));
      double in_max = arma::max(arma::max(inside));
      double in_min = arma::min(arma::min(inside));

      CHECK ( in_max < 4 );  
      CHECK ( in_min > -4 );  
    }

    SUBCASE("DELAYED-DIPPING-ARGS")
    {
      D = 2; N = 10; Nc = 2;

      data::classification::Dataset data(D, N, Nc);
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

      arma::Mat<DTYPE> inside = data.inputs_(arma::span(0,1),arma::span(N,2*N-1));
      double in_max = arma::max(arma::max(inside));
      double in_min = arma::min(arma::min(inside));

      CHECK ( in_max < 2. );
      CHECK ( in_min > -2. );

    }

    SUBCASE("IMBALANCE-GAUSSIAN")
    {
      D = 2; N = 10; Nc = 2; double perc = 0.2;

      data::classification::Dataset data(D, N, Nc);
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
    data::regression::Dataset data(2, 10);
    data::regression::Dataset tdata,tbdata;
    data.Generate(1,0,"Linear",0);
    data::regression::Transformer trans(data);
    tdata = trans.Trans(data);
    tbdata = trans.InvTrans(tdata);

    CHECK ( arma::sum(data.inputs_(0,0) - tbdata.inputs_(0,0))  <= tol );
    CHECK ( arma::sum(data.labels_(0,0) - tbdata.labels_(0,0))  <= tol );
    
  }

  TEST_CASE("CLASSIFICATION")
  {
    data::classification::Dataset data(2, 10, 2);
    data::classification::Dataset tdata,tbdata;
    data.Generate("Simple");
    data::classification::Transformer trans(data);
    tdata = trans.Trans(data);
    tbdata = trans.InvTrans(tdata);

    CHECK ( arma::sum(data.inputs_(0,0) - tbdata.inputs_(0,0))  <= tol );
    CHECK ( arma::sum(data.labels_(0,0) - tbdata.labels_(0,0))  <= tol );
    
  }
}

TEST_SUITE("FUNCTIONAL") {

  size_t D = 1; size_t N = 10; size_t M=3;

  data::functional::Dataset funcs(D,N,M);

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

  data::functional::SineGen funcs(M);

  arma::Mat<DTYPE> inputs(1,N,arma::fill::randn);

  TEST_CASE("PHASE")
  {

    SUBCASE("NOISE=0")
    {
      arma::Mat<DTYPE> psi = funcs.Predict(inputs, "Phase");

      CHECK ( psi.n_cols == N );
      CHECK ( psi.n_rows == M );
      CHECK ( std::sin(inputs(0)+funcs.p_(0)) == psi(0,0) );
      CHECK ( std::sin(inputs(N-1)+funcs.p_(M-1)) == psi(M-1,N-1) );
    }
    SUBCASE("NOISE=0.1")
    {
      eps = 0.1;
      arma::Mat<DTYPE> psi = funcs.Predict(inputs, "Phase", eps);

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
      arma::Mat<DTYPE> psi = funcs.Predict(inputs, "Amplitude");

      CHECK ( psi.n_cols == N );
      CHECK ( psi.n_rows == M );
      CHECK ( funcs.a_(0)*std::sin(inputs(0)) == psi(0,0) );
      CHECK ( funcs.a_(M-1)*std::sin(inputs(N-1)) == psi(M-1,N-1) );
    }
    SUBCASE("NOISE=0.1")
    {
      eps = 0.1;
      arma::Mat<DTYPE> psi = funcs.Predict(inputs, "Amplitude", eps);

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
      arma::Mat<DTYPE> psi = funcs.Predict(inputs, "PhaseAmplitude");

      CHECK ( psi.n_cols == N );
      CHECK ( psi.n_rows == M );
      CHECK ( funcs.a_(0)*std::sin(inputs(0)+funcs.p_(0)) == psi(0,0) );
      CHECK ( funcs.a_(M-1)*std::sin(inputs(N-1)+funcs.p_(M-1))
                                                           == psi(M-1,N-1) );
    }
    SUBCASE("NOISE=0.1")
    {
      eps = 0.1;
      arma::Mat<DTYPE> psi = funcs.Predict(inputs, "PhaseAmplitude", eps);

      CHECK ( psi.n_cols == N );
      CHECK ( psi.n_rows == M );
      CHECK ( funcs.a_(0)*std::sin(inputs(0)+funcs.p_(0)) != psi(0,0) );
      CHECK ( funcs.a_(M-1)*std::sin(inputs(N-1)+funcs.p_(0)) != psi(M-1,N-1) );
    }
  }
}

TEST_SUITE("EXTRACT_CLASSES") {
    int D; int N=10; int Nc=2; double tol = 1e-6;
    namespace ac = algo::classification;
    TEST_CASE("1D")
    {
      D = 1;

      data::classification::Dataset data(D, N, Nc);
      std::string type = "Simple";
      data.Generate(type);

      std::tuple<arma::Mat<DTYPE>, arma::uvec> collect0;
      std::tuple<arma::Mat<DTYPE>, arma::uvec> collect1;
      collect0 = ac::extract_class(data.inputs_, data.labels_,0);
      collect1 = ac::extract_class(data.inputs_, data.labels_,1);

      arma::Mat<DTYPE> out0 = std::get<0>(collect0);
      arma::Mat<DTYPE> out1 = std::get<0>(collect1);

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

      data::classification::Dataset data(D, N, Nc);
      std::string type = "Simple";
      data.Generate(type);

      std::tuple<arma::Mat<DTYPE>, arma::uvec> collect0;
      std::tuple<arma::Mat<DTYPE>, arma::uvec> collect1;
      collect0 = ac::extract_class(data.inputs_, data.labels_,0);
      collect1 = ac::extract_class(data.inputs_, data.labels_,1);

      arma::Mat<DTYPE> out0 = std::get<0>(collect0);
      arma::Mat<DTYPE> out1 = std::get<0>(collect1);

      arma::uvec id0 = arma::regspace<arma::uvec>(0,1,N-1);
      arma::uvec id1 = arma::regspace<arma::uvec>(N,1,2*N-1);

      CHECK ( arma::norm((data.inputs_.cols(id0)-out0)) <= tol );
      CHECK ( arma::norm(data.inputs_.cols(id1)-out1) <= tol );

      CHECK ( arma::norm(std::get<1>(collect0) - id0) <=tol );
      CHECK ( arma::norm(std::get<1>(collect1) - id1) <=tol );
    }
}



TEST_SUITE("SPLIT-REGRESSION-DATASET") {
    
    size_t D = 1; 
    size_t N = 10; size_t Ntrn = 8; double testratio= 0.2;

    data::regression::Dataset dataset(D,N);
    data::regression::Dataset trainset;
    data::regression::Dataset testset;

    TEST_CASE("NUMBER")
    {
      dataset.Generate(std::string("Linear"),0.);

      data::Split(dataset,trainset,testset,Ntrn);

      arma::Mat<DTYPE> reconst_input = 
      arma::sort(arma::join_rows(trainset.inputs_,testset.inputs_),"ascend",1);

      arma::Mat<DTYPE> reconst_label = 
      arma::sort(arma::join_rows(trainset.labels_,testset.labels_),"ascend",1);

      arma::Mat<DTYPE> sortinp = arma::sort(dataset.inputs_,"ascend",1);
      arma::Mat<DTYPE> sortlab = arma::sort(dataset.labels_,"ascend",1);

      CHECK ( trainset.size_ == 8 );
      CHECK ( testset.size_ == 2 );
      CHECK ( arma::approx_equal(reconst_input, sortinp, "absdiff", 1e-3) );
      CHECK ( arma::approx_equal(reconst_label, sortlab, "absdiff", 1e-3) );
    }

    TEST_CASE("PERCENT")
    {
      dataset.Generate(std::string("Linear"),0.);

      data::Split(dataset,trainset,testset,testratio);

      arma::Mat<DTYPE> reconst_input = 
      arma::sort(arma::join_rows(trainset.inputs_,testset.inputs_),"ascend",1);

      arma::Mat<DTYPE> reconst_label = 
      arma::sort(arma::join_rows(trainset.labels_,testset.labels_),"ascend",1);

      arma::Mat<DTYPE> sortinp = arma::sort(dataset.inputs_,"ascend",1);
      arma::Mat<DTYPE> sortlab = arma::sort(dataset.labels_,"ascend",1);

      CHECK ( trainset.size_ == 8 );
      CHECK ( testset.size_ == 2 );
      CHECK ( arma::approx_equal(reconst_input, sortinp, "absdiff", 1e-3) );
      CHECK ( arma::approx_equal(reconst_label, sortlab, "absdiff", 1e-3) );
    }

}

TEST_SUITE("SPLIT-CLASSIFICATION-DATASET") {
    
    size_t D = 1; size_t Nc = 2;
    size_t N = 5; size_t Ntrn = 8; double testratio= 0.2;

    data::classification::Dataset dataset(D,N,Nc);
    data::classification::Dataset trainset;
    data::classification::Dataset testset;

    TEST_CASE("NUMBER")
    {
      dataset.Generate("Simple");

      data::Split(dataset,trainset,testset,Ntrn);

      arma::Mat<DTYPE> reconst_input = 
      arma::sort(arma::join_rows(trainset.inputs_,testset.inputs_),"ascend",1);

      arma::Row<size_t> reconst_label = 
      arma::sort(arma::join_rows(trainset.labels_,testset.labels_),"ascend",1);

      arma::Mat<DTYPE> sortinp = arma::sort(dataset.inputs_,"ascend",1);
      arma::Row<size_t> sortlab = arma::sort(dataset.labels_,"ascend",1);

      CHECK ( trainset.size_ == 8 );
      CHECK ( testset.size_ == 2 );
      CHECK ( arma::approx_equal(reconst_input, sortinp, "absdiff", 1e-3) );
      CHECK ( arma::approx_equal(reconst_label, sortlab, "absdiff", 1e-3) );
    }

    TEST_CASE("PERCENT")
    {
      dataset.Generate("Simple");

      data::Split(dataset,trainset,testset,testratio);

      arma::Mat<DTYPE> reconst_input = 
      arma::sort(arma::join_rows(trainset.inputs_,testset.inputs_),"ascend",1);

      arma::Row<size_t>reconst_label = 
      arma::sort(arma::join_rows(trainset.labels_,testset.labels_),"ascend",1);

      arma::Mat<DTYPE> sortinp = arma::sort(dataset.inputs_,"ascend",1);
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

    data::classification::Dataset dataset(D,N,Nc);
    data::classification::Dataset trainset;
    data::classification::Dataset testset;
    
    arma::Mat<DTYPE> traininp,testinp; arma::Row<size_t> trainlab,testlab;

    TEST_CASE("DATASET")
    {
      dataset.Generate("Simple");

      data::StratifiedSplit(dataset,trainset,testset,Ntrn);

      arma::Mat<DTYPE> reconst_input = 
      arma::sort(arma::join_rows(trainset.inputs_,testset.inputs_),"ascend",1);

      arma::Row<size_t> reconst_label = 
      arma::sort(arma::join_rows(trainset.labels_,testset.labels_),"ascend",1);

      arma::Mat<DTYPE> sortinp = arma::sort(dataset.inputs_,"ascend",1);
      arma::Row<size_t> sortlab = arma::sort(dataset.labels_,"ascend",1);

      CHECK ( trainset.size_ == 8 );
      CHECK ( testset.size_ == 2 );
      CHECK ( arma::approx_equal(reconst_input, sortinp, "absdiff", 1e-3) );
      CHECK ( arma::approx_equal(reconst_label, sortlab, "absdiff", 1e-3) );
    }

    TEST_CASE("EXPLICIT")
    {
      dataset.Generate("Simple");

      data::StratifiedSplit(dataset.inputs_, dataset.labels_,
                                   traininp, testinp,
                                   trainlab, testlab,
                                   Ntrn);

      arma::Mat<DTYPE> reconst_input = 
      arma::sort(arma::join_rows(traininp, testinp),"ascend",1);

      arma::Row<size_t> reconst_label = 
      arma::sort(arma::join_rows(trainlab, testlab),"ascend",1);

      arma::Mat<DTYPE> sortinp = arma::sort(dataset.inputs_,"ascend",1);
      arma::Row<size_t> sortlab = arma::sort(dataset.labels_,"ascend",1);

      CHECK ( traininp.n_cols == 8 );
      CHECK ( trainlab.n_cols == 8 );
      CHECK ( testinp.n_cols == 2 );
      CHECK ( testlab.n_cols == 2 );
      CHECK ( arma::approx_equal(reconst_input, sortinp, "absdiff", 1e-3) );
      CHECK ( arma::approx_equal(reconst_label, sortlab, "absdiff", 1e-3) );
    }

}

TEST_SUITE("Manipulation") {
  TEST_CASE("SetDiff")
  {
    arma::uvec a = {1,2,3,4};
    arma::uvec b = {1,3};
    arma::uvec expect = {2,4};

    auto res = data::SetDiff(a,b);
    CHECK ( arma::all(res == expect) );


  }
  
}

#
#endif
