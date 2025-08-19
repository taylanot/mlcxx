/**
 * @file test_data.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_DATA_H 
#define TEST_DATA_H

TEST_SUITE("DATA") 
{
  TEST_CASE("GRAM")
  {

  }
  TEST_CASE("TRANSFORM")
  {

  }
}

TEST_SUITE("DATASET") 
{
  template<class DATASET>
  void check_size(DATASET dataset, size_t dimension,
                  size_t sample_size, size_t num_class=0)
  {
    CHECK ( dataset.inputs_.n_cols == sample_size );
    CHECK ( dataset.inputs_.n_rows == dimension );
    CHECK ( dataset.labels_.n_elem == sample_size );
    if constexpr (std::is_same<DATASET, data::Dataset<arma::Row<size_t>>>::value 
            || std::is_same<DATASET, data::oml::Dataset<size_t>>::value )
      CHECK ( arma::unique(dataset.labels_).eval().n_elem == num_class );
  }

  TEST_CASE("Sizes of Datasets and Collect")
  {
    data::oml::Dataset<size_t> omlclas(61);
    data::oml::Dataset<DTYPE> omlreg(44024);

    data::Dataset<arma::Row<size_t>> clas(3);
    data::Dataset<arma::Row<DTYPE>> reg(3);

    clas.Banana(10);
    reg.Linear(10);

    check_size(omlclas,4,150,3);
    check_size(omlreg,8,20640);
    check_size(reg,3,10);
    check_size(clas,2,20,2);

    {
      data::oml::Collect collect(283);
      CHECK ( collect.GetSize() == 53 );
    }

    {
      arma::Row<size_t> ids = {61};
      data::oml::Collect collect(ids);
      CHECK ( collect.GetSize() == 1 );
      auto dataset2 = collect.GetNext();
      check_size(dataset2,4,150,3);
    }
  }

  TEST_CASE("Read-oml::Dataset")
  {
    data::oml::Dataset<size_t> dataset(61);

    // Expected first input column
    arma::Col<DTYPE> first = {5.1, 3.5, 1.4, 0.2};
    CHECK(arma::approx_equal(dataset.inputs_.col(0), first, "absdiff", 1e-12));
    // Expected first label
    CHECK(dataset.labels_(0) == 0);

    // Expected last input column
    arma::Col<DTYPE> last = {5.9, 3.0, 5.1, 1.8};
    CHECK(arma::approx_equal(dataset.inputs_.col(dataset.size_ - 1), 
                              last, "absdiff", 1e-12));
    // Expected last label
    CHECK(dataset.labels_(dataset.size_ - 1) == 2);
  }

  TEST_CASE("Generate-Linear-1D")
  {
    data::Dataset<arma::Row<DTYPE>> dataset(1);
    dataset.Linear(10,1e-18);
    auto input = arma::conv_to<arma::Row<DTYPE>>::from(dataset.inputs_);
    CHECK( arma::approx_equal(input,dataset.labels_,"absdiff",1e-12) );
    dataset.Linear(10,1.);
    auto inputstd = arma::conv_to<arma::Row<DTYPE>>::from(dataset.inputs_);
    CHECK( !arma::approx_equal(inputstd,dataset.labels_,"absdiff",1e-12) );
  }

  TEST_CASE("Generate-Sine-1D")
  {
    data::Dataset<arma::Row<DTYPE>> dataset(1);
    dataset.Sine(10,1e-18);
    auto input = arma::conv_to<arma::Row<DTYPE>>::from(dataset.inputs_);
    CHECK( arma::approx_equal(arma::sin(input),dataset.labels_,
                              "absdiff",1e-12) );
    dataset.Sine(10,1.);
    auto inputstd = arma::conv_to<arma::Row<DTYPE>>::from(dataset.inputs_);
    CHECK( !arma::approx_equal(arma::sin(inputstd),dataset.labels_,
                                "absdiff",1e-12) );
  }

  TEST_CASE("Generate-Gaussian")
  {
    data::Dataset<arma::Row<size_t>> dataset(2);
    dataset.Gaussian(1000,{-1,1});
    arma::Mat<DTYPE>  means(2,1);
    CHECK( arma::approx_equal(arma::mean(dataset.inputs_,1),means,
                              "absdiff",1e-1) );
    CHECK ( double(arma::accu(dataset.labels_))/dataset.size_ == 0.5 );
  }

  TEST_CASE("Generate-Banana")
  {
    data::Dataset<arma::Row<size_t>> dataset(2);
    size_t N=1000;
    dataset.Banana(N);

    // Class splits
    arma::Mat<DTYPE> class0 = dataset.inputs_.cols(0, N-1);
    arma::Mat<DTYPE> class1 = dataset.inputs_.cols(N, 2*N-1);
    // Mean check
    arma::Col<DTYPE> mean0 = arma::mean(class0, 1);
    arma::Col<DTYPE> mean1 = arma::mean(class1, 1);

    CHECK(mean1(0) < mean0(0));   // class1 shifted left
    CHECK(arma::norm(mean0 - mean1, 2) > 0.5);
    // Range check
    arma::Row<DTYPE> radii = arma::sqrt(arma::sum(
                                        arma::square(dataset.inputs_),0));
    CHECK(arma::all(radii < 10.5));
    CHECK ( double(arma::accu(dataset.labels_))/dataset.size_ == 0.5 );
  }

  TEST_CASE("Generate-Dipping")
  {
    size_t N = 200;
    DTYPE r = 2.0;
    DTYPE noise_std = 0.05;
    // Generate dataset
    data::Dataset<arma::Row<size_t>> dataset(2);
    dataset.Dipping(N, r, noise_std);

    CHECK ( double(arma::accu(dataset.labels_))/dataset.size_ == 0.5 );

    // 3. Cluster separation
    arma::Mat<DTYPE> class0 = dataset.inputs_.cols(0, N-1);
    arma::Mat<DTYPE> class1 = dataset.inputs_.cols(N, 2*N-1);

    arma::Col<DTYPE> mean0 = arma::mean(class0, 1);
    arma::Col<DTYPE> mean1 = arma::mean(class1, 1);

    // Class0 points should have radius ~ r
    arma::Row<DTYPE> radii0 = arma::sqrt(arma::sum(arma::square(class0), 0));
    DTYPE avg_radius0 = arma::mean(radii0);
    CHECK(avg_radius0 == doctest::Approx(r).epsilon(0.2));

    // Class1 should be near origin
    DTYPE dist_origin = arma::norm(mean1, 2);
    CHECK(dist_origin == doctest::Approx(0.0).epsilon(0.2));

    // 4. Noise effect: higher std should spread points more
    data::Dataset<arma::Row<size_t>> dataset2(2);
    dataset2.Dipping(N, r, 0.5); // bigger noise
    DTYPE spread_low  = arma::mean(arma::sqrt(
                                  arma::sum(arma::square(dataset.inputs_), 0)));
    DTYPE spread_high = arma::mean(arma::sqrt(
                                  arma::sum(arma::square(dataset2.inputs_), 0)));
    CHECK(spread_high > spread_low);
  }
}

/* TEST_SUITE("TRANSFORM") { */

/*   double tol = 1e-6; */

/*   TEST_CASE("REGRESSION") */
/*   { */
/*     data::regression::Dataset data(2, 10); */
/*     data::regression::Dataset tdata,tbdata; */
/*     data.Generate(1,0,"Linear",0); */
/*     data::regression::Transformer trans(data); */
/*     tdata = trans.Trans(data); */
/*     tbdata = trans.InvTrans(tdata); */

/*     CHECK ( arma::sum(data.inputs_(0,0) - tbdata.inputs_(0,0))  <= tol ); */
/*     CHECK ( arma::sum(data.labels_(0,0) - tbdata.labels_(0,0))  <= tol ); */
    
/*   } */

/*   TEST_CASE("CLASSIFICATION") */
/*   { */
/*     data::classification::Dataset data(2, 10, 2); */
/*     data::classification::Dataset tdata,tbdata; */
/*     data.Generate("Simple"); */
/*     data::classification::Transformer trans(data); */
/*     tdata = trans.Trans(data); */
/*     tbdata = trans.InvTrans(tdata); */

/*     CHECK ( arma::sum(data.inputs_(0,0) - tbdata.inputs_(0,0))  <= tol ); */
/*     CHECK ( arma::sum(data.labels_(0,0) - tbdata.labels_(0,0))  <= tol ); */
    
/*   } */
/* } */


TEST_SUITE("MANIP") 
{
  TEST_CASE("SetDiff")
  {
    arma::uvec a = {1,2,3,4};
    arma::uvec b = {1,3};
    arma::uvec expect = {2,4};

    auto res = data::SetDiff(a,b);
    CHECK ( arma::all(res == expect) );
  }
  TEST_CASE("Migrate")
  {

  }
  TEST_CASE("Split")
  {

  }
  TEST_CASE("StratifiedSplit")
  {

  }
}

TEST_SUITE("SAMPLE") 
{
   TEST_CASE("RandomSelect")
  {

  }
  TEST_CASE("Additive")
  {

  }
  TEST_CASE("Bootstrap")
  {

  }
 
}
#endif
