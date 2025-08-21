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
    arma::Mat<DTYPE> X = { 1,2,3 };
    data::Gram<mlpack::LinearKernel> gram;
    {
      auto mat = gram.GetMatrix(X);
      CHECK( arma::approx_equal(mat,X.t()*X,"absdiff",1e-6) );
    }
    {
      auto mat = gram.GetMatrix2(X,X);
      CHECK( arma::approx_equal(mat,X*X.t(),"absdiff",1e-6) );
    }
    {
      auto mat = gram.GetApprox(X,X,3);
      CHECK( arma::approx_equal(mat,gram.GetMatrix(X),"absdiff",1e-6) ) ;
    }
  }

  TEST_CASE("TRANSFORM")
  {
    arma::Mat<DTYPE> X = {{1,2,3}};
    arma::Row<DTYPE> y = {-1.2247,0,1.2247};
    {
      data::Dataset<arma::Row<DTYPE>> dataset(X,y);
      data::Transformer trans(dataset);
      auto tdataset = trans.Trans(dataset);
      CHECK( arma::approx_equal(tdataset.inputs_,y,"absdiff",1e-3) ); 
      CHECK( arma::approx_equal(tdataset.labels_,y,"absdiff",1e-3) ); 
      auto itdataset = trans.InvTrans(tdataset);
      CHECK( arma::approx_equal(itdataset.inputs_,X,"absdiff",1e-3) ); 
      CHECK( arma::approx_equal(itdataset.labels_,y,"absdiff",1e-3) ); 
    }
    // Testing the Row<size_t> version
    {
      arma::Row<size_t> y_ = {0,1,1};
      data::Dataset<arma::Row<size_t>> dataset(X,y_);
      data::Transformer trans(dataset);
      auto tdataset = trans.Trans(dataset);
      CHECK( arma::approx_equal(tdataset.inputs_,y,"absdiff",1e-3) ); 
      CHECK( arma::approx_equal(tdataset.labels_,y_,"absdiff",1e-3) ); 
      auto itdataset = trans.InvTrans(tdataset);
      CHECK( arma::approx_equal(itdataset.inputs_,X,"absdiff",1e-3) ); 
      CHECK( arma::approx_equal(itdataset.labels_,y_,"absdiff",1e-3) ); 
    }
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
    {
      {
        arma::Mat<DTYPE> a = {1,2,3,4};
        arma::Mat<DTYPE> b = {5,6,7,8};

        arma::Row<DTYPE> a_ = a;
        arma::Row<DTYPE> b_ = b;

        data::Dataset<arma::Row<DTYPE>> to(a,a_);
        data::Dataset<arma::Row<DTYPE>> from(b,b_);

        data::Migrate(to,from,2);

        CHECK( to.size_ == 6 );
        CHECK( from.size_ == 2 );

        arma::Mat<DTYPE> expect {1,2,3,4,5,6,7,8};
        arma::Mat<DTYPE> comb = arma::unique(
                          arma::join_horiz(to.inputs_,from.inputs_)).eval().t();
        CHECK( arma::approx_equal(comb, expect,"absdiff",1e-6) );
        CHECK( arma::approx_equal(to.inputs_,
                           arma::conv_to<arma::Mat<DTYPE>>::from(to.labels_),
                           "absdiff",1e-6) );
      }

      {
        arma::Mat<DTYPE> a = {1,2,3,4};
        arma::Mat<DTYPE> b = {5,6,7,8};

        auto a_ = arma::conv_to<arma::Row<size_t>>::from(a);
        auto b_ = arma::conv_to<arma::Row<size_t>>::from(b);

        data::Dataset<arma::Row<size_t>> to(a,a_);
        data::Dataset<arma::Row<size_t>> from(b,b_);

        data::Migrate(to,from,2);

        CHECK( to.size_ == 6 );
        CHECK( from.size_ == 2 );

        arma::Mat<DTYPE> expect {1,2,3,4,5,6,7,8};
        arma::Mat<DTYPE> comb = arma::unique(
                          arma::join_horiz(to.inputs_,from.inputs_)).eval().t();
        CHECK( arma::approx_equal(comb, expect,"absdiff",1e-6) );
        CHECK( arma::approx_equal(to.inputs_,
                           arma::conv_to<arma::Mat<DTYPE>>::from(to.labels_),
                           "absdiff",1e-6) );
      }

    }
  }
  TEST_CASE("Split")
  {
    {
        arma::Mat<DTYPE> a = {1,2,3,4,5,6,7,8,9,10};
        arma::Row<DTYPE> a_ = a;

        data::Dataset<arma::Row<DTYPE>> dataset(a,a_);

        data::Dataset<arma::Row<DTYPE>> set1, set2;

        {
          data::Split(dataset, set1,set2, size_t(5));

          CHECK( set1.size_ == 5 );
          CHECK( set2.size_ == 5 );
          CHECK( dataset.size_ == 10 );

          CHECK( arma::approx_equal(set1.inputs_,set1.labels_,"absdiff",1e-12) );
          CHECK( arma::approx_equal(set2.inputs_,set2.labels_,"absdiff",1e-12) );
          arma::Mat<DTYPE> sorted = arma::sort(
                arma::join_horiz(set1.inputs_,set2.inputs_),"ascend",1);
          CHECK( arma::approx_equal(sorted,a,"absdiff",1e-12) );
        }
        {
          data::Split(dataset, set1,set2, 0.5);

          CHECK( set1.size_ == 5 );
          CHECK( set2.size_ == 5 );
          CHECK( dataset.size_ == 10 );

          CHECK( arma::approx_equal(set1.inputs_,set1.labels_,"absdiff",1e-12) );
          CHECK( arma::approx_equal(set2.inputs_,set2.labels_,"absdiff",1e-12) );
          arma::Mat<DTYPE> sorted = arma::sort(
                arma::join_horiz(set1.inputs_,set2.inputs_),"ascend",1);
          CHECK( arma::approx_equal(sorted,a,"absdiff",1e-12) );
        }
    }
    {
        arma::Mat<DTYPE> a = {1,2,3,4,5,6,7,8,9,10};
        auto a_ = arma::conv_to<arma::Row<size_t>>::from(a);

        data::Dataset<arma::Row<size_t>> dataset(a,a_);

        data::Dataset<arma::Row<size_t>> set1, set2;
        {
          data::Split(dataset,set1,set2,size_t(5));

          CHECK( set1.size_ == 5 );
          CHECK( set2.size_ == 5 );
          CHECK( dataset.size_ == 10 );

          CHECK( arma::approx_equal(set1.inputs_,
                arma::conv_to<arma::Mat<DTYPE>>::from(set1.labels_),
                "absdiff",1e-12) );

          CHECK( arma::approx_equal(set2.inputs_,
                arma::conv_to<arma::Mat<DTYPE>>::from(set2.labels_),
                "absdiff",1e-12) );

          arma::Mat<DTYPE> sorted = arma::sort(
                arma::join_horiz(set1.inputs_,set2.inputs_),"ascend",1);
          CHECK( arma::approx_equal(sorted,a,"absdiff",1e-12) );
        }
        {
          data::Split(dataset,set1,set2,double(0.5));

          CHECK( set1.size_ == 5 );
          CHECK( set2.size_ == 5 );
          CHECK( dataset.size_ == 10 );

          CHECK( arma::approx_equal(set1.inputs_,
                arma::conv_to<arma::Mat<DTYPE>>::from(set1.labels_),
                "absdiff",1e-12) );

          CHECK( arma::approx_equal(set2.inputs_,
                arma::conv_to<arma::Mat<DTYPE>>::from(set2.labels_),
                "absdiff",1e-12) );

          arma::Mat<DTYPE> sorted = arma::sort(
                arma::join_horiz(set1.inputs_,set2.inputs_),"ascend",1);
          CHECK( arma::approx_equal(sorted,a,"absdiff",1e-12) );
        }

    }
  }

  TEST_CASE("StratifiedSplit")
  {
    {
      arma::Mat<DTYPE> a = {1,2,3,4,5,6,7,8,9,10};
      arma::Row<size_t> a_ = {0,0,0,0,0,1,1,1,1,1} ;

      data::Dataset<arma::Row<size_t>> dataset(a,a_);

      data::Dataset<arma::Row<size_t>> set1, set2;
      {
        data::StratifiedSplit(dataset, set1, set2, size_t(2));

        CHECK( set1.size_ == 2 );
        CHECK( set2.size_ == 8 );
        CHECK( arma::unique(set1.labels_).eval().n_elem == 2 );
        CHECK( arma::unique(set2.labels_).eval().n_elem == 2 );
        CHECK( dataset.size_ == 10 );

        arma::uvec idx = sort_index(arma::join_horiz(set1.inputs_,set2.inputs_));
        arma::Mat<DTYPE> sorted = arma::sort(
              arma::join_horiz(set1.inputs_,set2.inputs_),"ascend",1);
        CHECK( arma::approx_equal(sorted,a,"absdiff",1e-12) );
        auto sorted_labels = arma::conv_to<arma::Row<size_t>>::from(
          arma::join_horiz(set1.labels_,set2.labels_).eval().cols(idx));
        CHECK( arma::approx_equal( sorted_labels, a_,"absdiff",1e-12) );
      }
      {
        data::StratifiedSplit(dataset, set1, set2, double(0.8));

        CHECK( set1.size_ == 2 );
        CHECK( set2.size_ == 8 );
        CHECK( arma::unique(set1.labels_).eval().n_elem == 2 );
        CHECK( arma::unique(set2.labels_).eval().n_elem == 2 );
        CHECK( dataset.size_ == 10 );

        arma::uvec idx = sort_index(arma::join_horiz(set1.inputs_,set2.inputs_));
        arma::Mat<DTYPE> sorted = arma::sort(
              arma::join_horiz(set1.inputs_,set2.inputs_),"ascend",1);
        CHECK( arma::approx_equal(sorted,a,"absdiff",1e-12) );
        auto sorted_labels = arma::conv_to<arma::Row<size_t>>::from(
          arma::join_horiz(set1.labels_,set2.labels_).eval().cols(idx));
        CHECK( arma::approx_equal( sorted_labels, a_,"absdiff",1e-12) );
      }

    }
    {
      arma::Mat<DTYPE> a = {1,2,3,4,5,6,7,8,9,10};
      arma::Row<size_t> a_ = {0,0,0,0,1,1,1,1,1,1} ;

      data::Dataset<arma::Row<size_t>> dataset(a,a_);

      data::Dataset<arma::Row<size_t>> set1, set2;
      {
        data::StratifiedSplit(dataset, set1, set2, size_t(4));
        
        CHECK( set1.size_ == 5 );
        CHECK( set2.size_ == 5 );
        CHECK( arma::unique(set1.labels_).eval().n_elem == 2 );
        CHECK( arma::unique(set2.labels_).eval().n_elem == 2 );
        CHECK( DTYPE(arma::accu(set1.labels_))/3. == 1. );
        CHECK( DTYPE(arma::accu(set2.labels_))/3. == 1. );
        CHECK( dataset.size_ == 10 );

        arma::uvec idx = sort_index(arma::join_horiz(set1.inputs_,set2.inputs_));
        arma::Mat<DTYPE> sorted = arma::sort(
              arma::join_horiz(set1.inputs_,set2.inputs_),"ascend",1);
        CHECK( arma::approx_equal(sorted,a,"absdiff",1e-12) );
        auto sorted_labels = arma::conv_to<arma::Row<size_t>>::from(
          arma::join_horiz(set1.labels_,set2.labels_).eval().cols(idx));
        CHECK( arma::approx_equal( sorted_labels, a_,"absdiff",1e-12) );
      }
    }
  }
}

TEST_SUITE("SAMPLE") 
{
  TEST_CASE("RandomSelect")
  {
    std::vector seeds = {1,2,3};
    for (size_t seed : seeds)
    {
      std::vector<std::pair<arma::uvec, arma::uvec>> sets;
      
      // Ns = {1,2,3}
      auto ns = arma::regspace<arma::Row<size_t>>(1, 3);

      // Instantiate functor
      data::RandomSelect<> selector;

      // Fill sets
      selector(5, ns, 1, sets, seed);
      std::set<std::string> seenPairs;
      // Check each split
      for (size_t i = 0; i < ns.n_elem; i++)
      {
        auto train = sets[i].first;
        auto test  = sets[i].second;

        // 1. Check sizes
        CHECK(train.n_elem == ns[i]);
        CHECK(test.n_elem == 5 - ns[i]);

        // 2. No overlap between train and test
        CHECK(arma::intersect(train, test).is_empty());

        // 3. No duplicates inside train/test
        CHECK(arma::unique(train).eval().n_elem == train.n_elem);
        CHECK(arma::unique(test).eval().n_elem == test.n_elem);

        // 4. All train-test pairs different
        std::ostringstream oss;
        train.t().raw_print(oss);
        test.t().raw_print(oss);
        auto repr = oss.str();
        CHECK(seenPairs.count(repr) == 0);
        seenPairs.insert(repr);
      }
    }
  }
  TEST_CASE("Additive")
  {
    std::vector<size_t> seeds = {1, 2, 3};
    for (size_t seed : seeds)
    {
      std::vector<std::pair<arma::uvec, arma::uvec>> sets;

      // Ns = {1,2,3}
      auto ns = arma::regspace<arma::Row<size_t>>(1, 3);

      // Instantiate functor
      data::Additive<> selector;

      // Fill sets
      selector(5, ns, 1, sets, seed);

      std::set<std::string> seenPairs;

      arma::uvec prevTrain;  // to check additive property

      // Check each split
      for (size_t i = 0; i < ns.n_elem; i++)
      {
        auto train = sets[i].first;
        auto test  = sets[i].second;

        // Sort for comparison
        arma::uvec sortedTrain = arma::sort(train);
        arma::uvec sortedTest  = arma::sort(test);

        // 1. Sizes
        CHECK(sortedTrain.n_elem == ns[i]);
        CHECK(sortedTest.n_elem == 5 - ns[i]);

        // 2. No overlap
        CHECK(arma::intersect(sortedTrain, sortedTest).is_empty());

        // 3. No duplicates
        CHECK(arma::unique(sortedTrain).eval().n_elem == sortedTrain.n_elem);
        CHECK(arma::unique(sortedTest).eval().n_elem == sortedTest.n_elem);

        // 4. Deterministic partition: train ∪ test = {0,1,2,3,4}
        arma::uvec all = arma::join_vert(sortedTrain, sortedTest);
        CHECK(arma::unique(all).eval().n_elem == 5);

        // 5. Additive property
        if (i > 0)
        {
          // New train must contain all previous train elements
          bool subset = true;
          for (size_t j = 0; j < prevTrain.n_elem; j++)
          {
            if (!arma::any(sortedTrain == prevTrain[j]))
            {
              subset = false;
              break;
            }
          }
          CHECK(subset);

          // And exactly one new element
          CHECK(sortedTrain.n_elem == prevTrain.n_elem + 1);
        }
        prevTrain = sortedTrain;

        // 6. All train-test pairs different
        std::ostringstream oss;
        sortedTrain.t().raw_print(oss);
        sortedTest.t().raw_print(oss);
        auto repr = oss.str();
        CHECK(seenPairs.count(repr) == 0);
        seenPairs.insert(repr);
      }
    }
  }
  TEST_CASE("Bootstrap")
  {
    std::vector<size_t> seeds = {1, 2, 3};
    for (size_t seed : seeds)
    {
      std::vector<std::pair<arma::uvec, arma::uvec>> sets;

      // Ns = {1,2,3}
      auto ns = arma::regspace<arma::Row<size_t>>(1, 3);

      // Instantiate functor
      data::Bootstrap<> selector;

      // Fill sets
      selector(5, ns, 1, sets, seed);

      std::set<std::string> seenPairs;

      // Check each split
      for (size_t i = 0; i < ns.n_elem; i++)
      {
        auto train = sets[i].first;
        auto test  = sets[i].second;

        // 1. Check sizes
        CHECK(train.n_elem == ns[i]);
        CHECK(test.n_elem >= 5 - ns[i]);

        // 2. No overlap between train and test
        CHECK(arma::intersect(train, test).is_empty());

        // 3. Allow duplicates in train, but test must be unique
        CHECK(arma::unique(test).eval().n_elem == test.n_elem);

        // 4. All train-test pairs different
        std::ostringstream oss;
        train.t().raw_print(oss);
        test.t().raw_print(oss);
        auto repr = oss.str();
        CHECK(seenPairs.count(repr) == 0);
        seenPairs.insert(repr);
      }
    }
  }
}
#endif
