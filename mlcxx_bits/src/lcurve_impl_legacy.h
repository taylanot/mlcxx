/**
 * @file lcurve_impl.h
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
 *
 * TODO: 
 * - 
 * - LOSS in the bare version is not doing anything! Why did I put it there?
 *   # Now, I remember why... It is planned for situations when the 
 *   ComputeError or ComputeAccuracy are not available!
 * - omp parallel is not working with NN implementations because two omp 
 *   parallels are recursing probably?!
 *   
 *
 *
 */

#ifndef LCURVE_IMPL_LEGACY_H
#define LCURVE_IMPL_LEGACY_H

namespace src {
namespace regression {

//=============================================================================
// LCurve
//=============================================================================

template<class MODEL,
         class LOSS>
LCurve<MODEL,LOSS>::LCurve ( const arma::irowvec& Ns,
                             const double& repeat ) :
repeat_(repeat), Ns_(Ns)
{
  test_errors_.resize(repeat_,Ns_.n_elem);
  train_errors_.resize(repeat_,Ns_.n_elem);
} 
      
template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::Generate ( const std::filesystem::path filename,
                                    const arma::mat& inputs,
                                    const arma::rowvec& labels,
                                    const Ts&... args )
{
  Generate(inputs, labels, args...);
  Save(filename);
}

template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::Generate ( const bool save_all, 
                                    const std::filesystem::path filename,
                                    const arma::mat& inputs,
                                    const arma::mat& labels,
                                    const Ts&... args )
{
  if (!save_all)
    Generate(inputs, labels, args...);
  else
  {
    BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
          "There are not enough data for test set creation!" );
    std::filesystem::path parent;
    if (filename.has_relative_path())
      parent = filename.parent_path();
    else
      parent = filename.stem();

  ProgressBar pb("LCurveGenerate", Ns_.n_elem*repeat_);
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      std::filesystem::path dir = parent/std::to_string(i)/std::to_string(j);
      std::filesystem::create_directories(dir);


      const auto res = utils::data::Split(inputs, labels, size_t(Ns_(i)));
      arma::mat Xtrn = std::get<0>(res);
      arma::mat Xtst = std::get<1>(res);
      arma::mat ytrn = std::get<2>(res);
      arma::mat ytst = std::get<3>(res);

      utils::Save(dir/"train_inp.csv", Xtrn);
      utils::Save(dir/"train_lab.csv", ytrn);
      utils::Save(dir/"test_inp.csv", Xtst);
      utils::Save(dir/"test_lab.csv", ytst);

      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = model.ComputeError(Xtst, ytst);
      train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
      /* pb.Update(); */
    }
  }

    arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                      arma::stddev(train_errors_));
    arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                     arma::stddev(test_errors_));

    results_ = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

    stats_ = std::make_tuple(std::move(train),
                             std::move(test));

    utils::Save(parent/"train.csv",train_errors_);
    utils::Save(parent/"test.csv",test_errors_);
  }

  Save(filename);
}
template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::Generate ( const bool save_all, 
                                    const std::filesystem::path filename,
                                    const arma::mat& inputs,
                                    const arma::rowvec& labels,
                                    const Ts&... args )
{
  if (!save_all)
    Generate(inputs, labels, args...);
  else
  {
    BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
          "There are not enough data for test set creation!" );
    std::filesystem::path parent;
    if (filename.has_relative_path())
      parent = filename.parent_path();
    else
      parent = filename.stem();

    ProgressBar pb("LCurve.Generate", Ns_.n_elem*repeat_);
    #pragma omp parallel for collapse(2)
    for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
    {
      for(size_t j=0; j < size_t(repeat_); j++)
      {
        std::filesystem::path dir = parent/std::to_string(i)/std::to_string(j);
        std::filesystem::create_directories(dir);

        const auto res = utils::data::Split(inputs, labels, Ns_(i));
        arma::mat Xtrn = std::get<0>(res);
        arma::mat Xtst = std::get<1>(res);
        arma::rowvec ytrn = std::get<2>(res);
        arma::rowvec ytst = std::get<3>(res);

        utils::Save(dir/"train_inp.csv", Xtrn);
        utils::Save(dir/"train_lab.csv", ytrn);
        utils::Save(dir/"test_inp.csv", Xtst);
        utils::Save(dir/"test_lab.csv", ytst);

        MODEL model(Xtrn, ytrn, args...);
        test_errors_(j,i) = model.ComputeError(Xtst, ytst);
        train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
        /* pb.Update(); */
      }
    }

      arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                        arma::stddev(train_errors_));
      arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                       arma::stddev(test_errors_));

      results_ = 
        arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

      stats_ = std::make_tuple(std::move(train),
                               std::move(test));
  }

  Save(filename);
}

template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::Generate ( const std::filesystem::path filename,
                                    const arma::mat& inputs,
                                    const arma::mat& labels,
                                    const Ts&... args )
{
  Generate(inputs, labels, args...);
  Save(filename);
}

template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::Generate ( const arma::mat& inputs,
                                    const arma::rowvec& labels,
                                    const Ts&... args )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );

  ProgressBar pb("LCurve.Generate", Ns_.n_elem*repeat_);
  #pragma omp parallel for collapse(2)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = utils::data::Split(inputs, labels, size_t(Ns_(i)));
      arma::mat Xtrn = std::get<0>(res);
      arma::mat Xtst = std::get<1>(res);
      arma::rowvec ytrn = std::get<2>(res);
      arma::rowvec ytst = std::get<3>(res);
      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = model.ComputeError(Xtst, ytst);
      train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
      /* pb.Update(); */
    }
  }

    arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                      arma::stddev(train_errors_));
    arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                     arma::stddev(test_errors_));

    results_ = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

    stats_ = std::make_tuple(std::move(train),
                             std::move(test));
}

template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::Generate ( const arma::mat& inputs,
                                    const arma::mat& labels,
                                    const Ts&... args )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );

  ProgressBar pb("LCurve.Generate", Ns_.n_elem*repeat_);
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = utils::data::Split(inputs, labels, size_t(Ns_(i)));
      arma::mat Xtrn = std::get<0>(res);
      arma::mat Xtst = std::get<1>(res);
      arma::mat ytrn = std::get<2>(res);
      arma::mat ytst = std::get<3>(res);

      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = model.ComputeError(Xtst, ytst);
      train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
      /* pb.Update(); */
    }
  }

    arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                      arma::stddev(train_errors_));
    arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                     arma::stddev(test_errors_));

    results_ = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

    stats_ = std::make_tuple(std::move(train),
                             std::move(test));
}

template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::Generate ( const utils::data::regression::Dataset& 
                                                                      trainset,
                                    const utils::data::regression::Dataset&
                                                                      testset,
                                    const Ts&... args )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(trainset.inputs_.n_cols), 
        "There are not enough data for learning curve generation!" );
  try
  {
    BOOST_ASSERT_MSG( size_t(trainset.labels_.n_rows) == size_t(1), 
        "Training labels is not 1D..." );

    ProgressBar pb("LCurve.Generate", Ns_.n_elem*repeat_);
    for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
    {
      for(size_t j=0; j < size_t(repeat_); j++)
      {
        const auto res = utils::data::Split(trainset.inputs_,
                                            arma::conv_to<arma::rowvec>
                                              ::from(trainset.labels_),
                                            size_t(Ns_(i)));
        arma::mat Xtrn = std::get<0>(res);
        arma::rowvec ytrn = std::get<2>(res);

        MODEL model(Xtrn, ytrn, args...);
        test_errors_(j,i) = model.ComputeError(testset.inputs_,
                                               arma::conv_to<arma::rowvec>
                                                ::from(testset.labels_));
        train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
        /* pb.Update(); */
      }
    }

      arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                        arma::stddev(train_errors_));
      arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                       arma::stddev(test_errors_));

      results_ = 
        arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

      stats_ = std::make_tuple(std::move(train),
                               std::move(test)); 
  }
  catch(...)
  {
    PRINT("Trying the matrix version (without parallelization)...")
    #pragma omp for
    for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
    {
      for(size_t j=0; j < size_t(repeat_); j++)
      {
        const auto res = utils::data::Split(trainset.inputs_,
                                            trainset.labels_,
                                            size_t(Ns_(i)));
        arma::mat Xtrn = std::get<0>(res);
        arma::mat ytrn = std::get<2>(res);

        MODEL model(Xtrn, ytrn, args...);
        test_errors_(j,i) = model.ComputeError(testset.inputs_,testset.labels_);
        train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
      }
    }

      arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                        arma::stddev(train_errors_));
      arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                       arma::stddev(test_errors_));

      results_ = 
        arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

      stats_ = std::make_tuple(std::move(train),
                               std::move(test));
  }
  

}

template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::ParallelGenerate 
                                  ( const utils::data::regression::Dataset& 
                                                                      trainset,
                                    const utils::data::regression::Dataset&
                                                                      testset,
                                    const Ts&... args )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(trainset.inputs_.n_cols), 
        "There are not enough data for learning curve generation!" );
  try
  {
    BOOST_ASSERT_MSG( size_t(trainset.labels_.n_rows) == size_t(1), 
        "Training labels is not 1D..." );

    ProgressBar pb("ParallelGenerate", Ns_.n_elem*repeat_);
    #pragma omp parallel for collapse (2)
    for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
    {
      for(size_t j=0; j < size_t(repeat_); j++)
      {
        const auto res = utils::data::Split(trainset.inputs_,
                                            arma::conv_to<arma::rowvec>
                                              ::from(trainset.labels_),
                                            size_t(Ns_(i)));
        arma::mat Xtrn = std::get<0>(res);
        arma::rowvec ytrn = std::get<2>(res);

        MODEL model(Xtrn, ytrn, args...);
        test_errors_(j,i) = model.ComputeError(testset.inputs_,
                                               arma::conv_to<arma::rowvec>
                                                ::from(testset.labels_));
        train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
        /* pb.Update(); */
      }
    }

      arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                        arma::stddev(train_errors_));
      arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                       arma::stddev(test_errors_));

      results_ = 
        arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

      stats_ = std::make_tuple(std::move(train),
                               std::move(test)); 

  }
  catch(...)
  {

    ProgressBar pb("ParallelGenerate", Ns_.n_elem*repeat_);
    #pragma omp parallel for collapse (2)
    for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
    {
      for(size_t j=0; j < size_t(repeat_); j++)
      {
        const auto res = utils::data::Split(trainset.inputs_,
                                            trainset.labels_,
                                            size_t(Ns_(i)));
        arma::mat Xtrn = std::get<0>(res);
        arma::mat ytrn = std::get<2>(res);

        MODEL model(Xtrn, ytrn, args...);
        test_errors_(j,i) = model.ComputeError(testset.inputs_,testset.labels_);
        train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
        /* pb.Update(); */
      }
    }

      arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                        arma::stddev(train_errors_));
      arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                       arma::stddev(test_errors_));

      results_ = 
        arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

      stats_ = std::make_tuple(std::move(train),
                               std::move(test));
  }
  

}
template<class MODEL,
         class LOSS>
template<class... Ts>
void  LCurve<MODEL,LOSS>::Save ( const std::filesystem::path filename )
{
  utils::Save(filename, results_);
}

//=============================================================================
// VariableLCurve
//=============================================================================

template<class MODEL,
         class LOSS>
VariableLCurve<MODEL,LOSS>::VariableLCurve ( const arma::irowvec& Ns,
                                             const arma::irowvec& repeat ) :
repeat_(repeat), Ns_(Ns) 
{
  test_errors_.resize(repeat_.max(),Ns_.n_elem);
  train_errors_.resize(repeat_.max(),Ns_.n_elem);
} 
      
template<class MODEL,
         class LOSS>
template<class... Ts>
void VariableLCurve<MODEL,LOSS>::Generate ( const arma::mat& inputs,
                                            const arma::rowvec& labels,
                                            const Ts&... args )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );

  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    #pragma omp parallel for 
    for(size_t j=0; j < size_t(repeat_(i)); j++)
    {
      const auto res = utils::data::Split(inputs, labels, Ns_(i));
      arma::mat Xtrn = std::get<0>(res);
      arma::mat Xtst = std::get<1>(res);
      arma::mat ytrn = std::get<2>(res);
      arma::mat ytst = std::get<3>(res);
      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = model.ComputeError(Xtst, ytst);
      train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
    }

    if (repeat_(i) != repeat_.max())
    {
      train_errors_.col(i).rows(repeat_(i),repeat_.max()-1).fill(arma::datum::nan);
      test_errors_.col(i).rows(repeat_(i),repeat_.max()-1).fill(arma::datum::nan);
    }
  }

  arma::mat train(2,Ns_.n_elem);
  arma::mat test(2,Ns_.n_elem);

  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
  {
    train(0,i) = arma::mean(train_errors_.col(i).eval()
                           (arma::find_finite(train_errors_.col(i)).eval()));
    train(1,i) = arma::stddev(train_errors_.col(i).eval()
                           (arma::find_finite(train_errors_.col(i)).eval()));

    test(0,i) = arma::mean(test_errors_.col(i).eval()
                           (arma::find_finite(test_errors_.col(i)).eval()));
    test(1,i) = arma::stddev(test_errors_.col(i).eval()
                           (arma::find_finite(test_errors_.col(i)).eval()));
  }


  results_ = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

  stats_ = std::make_tuple(std::move(train),
                             std::move(test));
}

template<class MODEL,
         class LOSS>
template<class... Ts>
void  VariableLCurve<MODEL,LOSS>::Save ( const std::filesystem::path filename )
{
  utils::Save(filename, results_);
}

} // namespace regression
} // namespace src

namespace src {
namespace classification {

//=============================================================================
// LCurve
//=============================================================================

template<class MODEL,
         class LOSS>
LCurve<MODEL,LOSS>::LCurve ( const arma::irowvec& Ns,
                             const double& repeat ) :
repeat_(repeat), Ns_(Ns)
{
  test_errors_.resize(repeat_,Ns_.n_elem);
  train_errors_.resize(repeat_,Ns_.n_elem);
} 
      
template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::Generate ( const std::filesystem::path filename,
                                    const arma::mat& inputs,
                                    const arma::Row<size_t>& labels,
                                    const Ts&... args )
{
  Generate(inputs, labels, args...);
  Save(filename);
}

template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::Generate ( const arma::mat& inputs,
                                    const arma::Row<size_t>& labels,
                                    const Ts&... args )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );

  #pragma omp parallel for collapse(2)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = utils::data::Split(inputs,
                                          labels, size_t(Ns_(i)));
      arma::mat Xtrn = std::get<0>(res);
      arma::mat Xtst = std::get<1>(res);
      arma::Row<size_t> ytrn = std::get<2>(res);
      arma::Row<size_t> ytst = std::get<3>(res);
      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = (100. - model.ComputeAccuracy(Xtst, ytst)) / 100.;
      train_errors_(j,i) = (100. - model.ComputeAccuracy(Xtrn, ytrn)) / 100.;
    }

    arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                      arma::stddev(train_errors_));
    arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                     arma::stddev(test_errors_));

    results_ = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

    stats_ = std::make_tuple(std::move(train),
                             std::move(test));
}

template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::StratifiedGenerate ( const arma::mat& inputs,
                                              const arma::Row<size_t>& labels,
                                              const Ts&... args )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(inputs.n_cols), 
        "There are not enough data for test set creation!" );

  #pragma omp parallel for collapse(2)
  for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
    for(size_t j=0; j < size_t(repeat_); j++)
    {
      const auto res = utils::data::StratifiedSplit(inputs,
                                          labels, size_t(Ns_(i)));
      arma::mat Xtrn = std::get<0>(res);
      arma::mat Xtst = std::get<1>(res);
      arma::Row<size_t> ytrn = std::get<2>(res);
      arma::Row<size_t> ytst = std::get<3>(res);
      MODEL model(Xtrn, ytrn, args...);
      test_errors_(j,i) = (100. - model.ComputeAccuracy(Xtst, ytst)) / 100.;
      train_errors_(j,i) = (100. - model.ComputeAccuracy(Xtrn, ytrn)) / 100.;
    }

    arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                      arma::stddev(train_errors_));
    arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                     arma::stddev(test_errors_));

    results_ = 
      arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

    stats_ = std::make_tuple(std::move(train),
                             std::move(test));
}
template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::StratifiedGenerate 
                                  ( const utils::data::classification::Dataset& 
                                                                      trainset,
                                    const utils::data::classification::Dataset&
                                                                      testset,
                                    const Ts&... args )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(trainset.inputs_.n_cols), 
        "There are not enough data for learning curve generation!" );

    #pragma omp parallel for collapse (2)
    for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
      for(size_t j=0; j < size_t(repeat_); j++)
      {
        const auto res = utils::data::StratifiedSplit(trainset.inputs_,
                                            trainset.labels_,
                                            size_t(Ns_(i)));
        arma::mat Xtrn = std::get<0>(res);
        arma::Row<size_t> ytrn = std::get<2>(res);

        MODEL model(Xtrn, ytrn, args...);
        test_errors_(j,i) = model.ComputeError(testset.inputs_,
                                               testset.labels_);
        train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
      }

      arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                        arma::stddev(train_errors_));
      arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                       arma::stddev(test_errors_));

      results_ = 
        arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

      stats_ = std::make_tuple(std::move(train),
                               std::move(test)); 
  

}

template<class MODEL,
         class LOSS>
template<class... Ts>
void LCurve<MODEL,LOSS>::Generate 
                                  ( const utils::data::classification::Dataset& 
                                                                      trainset,
                                    const utils::data::classification::Dataset&
                                                                      testset,
                                    const Ts&... args )
{

  BOOST_ASSERT_MSG( int(Ns_.max()) < int(trainset.inputs_.n_cols), 
        "There are not enough data for learning curve generation!" );

    #pragma omp parallel for collapse (2)
    for (size_t i=0; i < size_t(Ns_.n_elem) ; i++)
      for(size_t j=0; j < size_t(repeat_); j++)
      {
        const auto res = utils::data::Split(trainset.inputs_,
                                            trainset.labels_,
                                            size_t(Ns_(i)));
        arma::mat Xtrn = std::get<0>(res);
        arma::Row<size_t> ytrn = std::get<2>(res);

        MODEL model(Xtrn, ytrn, args...);
        test_errors_(j,i) = model.ComputeError(testset.inputs_,
                                               testset.labels_);
        train_errors_(j,i) = model.ComputeError(Xtrn, ytrn);
      }

      arma::mat train = arma::join_cols(arma::mean(train_errors_),
                                        arma::stddev(train_errors_));
      arma::mat test = arma::join_cols(arma::mean(test_errors_),
                                       arma::stddev(test_errors_));

      results_ = 
        arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns_), train, test);

      stats_ = std::make_tuple(std::move(train),
                               std::move(test)); 
  

}


template<class MODEL,
         class LOSS>
template<class... Ts>
void  LCurve<MODEL,LOSS>::Save ( const std::filesystem::path filename )
{
  utils::Save(filename, results_);
}

} // namespace classification
} // namespace src

#endif
