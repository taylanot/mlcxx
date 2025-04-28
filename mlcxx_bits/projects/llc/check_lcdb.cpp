/**
 * @file check_lcdb.cpp
 * @author Ozgur Taylan Turan
 *
 * Main file of mlcxx where you do not have to do anything...
 */

#include <headers.h>

void normalize( const std::filesystem::path& database_path )
{
  std::vector<std::filesystem::path> paths = 
    {database_path/"classification",database_path/"regression"};

  for(const std::filesystem::path& path: paths)
  {
    arma::mat train, test, ytrn, ytst; 
    arma::field<std::string> header_train, header_test; 

    train.load(arma::csv_name(path/"train_original.csv",
                                                      header_train));   
    test.load(arma::csv_name(path/"test_original.csv",
                                                      header_test));   
    arma::vec X = train.col(0);
    ytrn = train.cols(1,train.n_cols-1);
    ytst = test.cols(1,test.n_cols-1);
    arma::vec temp;

    for (size_t i=0; i<ytrn.n_cols; i++)
    {
      temp = ytrn.col(i);
      ytrn.col(i) = ytrn.col(i)/arma::trapz(X,temp).eval()(0);
      
    }

    for (size_t i=0; i<ytst.n_cols; i++)
    {
      temp = ytst.col(i);
      ytst.col(i) = ytst.col(i)/arma::trapz(X,temp).eval()(0);
    }

    arma::mat save_train = arma::join_horiz(X,ytrn);
    arma::mat save_test = arma::join_horiz(X,ytst);

    save_train.save(arma::csv_name(path/"train.csv",header_train));
    save_test.save(arma::csv_name(path/"test.csv",header_test));
  }
}

void split ( const std::filesystem::path& database_path )
{
  std::vector<std::filesystem::path> paths = 
  {database_path/"classification",database_path/"regression"};
  //{database_path/"regression"};

  for(const std::filesystem::path& path: paths)
  {
    auto splitted = utils::BulkLoadSplit(path,0.2);
    arma::mat trn, tst;
    arma::field<std::string> trn_head, tst_head;
    trn = std::get<0>(splitted);
    tst = std::get<1>(splitted);
    trn_head = std::get<2>(splitted);
    tst_head = std::get<3>(splitted);
    //PRINT_VAR(tst_head(0));
    //PRINT_VAR(trn_head(0));
    //PRINT_VAR(tst_head(tst.n_cols-10));
    //PRINT_VAR(trn_head(trn.n_cols-10));

    PRINT_VAR(trn_head.n_cols);
    PRINT_VAR(trn.n_cols);

    PRINT_VAR(tst_head.n_cols);
    PRINT_VAR(tst.n_cols);

    trn.save(arma::csv_name(path/"train_original.csv",trn_head));
    tst.save(arma::csv_name(path/"test_original.csv",tst_head));
  }
  
  
  //arma::field<int> header(1,10);
  //for (int i=0;i<10;i++)
  //{
  //  header(i) = i;
  //}
  //arma::uvec idx = {2,3,4,5,6,7};
  //PRINT(header)
  //arma::field<std::string> child1(2);
  //arma::field<std::string> child2(2);
  //arma::field<arma::field<std::string>> parent(2);
  //child1(0) = "ali";
  //child2(0) = "veli";
  //child1(1) = "celi";
  //child2(1) = "deli";

  //parent(0) = child1;
  //parent(1) = child2;
  //PRINT(parent);
  //arma::mat train = std::get<0>(splitted);
  //arma::mat test = std::get<1>(splitted);
  //utils::Save(path/"train_original.csv", train,false);
  //utils::Save(path/"test_original.csv", test,false);

  
}

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  split(".llc-paper/LCDB_0_12");
  normalize(".llc-paper/LCDB_0_12");
  return 0; 
}
