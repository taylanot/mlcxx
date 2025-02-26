/**
 * @file linprog.h
 * @author Ozgur Taylan Turan
 *
 * Linear Programming Routine
 *
 *
 */
#ifndef LINPROG_H
#define LINPROG_H
namespace opt {

/*  Solving the problems in the form:
 *
 *  min dot(c,x)
 *  s.t dot(G,x) < h
 *      dot(A,x) = b
 *  you can choose to have or not
 *      x_i > 0 for i=1,N
 *
 */
template<class T=DTYPE, class... Ts>
bool linprog ( arma::Row<T>& x,
               const arma::Row<T>& c, 
               const arma::Mat<T>& G,
               const arma::Row<T>& h, 
               const arma::Mat<T>& A=arma::Mat<T>(),
               const arma::Row<T>& b=arma::Row<T>(),
               bool positive = true,
               bool verbose = true,
               const Ts&... args )

{
  x.set_size(c.n_elem);
  std::vector<double> values; 
  std::vector<int> index, start;
  if (A.n_elem == 0) 
    utils::CSC(G,values,index,start);
  else
  {
    assert(b.n_elem == A.n_rows);
    arma::Mat<T> G_ = arma::join_cols(G,A);
    utils::CSC(G_,values,index,start);
  }

  HighsModel model;
  model.lp_.num_col_ = x.n_elem;
  model.lp_.num_row_ = G.n_rows+A.n_rows;
  model.lp_.sense_ = ObjSense::kMinimize;
  model.lp_.col_cost_ = arma::conv_to<std::vector<double>>::from(
                                          arma::conv_to<arma::rowvec>::from(c));
  /* model.lp_.col_cost_ = arma::conv_to<std::vector<double>>::from(c); */

  // the uper and lower bounds of your x 
  std::vector<double> col_lower;
  if (positive)
    col_lower = std::vector<double>(c.n_elem,0.);
  else
    col_lower = std::vector<double>(c.n_elem,-1.e10);

  std::vector<double> col_upper(c.n_elem,1.e10);
  model.lp_.col_lower_ = col_lower;
  model.lp_.col_upper_ = col_upper;

  // upper and lower bounds of the constraints 
  std::vector<double> row_lower(G.n_rows,-1e10);
  /* std::vector<T> row_upper = arma::conv_to<std::vector<T>>::from(h); */
  std::vector<double> row_upper = arma::conv_to<std::vector<double>>::from(
      arma::conv_to<arma::rowvec>::from(h));

  if (A.n_elem != 0)
  {
    assert(b.n_elem == A.n_rows);
    for (size_t i=0;i<b.n_elem;i++)
    {
      row_lower.push_back(b[i]);
      row_upper.push_back(b[i]);
    }
  }
 
  model.lp_.row_lower_ = row_lower; 
  model.lp_.row_upper_ = row_upper;
  model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;

  model.lp_.a_matrix_.start_ = start;
  model.lp_.a_matrix_.index_ = index;
  model.lp_.a_matrix_.value_ = values;

  // Create a Highs instance and shut it up!
  Highs highs;
  HighsStatus return_status;
  if (!verbose)
    highs.setOptionValue("output_flag", "false");

  //Pass the model 
  return_status = highs.passModel(model);

  // Well they did a such a clean job!
  const HighsLp& lp = highs.getLp();

  // Solve the model
  return_status = highs.run();
  bool status (return_status==HighsStatus::kOk ? 1 : 0); 

  const HighsSolution& solution = highs.getSolution();
  for (int col=0; col < lp.num_col_; col++)
    x[col] = solution.col_value[col];
  return status;
}


} // namespace opt
#endif
