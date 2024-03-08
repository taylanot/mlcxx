/**
 * @file parallel.cpp
 * @author Ozgur Taylan Turan
 *
 * To prevent some problems with parallelization
 */


namespace utils {

template<typename T>
T mod(size_t n, T a)
{
    return n - floor(n/a)%a;
}   

size_t threadder ( size_t jobs )
{
  arma::rowvec pos = 
                  arma::regspace<arma::rowvec>(1,1,int(omp_get_max_threads()));
  return pos(arma::max(arma::find(mod(jobs, pos) == 0)));  
}

} // namespace utils
