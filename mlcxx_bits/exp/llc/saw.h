/**
 * @file saw.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef SAW_H 
#define SAW_H

//namespace exp {
//namespace llc {
//-----------------------------------------------------------------------------
//   _nchoosek
//
//   * Just a helper function for combinatin calculation
//    
//   @param n     : number of samples
//   @param k     : choosen number of samples
//-----------------------------------------------------------------------------
double _nchoosek ( const size_t& n,
                   const size_t& k )
{
  if ( k == 0 )
    return 1;
  else
    return boost::math::factorial<double> (n) / 
                                      ( boost::math::factorial<double> (k) *
                                        boost::math::factorial<double> (n-k) );
}

//-----------------------------------------------------------------------------
//   create_saw
//
//   * This function create a learning curve for absolute loss of a linear 
//    regression model for two point mass data distribution pA(prob) and 
//    pB(1-prob). Resulting curve has a saw like structure where the period 
//    of this saw shapes are deterimined by the prob. Only for Absolute Loss
//
//   @param N     : number of maximum training points 
//   @param prob  : initial probability of point A
//-----------------------------------------------------------------------------
std::tuple<arma::rowvec,
           arma::rowvec> create_saw ( const size_t& N,
                                      const double& prob )
{
  arma::rowvec R(N);
  double pA,pB;

  pA = prob; pB = 1-prob;

  double xA, yA, xB, yB;
  xA = 1; yA = 1; xB = 0.1; yB = 1;

  double wemp, Rt;
  size_t NB;
  
  for ( size_t M=1; M<N; M++ )
  {
    for ( size_t NA=0; NA < M; NA++ )
    {
      NB = M - NA;
      if ( NA*xA > NB*xB )
        wemp = yA / xA;
      else if ( NA*xA > NB*xB ) 
        wemp = yB / xB;
      else
        wemp = std::min(xA/yA,xB/yB);
      Rt = pA*std::abs(xA*wemp-yA) + pB*std::abs(xB*wemp-yB);
      R(M) += _nchoosek(M,NA)*std::pow(pA,NA)*std::pow(pB,NB)*Rt;
    }
  }
  return std::make_tuple(R,arma::regspace<arma::rowvec>(1,1,N));
}

//-----------------------------------------------------------------------------
//   create_saw
//
//   * This function create a learning curve for absolute loss of a linear 
//    regression model for two point mass data distribution pA(prob) and 
//    pB(1-prob). Resulting curve has a saw like structure where the period 
//    of this saw shapes are deterimined by the prob. Only for Absolute Loss
//
//   @param Ns    : training points to calculate Risk
//   @param prob  : initial probability of point A
//-----------------------------------------------------------------------------
arma::rowvec create_saw ( arma::rowvec& Ns,
                          const double& prob )
{
  arma::rowvec R(arma::size(Ns));
  double pA,pB;

  pA = prob; pB = 1-prob;

  double xA, yA, xB, yB;
  xA = 1; yA = 1; xB = 0.1; yB = 1;

  double wemp, Rt;
  size_t NB;
  
  arma::rowvec::iterator it = Ns.begin(); 
  arma::rowvec::iterator it_end = Ns.end(); 
  size_t  counter = 0;
  for ( ; it != it_end ; ++it )
  {
    for ( size_t NA=0; NA < (*it); NA++ )
    {
      NB = (*it) - NA;
      if ( NA*xA > NB*xB )
        wemp = yA / xA;
      else if ( NA*xA > NB*xB ) 
        wemp = yB / xB;
      else
        wemp = std::min(xA/yA,xB/yB);
      Rt = pA*std::abs(xA*wemp-yA) + pB*std::abs(xB*wemp-yB);
      R(counter) += _nchoosek((*it),NA)*std::pow(pA,NA)*std::pow(pB,NB)*Rt;
    }
    counter++;
  }
  return R;
}

//} // namespace llc
//} // namespace exp
#endif
