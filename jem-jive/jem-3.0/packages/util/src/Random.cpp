
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <cmath>
#include <cstdlib>
#include <jem/base/assert.h>
#include <jem/base/MemCache.h>
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/util/Random.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class Random
//=======================================================================

/*
  The Random class generates random numbers with a uniform
  distribution in the interval [0,1]. It is based on the subtractive
  method by Knuth. See also "Numerical Recipes in C", second edition,
  Chapter 7, page 283.
*/

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Random::MAX_VALUE = 1000000000;

const int  Random::COEFFC_   = 56;
const int  Random::SEED_     = 131803398;


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


Random::Random ( int seed )
{
  coeffs_ = (int*) MemCache::alloc ( COEFFC_ * sizeof(int) );

  try
  {
    restart ( seed );
  }
  catch ( ... )
  {
    MemCache::dealloc ( coeffs_, COEFFC_ * sizeof(int) );
    throw;
  }
}


Random::Random ( const Random& rhs )
{
  coeffs_ = (int*) MemCache::alloc ( COEFFC_ * sizeof(int) );

  try
  {
    copy_ ( rhs );
  }
  catch ( ... )
  {
    MemCache::dealloc ( coeffs_, COEFFC_ * sizeof(int) );
    throw;
  }
}


Random::Random ( Random&& rhs ) : Random ()
{
  swap ( rhs );
}


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Random::~Random ()
{
  MemCache::dealloc ( coeffs_, COEFFC_ * sizeof(int) );

  coeffs_ = nullptr;
}


//-----------------------------------------------------------------------
//   copy assignment operator
//-----------------------------------------------------------------------


Random& Random::operator = ( const Random& rhs )
{
  if ( this != &rhs )
  {
    copy_ ( rhs );
  }

  return *this;
}


Random& Random::operator = ( Random&& rhs ) noexcept
{
  swap ( rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void Random::swap ( Random& rhs ) noexcept
{
  jem::swap ( coeffs_, coeffs_ );
  jem::swap ( ipos_,   ipos_   );
  jem::swap ( jpos_,   jpos_   );
  jem::swap ( igauss_, igauss_ );
  jem::swap ( xgauss_, xgauss_ );
}


//-----------------------------------------------------------------------
//   restart
//-----------------------------------------------------------------------


void Random::restart ( int seed )
{
  int*       cf = coeffs_;

  const int  M  = COEFFC_ - 1;

  int        c0 = ::abs ( SEED_ - ::abs( seed ) ) % MAX_VALUE;
  int        c1 = 1;

  cf[M] = c0;

  for ( int i = 1; i <= M - 1; i++ )
  {
    int  j = (21 * i) % M;

    cf[j] = c1;
    c1    = c0 - c1;

    if ( c1 < 0 )
    {
      c1 += MAX_VALUE;
    }

    c0 = cf[j];
  }

  for ( int i = 0; i < 4; i++ )
  {
    for ( int j = 1; j <= M; j++)
    {
      cf[j] -= cf[1 + (j + 30) % M];

      if ( cf[j] < 0 )
      {
        cf[j] += MAX_VALUE;
      }
    }
  }

  ipos_   = 0;
  jpos_   = 31;
  igauss_ = 0;
  xgauss_ = 0.0;
}


//-----------------------------------------------------------------------
//   next
//-----------------------------------------------------------------------


double Random::next ()
{
  return (double) nextValue() * (1.0 / (double) MAX_VALUE);
}


idx_t Random::next ( idx_t ubound )
{
  JEM_ASSERT ( ubound >= 0 );

  double  x = (double) nextValue() * (1.0 / (double) MAX_VALUE);
  idx_t   n = (idx_t) ((double) ubound * x);

  if ( n >= ubound )
  {
    n = ubound - 1_idx;
  }

  return n;
}


//-----------------------------------------------------------------------
//   nextValue
//-----------------------------------------------------------------------


int Random::nextValue ()
{
  int*  cf = coeffs_;

  ipos_++;
  jpos_++;

  if ( ipos_ >= COEFFC_ )
  {
    ipos_ = 1;
  }

  if ( jpos_ >= COEFFC_ )
  {
    jpos_ = 1;
  }

  int  n = cf[ipos_] - cf[jpos_];

  if ( n < 0 )
  {
    n += MAX_VALUE;
  }

  cf[ipos_] = n;

  return n;
}


//-----------------------------------------------------------------------
//   nextGaussian
//-----------------------------------------------------------------------


double Random::nextGaussian ()
{
  if ( ! igauss_ )
  {
    double  a, b, f, r;

    do
    {
      a = 2.0 * next() - 1.0;
      b = 2.0 * next() - 1.0;
      r =a * a + b * b;
    }
    while ( r >= 1.0 || r == 0.0 );

    f       = std::sqrt ( -2.0 * std::log( r ) / r );
    xgauss_ = a * f;
    igauss_ = 1;

    return b * f;
  }
  else
  {
    igauss_ = 0;

    return xgauss_;
  }
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void Random::readFrom ( io::DataInput& in )
{
  in.decode ( coeffs_, COEFFC_ );
  decode ( in, ipos_, jpos_, igauss_, xgauss_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void Random::writeTo ( io::DataOutput& out ) const
{
  out.encode ( coeffs_, COEFFC_ );
  encode ( out, ipos_, jpos_, igauss_, xgauss_ );
}


//-----------------------------------------------------------------------
//   copy_
//-----------------------------------------------------------------------


void Random::copy_ ( const Random& rhs )
{
  for ( int i = 0; i < COEFFC_; i++ )
  {
    coeffs_[i] = rhs.coeffs_[i];
  }

  ipos_   = rhs.ipos_;
  jpos_   = rhs.jpos_;
  igauss_ = rhs.igauss_;
  xgauss_ = rhs.xgauss_;
}


JEM_END_PACKAGE( util )
