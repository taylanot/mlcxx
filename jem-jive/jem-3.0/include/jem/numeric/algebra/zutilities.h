
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

#ifndef JEM_NUMERIC_ALGEBRA_ZUTILITIES_H
#define JEM_NUMERIC_ALGEBRA_ZUTILITIES_H

#include <jem/numeric/NumberTraits.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   private functions
//=======================================================================

JEM_BEGIN_NAMESPACE( algebra )

//-----------------------------------------------------------------------
//   norm2
//-----------------------------------------------------------------------


template <class T>

  T                       norm2

  ( const std::complex<T>*  v,
    idx_t                   n )

{
  T  s = NumberTraits<T>::zero ();
  T  re, im;

  for ( idx_t i = 0; i < n; i++ )
  {
    re = v[i].real ();
    im = v[i].imag ();

    s += re * re + im * im;
  }

  return sqrt ( s );
}


template <class T>

  T norm2 ( const Array<std::complex<T> >& v )

{
  const std::complex<T>*  vp = v.addr ();

  const idx_t  n  = v.size   ();
  const idx_t  st = v.stride ();

  T  s  = NumberTraits<T>::zero ();
  T  re, im;

  for ( idx_t i = 0; i < n; i++ )
  {
    re = vp[i * st].real ();
    im = vp[i * st].imag ();

    s += re * re + im * im;
  }

  return sqrt ( s );
}

JEM_END_NAMESPACE( algebra )

//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   norm2
//-----------------------------------------------------------------------


template <class T, int N, class E>

  inline T  norm2 ( const Tuple<std::complex<T>,N,1,E>&  v )

{
  T  t = v[0].real() * v[0].real() + v[0].imag() * v[0].imag();

  for ( int i = 1; i < N; i++ )
  {
    t += v[i].real() * v[i].real() + v[i].imag() * v[i].imag();
  }

  return sqrt ( t );
}


template <class T, class E>

  inline T  norm2 ( const Tuple<std::complex<T>,0,1,E>&  v )

{
  return T ();
}


template <class T, class E>

  inline T  norm2 ( const Tuple<std::complex<T>,1,1,E>&  v )

{
  T  t = v[0].real() * v[0].real() + v[0].imag() * v[0].imag();

  return sqrt ( t );
}


template <class T, class E>

  inline T  norm2 ( const Tuple<std::complex<T>,2,1,E>&  v )

{
  T  t =

    v[0].real() * v[0].real() + v[0].imag() * v[0].imag() +
    v[1].real() * v[1].real() + v[1].imag() * v[1].imag();

  return sqrt ( t );
}


template <class T, class E>

  inline T  norm2 ( const Tuple<std::complex<T>,3,1,E>&  v )

{
  T  t =

    v[0].real() * v[0].real() + v[0].imag() * v[0].imag() +
    v[1].real() * v[1].real() + v[1].imag() * v[1].imag() +
    v[2].real() * v[2].real() + v[2].imag() * v[2].imag();

  return sqrt ( t );
}


template <class T, class E>

  T  norm2 ( const Array<std::complex<T>,1,E>& v )

{
  const idx_t  n = v.size ();

  T   s = NumberTraits<T>::zero ();


  if ( v.isContiguous() )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      std::complex<T>  z = v.getFast (i);

      s += z.real() * z.real() + z.imag() * z.imag();
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      std::complex<T>  z = v[i];

      s += z.real() * z.real() + z.imag() * z.imag();
    }
  }

  return sqrt ( s );
}


template <class T>

  inline T  norm2 ( const Array<std::complex<T> >& v )

{
  if ( v.isContiguous() )
  {
    return algebra::norm2 ( v.addr(), v.size() );
  }
  else
  {
    return algebra::norm2 ( v );
  }
}


JEM_END_PACKAGE( numeric )

#endif
