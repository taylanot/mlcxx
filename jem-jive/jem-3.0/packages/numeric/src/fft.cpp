
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
#include <jem/base/IllegalArgumentException.h>
#include <jem/numeric/fft.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   internal functions
//=======================================================================


static bool                 isPow2_

  ( idx_t                     n );

template <class T>

  static void               fft_

  ( T* JEM_RESTRICT           a,
    idx_t                     len,
    int                       isign );

template <class T>

  static void               rfft_

  ( T* JEM_RESTRICT           a,
    idx_t                     len,
    int                       isign );


//-----------------------------------------------------------------------
//   isPow2_
//-----------------------------------------------------------------------


bool isPow2_ ( idx_t n )
{
  JEM_ASSERT ( n >= 0 );

  uidx_t  k = (uidx_t) n;

  while ( k )
  {
    if ( k & 1 )
    {
      k = k >> 1;
      break;
    }

    k = k >> 1;
  }

  return (k == 0);
}


//-----------------------------------------------------------------------
//   fft_
//-----------------------------------------------------------------------


template <class T>

  void             fft_

  ( T* JEM_RESTRICT  a,
    idx_t            len,
    int              isign )

{
  double  wtemp, wr, wpr, wpi, wi, theta;
  T       tempr, tempi;
  idx_t   i, istep, j, m, mmax, n;


  if ( isign < 0 )
  {
    isign = -1_idx;
  }
  else
  {
    isign =  1_idx;
  }

  n = len << 1_idx;
  j = 1_idx;

  for ( i = 1; i < n; i += 2 )
  {
    if ( j > i )
    {
      jem::swap ( a[j],   a[i]   );
      jem::swap ( a[j+1], a[i+1] );
    }

    m = n >> 1_idx;

    while ( m >= 2 && j > m )
    {
      j  -= m;
      m >>= 1;
    }

    j += m;
  }

  // Danielson-Lanczos algorithm.

  mmax = 2_idx;

  while ( n > mmax )
  {
    istep = mmax << 1_idx;
    theta = isign * (6.28318530717959 / (double) mmax);
    wtemp = std::sin ( 0.5 * theta );
    wpr   = -2.0 * wtemp * wtemp;

    wpi   = std::sin ( theta );
    wr    = 1.0;
    wi    = 0.0;

    for ( m = 1; m < mmax; m += 2_idx )
    {
      for ( i = m; i <= n; i += istep )
      {
        j       = i + mmax;

        tempr   = (T) (wr * a[j]   - wi * a[j+1]);
        tempi   = (T) (wr * a[j+1] + wi * a[j]);

        a[j]    = a[i]   - tempr;
        a[j+1]  = a[i+1] - tempi;
        a[i]   += tempr;
        a[i+1] += tempi;
      }

      wtemp = wr;
      wr    = wtemp * wpr - wi    * wpi + wr;
      wi    = wi    * wpr + wtemp * wpi + wi;
    }

    mmax = istep;
  }
}


//-----------------------------------------------------------------------
//   rfft_
//-----------------------------------------------------------------------


template <class T>

  void             rfft_

  ( T* JEM_RESTRICT  a,
    idx_t            len,
    int              isign )

{
  double  wr, wi, wpr, wpi, wtemp, theta;
  T       c1, c2, h1r, h1i, h2r, h2i;
  idx_t   i, i1, i2, i3, i4, np3;


  c1    = (T) 0.5;
  theta = 3.141592653589793 / (double) (len >> 1);

  if ( isign > 0 )
  {
    fft_ ( a, len >> 1, isign );

    c2    = (T) -0.5;
  }
  else
  {
    c2    = (T)  0.5;
    theta = -theta;
  }

  wtemp = std::sin ( 0.5 * theta );
  wpr   = -2.0 * wtemp * wtemp;
  wpi   = std::sin ( theta );
  wr    = 1.0 + wpr;
  wi    = wpi;
  np3   = len + 3;

  for ( i = 2; i <= (len >> 2); i++ )
  {
    i1    =  i   + i - 1;
    i2    =  i1  + 1;
    i3    =  np3 - i2;
    i4    =  i3  + 1;

    h1r   =  c1 * (a[i1] + a[i3]);
    h1i   =  c1 * (a[i2] - a[i4]);
    h2r   = -c2 * (a[i2] + a[i4]);
    h2i   =  c2 * (a[i1] - a[i3]);

    a[i1] = (T) ( h1r + wr * h2r - wi * h2i);
    a[i2] = (T) ( h1i + wr * h2i + wi * h2r);
    a[i3] = (T) ( h1r - wr * h2r + wi * h2i);
    a[i4] = (T) (-h1i + wr * h2i + wi * h2r);

    wtemp = wr;
    wr    = wtemp * wpr - wi    * wpi + wr;
    wi    = wi    * wpr + wtemp * wpi + wi;
  }

  if ( isign > 0 )
  {
    h1r  = a[1];
    a[1] = h1r + a[2];
    a[2] = h1r - a[2];
  }
  else
  {
    h1r  = a[1];
    a[1] = c1 * (h1r + a[2]);
    a[2] = c1 * (h1r - a[2]);

    fft_ ( a, len >> 1, isign );
  }
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   fft
//-----------------------------------------------------------------------


void                    fft

  ( const Array<float>&   data,
    int                   isign )

{
  JEM_PRECHECK ( isign != 0 );

  const idx_t  len = data.size ();


  if ( len <= 1 )
  {
    return;
  }

  if ( ! isPow2_( len ) )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "invalid array size: %d; must be a power of two",
        len
      )
    );
  }

  if ( data.stride() == 1_idx )
  {
    rfft_ ( data.addr() - 1, len, isign );
  }
  else
  {
    Array<float>  temp ( len );

    temp = data;

    rfft_ ( temp.addr() - 1, len, isign );

    data = temp;
  }
}


void                    fft

  ( const Array<double>&  data,
    int                   isign )

{
  JEM_PRECHECK ( isign != 0 );

  const idx_t  len = data.size ();


  if ( len <= 1 )
  {
    return;
  }

  if ( ! isPow2_( len ) )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "invalid array size: %d; must be a power of two",
        len
      )
    );
  }

  if ( data.stride() == 1_idx )
  {
    rfft_ ( data.addr() - 1, len, isign );
  }
  else
  {
    Array<double>  temp ( len );

    temp = data;

    rfft_ ( temp.addr() - 1, len, isign );

    data = temp;
  }
}


JEM_END_PACKAGE( numeric )
