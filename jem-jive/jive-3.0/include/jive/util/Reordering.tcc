
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_UTIL_REORDERING_TCC
#define JIVE_UTIL_REORDERING_TCC


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   private helper functions
//=======================================================================

//-----------------------------------------------------------------------
//   reorder2
//-----------------------------------------------------------------------


template <class T, int N>

  void                    reorder2

  ( const jem::Array<T,N>&  data,
    const jem::Array<T,N>&  orig,
    const Reordering&       reord )

{
  const idx_t* JEM_RESTRICT  rmap = reord.rmapPtr ();

  const idx_t  oldSize = orig .size    ();
  const idx_t  newSize = reord.newSize ();

  for ( idx_t i = reord.start(); i < newSize; i++ )
  {
    idx_t  j = rmap[i];

    if ( j < oldSize )
    {
      data[i] = orig[j];
    }
  }
}


template <class T>

  void                    reorder2

  ( const jem::Array<T,2>&  data,
    const jem::Array<T,2>&  orig,
    const Reordering&       reord )

{
  const idx_t* JEM_RESTRICT  rmap = reord.rmapPtr ();

  const idx_t  oldSize = orig .size    ();
  const idx_t  newSize = reord.newSize ();

  const idx_t  size0   = orig.size (0);

  for ( idx_t j = reord.start(); j < newSize; j++ )
  {
    idx_t  k = rmap[j];

    if ( k < oldSize )
    {
      for ( idx_t i = 0; i < size0; i++ )
      {
        data(i,j) = orig(i,k);
      }
    }
  }
}


template <class T>

  void                    reorder2

  ( const jem::Array<T,3>&  data,
    const jem::Array<T,3>&  orig,
    const Reordering&       reord )

{
  const idx_t* JEM_RESTRICT  rmap = reord.rmapPtr ();

  const idx_t  oldSize = orig .size    ();
  const idx_t  newSize = reord.newSize ();

  const idx_t  size0   = orig.size (0);
  const idx_t  size1   = orig.size (1);

  for ( idx_t k = reord.start(); k < newSize; k++ )
  {
    idx_t  p = rmap[k];

    if ( p < oldSize )
    {
      for ( idx_t j = 0; j < size1; j++ )
      {
        for ( idx_t i = 0; i < size0; i++ )
        {
          data(i,j,k) = orig(i,j,p);
        }
      }
    }
  }
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   reorder
//-----------------------------------------------------------------------


template <class T, int N>

  void                    reorder

  ( jem::Array<T,N>&        data,
    const Reordering&       reord,
    T                       fill )

{
  using jem::Tuple;

  if ( reord.isSimple() && data.size() >= reord.oldSize() )
  {
    reorder2     ( data, data, reord );
    data.reshape ( reord.newSize() );
  }
  else
  {
    Tuple<idx_t,N>   sh = data.shape ();

    sh[N - 1] = reord.newSize ();

    jem::Array<T,N>  tmp ( sh );

    idx_t  n = jem::min ( reord.start(), data.size() );

    tmp[slice(BEGIN,n)] = data[slice(BEGIN,n)];
    tmp[slice(n,END)]   = fill;

    reorder2  ( tmp, data, reord );
    data.swap ( tmp );
  }
}


template <class T, int N>

  void                    reorder

  ( jem::Array<T,N>&        data,
    int                     idim,
    const Reordering&       reord,
    T                       fill )

{
  if ( idim == (N - 1) )
  {
    reorder ( data, reord, fill );
  }
  else
  {
    JEM_PRECHECK2 ( idim >= 0 && idim < N,
                    "invalid dimension to reorder" );

    jem::Tuple<int,N>  iperm;

    for ( int i = 0; i < N; i++ )
    {
      iperm[i] = i;
    }

    jem::swap ( iperm[N - 1], iperm[idim] );

    jem::Array<T,N>  tmp = data.transpose ( iperm );

    reorder  ( tmp, reord, fill );
    data.ref ( tmp.transpose( iperm ) );
  }
}


template <class T>

  inline void             reorder

  ( jem::Array<T>&          data,
    int                     idim,
    const Reordering&       reord,
    T                       fill = T() )

{
  JEM_PRECHECK2 ( idim == 0, "invalid dimension to reorder "
                  "(should be zero)");

  reorder ( data, reord, fill );
}


JIVE_END_PACKAGE( util )

#endif
