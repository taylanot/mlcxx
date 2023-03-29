
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


#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/select.h>
#include <jive/util/Reordering.h>


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class Reordering
//=======================================================================

//-----------------------------------------------------------------------
//   default constructor
//-----------------------------------------------------------------------


Reordering::Reordering ()
{
  start_   = 0;
  newSize_ = 0;
  simple_  = true;
}


//-----------------------------------------------------------------------
//   forward map constructor
//-----------------------------------------------------------------------


Reordering::Reordering ( const IdxVector& fmap ) : fmap_ ( fmap )
{
  const idx_t*  JEM_RESTRICT  fp;
  idx_t*        JEM_RESTRICT  rp;

  const idx_t   osize = fmap.size ();

  bool          ok;

  idx_t         nsize;
  idx_t         start;
  idx_t         i, j;


  if ( ! fmap_.isContiguous() )
  {
    fmap_.ref ( fmap.clone() );
  }

  fp = fmap_.addr ();

  for ( i = 0; i < osize && fp[i] == i; i++ ) ;

  start_  = start = i;
  simple_ = true;

  for ( i = nsize = start; i < osize; i++ )
  {
    if ( fp[i] >= 0 )
    {
      if ( fp[i] != nsize )
      {
        simple_ = false;
      }

      nsize++;
    }
  }

  newSize_ = nsize;

  rmap_.resize ( osize );

  rp = rmap_.addr ();

  for ( i = 0; i < start; i++ )
  {
    rp[i] = i;
  }

  for ( i = start; i < nsize; i++ )
  {
    rp[i] = -1;
  }

  ok = true;
  j  = nsize;

  for ( i = start; i < osize; i++ )
  {
    if ( fp[i] >= 0 )
    {
      if ( fp[i] >= nsize || rp[fp[i]] >= 0 )
      {
        ok = false;
      }
      else
      {
        rp[fp[i]] = i;
      }
    }
    else
    {
      rp[j++] = i;
    }
  }

  ok = ok && (j == osize);

  if ( ! ok )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      "invalid forward map array"
    );
  }
}


//-----------------------------------------------------------------------
//   reverse map constructor
//-----------------------------------------------------------------------


Reordering::Reordering

  ( const IdxVector&  rmap,
    idx_t             nsize ) :

    rmap_ ( rmap )

{
  JEM_PRECHECK2 ( nsize >= 0 && nsize <= rmap.size(),
                  "invalid Reordering size" );

  const idx_t*  JEM_RESTRICT  rp;
  idx_t*        JEM_RESTRICT  fp;

  const idx_t   osize = rmap.size ();

  bool          ok;

  idx_t         start;
  idx_t         i, j;


  newSize_ = nsize;

  if ( ! rmap_.isContiguous() )
  {
    rmap_.ref ( rmap.clone() );
  }

  fmap_.resize ( osize );

  rp = rmap_.addr ();
  fp = fmap_.addr ();

  for ( i = 0; i < nsize && rp[i] == i; i++ ) ;

  start_ = start = i;

  for ( i = 0; i < start; i++ )
  {
    fp[i] = i;
  }

  for ( i = start; i < osize; i++ )
  {
    fp[i] = -1;
  }

  simple_ = true;

  for ( i = start + 1; i < nsize; i++ )
  {
    if ( rp[i] <= rp[i - 1] )
    {
      simple_ = false;
    }
  }

  ok = true;

  for ( i = start; i < nsize; i++ )
  {
    j = rp[i];

    if ( j < 0 || j >= osize || fp[j] >= 0 )
    {
      ok = false;
    }
    else
    {
      fp[j] = i;
    }
  }

  for ( i = nsize; i < osize; i++ )
  {
    j = rp[i];

    if ( j < 0 || j >= osize || fp[j] >= 0 )
    {
      ok = false;
    }
  }

  if ( ! ok )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      "invalid reverse map array"
    );
  }
}


//-----------------------------------------------------------------------
//   copy constructor
//-----------------------------------------------------------------------


Reordering::Reordering ( const Reordering& rhs ) :

  fmap_ ( rhs.fmap_ ),
  rmap_ ( rhs.rmap_ )

{
  start_   = rhs.start_;
  newSize_ = rhs.newSize_;
  simple_  = rhs.simple_;
}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


Reordering& Reordering::operator = ( const Reordering& rhs )
{
  fmap_.ref ( rhs.fmap_ );
  rmap_.ref ( rhs.rmap_ );

  start_   = rhs.start_;
  newSize_ = rhs.newSize_;
  simple_  = rhs.simple_;

  return *this;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   makeEraseReordering
//-----------------------------------------------------------------------


Reordering          makeEraseReordering

  ( const IdxVector&  imask,
    idx_t             osize )

{
  idx_t*     JEM_RESTRICT  fp;

  IdxVector  fmap ( osize );

  idx_t      i, j;


  fmap        =  0;
  fmap[imask] = -1;

  fp = fmap.addr ();

  for ( i = j = 0; i < osize; i++ )
  {
    if ( fp[i] == 0 )
    {
      fp[i] = j++;
    }
  }

  return Reordering ( fmap );
}


JIVE_END_PACKAGE( util )
