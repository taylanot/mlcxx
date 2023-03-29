
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


#include <jem/base/array/utilities.h>
#include <jive/util/private/NameList.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


NameList::NameList () :

  isSorted_ ( true )

{}


NameList::NameList ( const StringVector& names ) :

  names_    ( names ),
  isSorted_ ( false )

{}


//-----------------------------------------------------------------------
//   addName
//-----------------------------------------------------------------------


idx_t NameList::addName ( const String& name )
{
  idx_t  index = linearSearch_ ( name );

  if ( index < 0 )
  {
    index     = names_.size ();
    isSorted_ = false;

    names_.pushBack ( name );
  }

  return index;
}


//-----------------------------------------------------------------------
//   linearSearch_
//-----------------------------------------------------------------------


idx_t NameList::linearSearch_ ( const String& name ) const
{
  const idx_t  n = names_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( name == names_[i] )
    {
      return i;
    }
  }

  return -1_idx;
}


//-----------------------------------------------------------------------
//   binarySearch_
//-----------------------------------------------------------------------


idx_t NameList::binarySearch_ ( const String& name ) const
{
  if ( ! isSorted_ )
  {
    const_cast<Self*>( this )->sortNames_ ();
  }

  idx_t  first = 0;
  idx_t  last  = names_.size ();

  while ( first < last )
  {
    idx_t  i   = first + (last - first) / 2;
    idx_t  cmp = name.compareTo ( names_[ iperm_[i] ] );

    if      ( cmp < 0 )
    {
      last = i;
    }
    else if ( cmp > 0 )
    {
      first = i + 1;
    }
    else
    {
      return iperm_[i];
    }
  }

  return -1_idx;
}


//-----------------------------------------------------------------------
//   sortNames_
//-----------------------------------------------------------------------


void NameList::sortNames_ ()
{
  const idx_t  n = names_.size ();

  if ( iperm_.size() != n )
  {
    iperm_.resize ( n );
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    iperm_[i] = i;
  }

  jem::sort ( iperm_, names_.toArray() );

  isSorted_ = true;
}


JIVE_END_PACKAGE( util )
