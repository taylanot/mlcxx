
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


#include <jem/base/assert.h>
#include <jem/base/utilities.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/utilities.h>
#include <jem/io/PrintWriter.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemMap.h>
#include <jive/util/GroupSet.h>


JEM_DEFINE_CLASS( jive::util::GroupSet );


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class GroupSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GroupSet::GroupSet ( const String& name ) :

  Super ( name )

{}


GroupSet::~GroupSet ()
{}


//-----------------------------------------------------------------------
//   getItemName
//-----------------------------------------------------------------------


String GroupSet::getItemName () const
{
  return ( getGroupedItems()->getItemName() + " group" );
}


//-----------------------------------------------------------------------
//   getSomeGroupMembers
//-----------------------------------------------------------------------


void GroupSet::getSomeGroupMembers

  ( const IdxVector&  iitems,
    const IdxVector&  index,
    idx_t             igroup ) const

{
  JEM_PRECHECK2 ( iitems.size() >= index.size(),
                  "item index array too small" );

  const idx_t  n    = index.size ();

  IdxVector    itmp ( getGroupSize( igroup ) );

  getGroupMembers ( itmp, igroup );

  for ( idx_t i = 0; i < n; i++ )
  {
    iitems[i] = itmp[ index[i] ];
  }
}


//-----------------------------------------------------------------------
//   getOneGroupMember
//-----------------------------------------------------------------------


idx_t GroupSet::getOneGroupMember ( idx_t index, idx_t igroup ) const
{
  IdxVector  itmp ( getGroupSize ( igroup ) );

  getGroupMembers ( itmp, igroup );

  return itmp[index];
}


//-----------------------------------------------------------------------
//   getMembersOf
//-----------------------------------------------------------------------


IdxVector GroupSet::getMembersOf

  ( const IdxVector&  igroups ) const

{
  const idx_t  n = igroups.size ();

  IdxVector    ibuf ( maxGroupSize() );
  IdxVector    iitems;

  idx_t        igroup;
  idx_t        i, j, k, m;


  k = 0;

  for ( i = 0; i < n; i++ )
  {
    k += getGroupSize ( igroups[i] );
  }

  iitems.resize ( k );

  j = 0;

  for ( i = 0; i < n; i++ )
  {
    igroup = igroups[i];
    m      = getGroupMembers ( ibuf, igroup );

    for ( k = 0; k < m; k++ )
    {
      iitems[j++] = ibuf[k];
    }
  }

  return iitems;
}


//-----------------------------------------------------------------------
//   getUniqueMembersOf
//-----------------------------------------------------------------------


IdxVector GroupSet::getUniqueMembersOf

  ( const IdxVector&  igroups ) const

{
  const idx_t  n = getGroupedItems()->size ();

  return makeUnique ( getMembersOf( igroups ), 0, n - 1 ).clone ();
}


//-----------------------------------------------------------------------
//   maxGroupSize
//-----------------------------------------------------------------------


idx_t GroupSet::maxGroupSize () const
{
  const idx_t  groupCount = size ();

  idx_t        maxSize    = 0;

  for ( idx_t igroup = 0; igroup < groupCount; igroup++ )
  {
    maxSize = jem::max ( maxSize, getGroupSize( igroup ) );
  }

  return maxSize;
}


//-----------------------------------------------------------------------
//   maxGroupSizeOf
//-----------------------------------------------------------------------


idx_t GroupSet::maxGroupSizeOf ( const IdxVector& igroups ) const
{
  const idx_t  igroupCount = igroups.size ();

  idx_t        maxSize     = 0;

  for ( idx_t i = 0; i < igroupCount; i++ )
  {
    maxSize = jem::max ( maxSize, getGroupSize( igroups[i] ) );
  }

  return maxSize;
}


//-----------------------------------------------------------------------
//   toVector
//-----------------------------------------------------------------------


IdxVector GroupSet::toVector () const
{
  const idx_t  n = size ();

  IdxVector    igroups ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    igroups[i] = i;
  }

  return getMembersOf ( igroups );
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


Topology GroupSet::toMatrix () const
{
  const idx_t  groupCount = size ();
  const idx_t  itemCount  = getGroupedItems()->size ();

  IdxVector    groupOffsets ( groupCount + 1 );
  IdxVector    itemIndices;

  idx_t        last = 0;


  for ( idx_t igroup = 0; igroup < groupCount; igroup++ )
  {
    groupOffsets[igroup] = getGroupSize ( igroup );
    last += groupOffsets[igroup];
  }

  itemIndices.resize ( last );

  last = 0;

  for ( idx_t igroup = 0; igroup < groupCount; igroup++ )
  {
    idx_t  n = groupOffsets[igroup];

    groupOffsets[igroup] = last;

    getGroupMembers ( itemIndices[ slice(last,last + n) ], igroup );

    last += n;
  }

  groupOffsets[groupCount] = last;

  return Topology ( jem::shape ( groupCount, itemCount ),
                    groupOffsets, itemIndices );
}


//-----------------------------------------------------------------------
//   printSubset
//-----------------------------------------------------------------------


void GroupSet::printSubset

  ( PrintWriter&      out,
    const IdxVector&  igroups ) const

{
  using jem::io::endl;
  using jem::io::space;

  const ItemMap*  groupMap = getItemMap        ();
  const ItemSet*  items    = getGroupedItems   ();
  const ItemMap*  itemMap  = items->getItemMap ();

  const idx_t     n        = igroups.size ();

  IdxMatrix       ibuf     ( maxGroupSize(), 2 );

  IdxVector       itemIDs  ( ibuf[0] );
  IdxVector       iitems   ( ibuf[1] );
  IdxVector       jitems;


  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  igroup = igroups[i];
    idx_t  k      = getGroupMembers ( iitems, igroup );

    if ( k != jitems.size() )
    {
      jitems.ref ( iitems[slice(BEGIN,k)] );
    }

    if ( i > 0 )
    {
      print ( out, endl );
    }

    print ( out, groupMap->getItemID( igroup ) );

    itemMap->getItemIDs ( itemIDs, jitems );

    for ( idx_t j = 0; j < k; j++ )
    {
      print ( out, space, itemIDs[j] );
    }

    print ( out, ';' );
  }
}


JIVE_END_PACKAGE( util )
