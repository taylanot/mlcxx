
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
#include <jem/base/ClassTemplate.h>
#include <jem/io/PrintWriter.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jive/util/ItemMap.h>
#include <jive/util/MemberSet.h>


JEM_DEFINE_CLASS( jive::util::MemberSet );


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class MemberSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MemberSet::MemberSet ( const String& name ) :

  Super ( name )

{}


MemberSet::~MemberSet ()
{}


//-----------------------------------------------------------------------
//   getItemName
//-----------------------------------------------------------------------


String MemberSet::getItemName () const
{
  return getCompoundItems()->getItemName() + " member";
}


//-----------------------------------------------------------------------
//   printSubset
//-----------------------------------------------------------------------


void MemberSet::printSubset

  ( PrintWriter&      out,
    const IdxVector&  imbrs ) const

{
  using jem::io::endl;
  using jem::io::space;

  const ItemMap*  mbrMap  = getItemMap        ();
  const ItemSet*  items   = getCompoundItems  ();
  const ItemMap*  itemMap = items->getItemMap ();

  const idx_t     n       = imbrs.size ();

  idx_t           iitem;
  idx_t           ilocal;


  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  imbr = imbrs[i];

    getMember ( iitem, ilocal, imbr );

    if ( i > 0 )
    {
      print ( out, endl );
    }

    print ( out,
            mbrMap ->getItemID( imbr ),  space,
            itemMap->getItemID( iitem ), space,
            ilocal );
  }
}


//-----------------------------------------------------------------------
//   getMembersOf
//-----------------------------------------------------------------------


IdxVector MemberSet::getMembersOf

  ( const IdxVector&  iitems ) const

{
  return getMembersOf ( iitems, getMembersPerItem() );
}


IdxVector MemberSet::getMembersOf

  ( const IdxVector&  iitems,
    const Topology&   topo )

{
  const idx_t  n = iitems.size ();

  IdxVector    itemOffsets = topo.getRowOffsets    ();
  IdxVector    mbrIndices  = topo.getColumnIndices ();

  IdxVector    imbrs;
  idx_t        iitem;
  idx_t        i, j, k, m;


  k = 0;

  for ( i = 0; i < n; i++ )
  {
    iitem = iitems[i];
    k    += (itemOffsets[iitem + 1] - itemOffsets[iitem]);
  }

  imbrs.resize ( k );

  j = 0;

  for ( i = 0; i < n; i++ )
  {
    iitem = iitems[i];
    m     = itemOffsets[iitem + 1];

    for ( k = itemOffsets[iitem]; k < m; k++ )
    {
      imbrs[j++] = mbrIndices[k];
    }
  }

  return imbrs;
}


//-----------------------------------------------------------------------
//   getMembersPerItem
//-----------------------------------------------------------------------


Topology MemberSet::getMembersPerItem () const
{
  const idx_t  itemCount = getCompoundItems()->size ();
  const idx_t  mbrCount  = size ();

  IdxVector    itemOffsets ( itemCount + 1 );
  IdxVector    mbrIndices;

  idx_t        iitem;
  idx_t        ilocal;
  idx_t        imbr;
  idx_t        i;


  itemOffsets = 0;

  for ( imbr = 0; imbr < mbrCount; imbr++ )
  {
    getMember ( iitem, ilocal, imbr );
    itemOffsets[iitem]++;
  }

  jem::numeric::SparseUtils::sumOffsets ( itemOffsets );

  mbrIndices.resize ( itemOffsets[itemCount] );

  for ( imbr = 0; imbr < mbrCount; imbr++ )
  {
    getMember ( iitem, ilocal, imbr );

    i             = itemOffsets[iitem];
    mbrIndices[i] = imbr;

    itemOffsets[iitem]++;
  }

  jem::numeric::SparseUtils::shiftOffsets ( itemOffsets );

  return Topology (
    jem::shape ( itemCount, mbrCount ),
    itemOffsets,
    mbrIndices
  );
}


JIVE_END_PACKAGE( util )
