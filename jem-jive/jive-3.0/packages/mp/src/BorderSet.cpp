
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
#include <jem/util/Properties.h>
#include <jive/util/ItemMap.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/XBorderSet.h>


JIVE_BEGIN_PACKAGE( mp )


//=======================================================================
//   class BorderSet
//=======================================================================

//-----------------------------------------------------------------------
//   getBorderItemIDs
//-----------------------------------------------------------------------


idx_t BorderSet::getBorderItemIDs

  ( const IdxVector&   itemIDs,
    idx_t              iborder ) const

{
  ItemSet*     items  = borders_().getGroupedItems ();
  IdxVector    iitems ( itemIDs.size() );

  const idx_t  n = getBorderItems   ( iitems, iborder );

  items->getItemMap()->getItemIDs ( itemIDs[slice(BEGIN,n)],
                                    iitems [slice(BEGIN,n)] );

  return n;
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void BorderSet::store

  ( BorderKind         kind,
    const Properties&  globdat ) const

{
  JEM_PRECHECK ( *this );

  Properties  itemdat = Globdat::getDataFor ( getItems(), globdat );

  if ( kind & RECV_BORDERS )
  {
    itemdat.set ( Globdat::RECV_BORDERS, data_ );
  }

  if ( kind & SEND_BORDERS )
  {
    itemdat.set ( Globdat::SEND_BORDERS, data_ );
  }
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


BorderSet BorderSet::find

  ( BorderKind           kind,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Properties  itemdat = Globdat::getDataFor ( items, globdat );

  Ref<Data>   borders;

  if ( kind & RECV_BORDERS )
  {
    itemdat.find ( borders, Globdat::RECV_BORDERS );
  }

  if ( kind & SEND_BORDERS )
  {
    itemdat.find ( borders, Globdat::SEND_BORDERS );
  }

  return BorderSet ( borders );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


BorderSet BorderSet::get

  ( BorderKind           kind,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Properties  itemdat = Globdat::getDataFor ( items, globdat );

  Ref<Data>   borders;

  if ( kind & RECV_BORDERS )
  {
    itemdat.find ( borders, Globdat::RECV_BORDERS );
  }

  if ( kind & SEND_BORDERS )
  {
    itemdat.find ( borders, Globdat::SEND_BORDERS );
  }

  if ( ! borders )
  {
    XBorderSet xborders ( newXBorderSet( kind, items ) );

    xborders.store ( kind, globdat );

    borders = xborders.getData ();
  }

  return BorderSet ( borders );
}


//-----------------------------------------------------------------------
//   hasBorders
//-----------------------------------------------------------------------


bool BorderSet::hasBorders

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Properties  itemdat = Globdat::getDataFor ( items, globdat );

  return ( itemdat.contains( Globdat::RECV_BORDERS ) ||
           itemdat.contains( Globdat::SEND_BORDERS ) );
}


JIVE_END_PACKAGE( mp )
