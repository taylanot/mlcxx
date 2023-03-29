
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
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/util/error.h>
#include <jive/util/ItemSet.h>
#include <jive/util/EmptyItemGroup.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::EmptyItemGroup );


using jem::checkedCast;
using jem::newInstance;


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class EmptyItemGroup
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


EmptyItemGroup::EmptyItemGroup ()
{}


EmptyItemGroup::EmptyItemGroup

  ( const Ref<ItemSet>&  items ) : items_ ( items )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );
}


EmptyItemGroup::~EmptyItemGroup ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void EmptyItemGroup::readFrom ( ObjectInput& in )
{
  items_ = checkedCast<ItemSet> ( in.decodeObject() );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void EmptyItemGroup::writeTo ( ObjectOutput& out ) const
{
  out.encodeObject ( items_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> EmptyItemGroup::clone () const
{
  return newInstance<Self> ( items_ );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t EmptyItemGroup::size () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   getIDs
//----------------------------------------------------------------------


IdxVector EmptyItemGroup::getIDs () const
{
  return IdxVector ();
}


//-----------------------------------------------------------------------
//   getIndices
//----------------------------------------------------------------------


IdxVector EmptyItemGroup::getIndices () const
{
  return IdxVector ();
}


//-----------------------------------------------------------------------
//   getIndex
//----------------------------------------------------------------------


idx_t EmptyItemGroup::getIndex ( idx_t i ) const
{
  indexError ( JEM_FUNC, "item", i, 0 );

  return -1;
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* EmptyItemGroup::getItems () const
{
  return items_.get ();
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool EmptyItemGroup::contains ( idx_t iitem ) const
{
  return false;
}


//-----------------------------------------------------------------------
//   filter
//-----------------------------------------------------------------------


void EmptyItemGroup::filter ( const IdxVector& iitems ) const
{
  const idx_t  n = iitems.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    iitems[i] = -iitems[i] - 1;
  }
}


JIVE_END_PACKAGE( util )
