
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
#include <jem/util/Event.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/util/error.h>
#include <jive/fem/BoundaryManager.h>
#include <jive/fem/AutoBndGroup.h>


JEM_DEFINE_SERIABLE_CLASS( jive::fem::AutoBndGroup );


JIVE_BEGIN_PACKAGE( fem )


typedef jem::util::ArrayBuffer<idx_t>  IdxBuffer;


//=======================================================================
//   class AutoBndGroup
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


AutoBndGroup::AutoBndGroup ()
{
  kind_    = -1;
  updated_ =  false;
}


AutoBndGroup::AutoBndGroup

  ( Kind                    kind,
    const Ref<BndManager>&  bman ) :

    bman_ ( bman ),
    kind_ ( kind )

{
  JEM_PRECHECK ( bman );

  init_ ();
}


AutoBndGroup::~AutoBndGroup ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void AutoBndGroup::readFrom ( ObjectInput& in )
{
  decode ( in, bman_, kind_ );
  init_  ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void AutoBndGroup::writeTo ( ObjectOutput& out ) const
{
  encode ( out, bman_, kind_ );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t AutoBndGroup::size () const
{
  if ( ! updated_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  return ibounds_.size ();
}


//-----------------------------------------------------------------------
//   getIndices
//-----------------------------------------------------------------------


IdxVector AutoBndGroup::getIndices () const
{
  if ( ! updated_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  return ibounds_;
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* AutoBndGroup::getItems () const
{
  return bman_->getBoundaries().getData ();
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool AutoBndGroup::contains ( idx_t ibound ) const
{
  if ( ! updated_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  if ( ibound < 0 || ibound > buddies_.size() )
  {
    util::indexError ( JEM_FUNC, "boundary",
                       ibound,   buddies_.size() );
  }

  if ( kind_ == INTERNAL )
  {
    return (buddies_[ibound] >= 0);
  }
  else
  {
    return (buddies_[ibound] < 0);
  }
}


//-----------------------------------------------------------------------
//   filter
//-----------------------------------------------------------------------


void AutoBndGroup::filter ( const IdxVector& ibounds ) const
{
  if ( ! updated_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  const idx_t  bndCount  = buddies_.size ();
  const idx_t  ibndCount = ibounds .size ();

  idx_t        ibound;
  idx_t        ib;


  for ( ib = 0; ib < ibndCount; ib++ )
  {
    ibound = ibounds[ib];

    if ( ibound < 0 || ibound > bndCount )
    {
      util::indexError ( JEM_FUNC, "boundary", ibound, bndCount );
    }

    if ( kind_ == INTERNAL )
    {
      if ( buddies_[ibound] < 0 )
      {
        ibounds[ib] = -1;
      }
    }
    else
    {
      if ( buddies_[ibound] >= 0 )
      {
        ibounds[ib] = -1;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void AutoBndGroup::init_ ()
{
  using jem::util::connect;

  connect ( bman_->newBuddiesEvent, this, & Self::invalidate_ );

  updated_ = false;
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void AutoBndGroup::update_ ()
{
  IdxVector    buddies  = bman_->getBuddies ();

  const idx_t  bndCount = buddies.size ();

  IdxBuffer    ibuf;

  idx_t        ibound;
  idx_t        jbound;


  ibuf.reserve ( bndCount );

  if ( kind_ == INTERNAL )
  {
    for ( ibound = 0; ibound < bndCount; ibound++ )
    {
      jbound = buddies[ibound];

      if ( jbound > ibound )
      {
        ibuf.pushBack ( ibound );
        ibuf.pushBack ( jbound );
      }
    }
  }
  else
  {
    for ( ibound = 0; ibound < bndCount; ibound++ )
    {
      if ( buddies[ibound] < 0 )
      {
        ibuf.pushBack ( ibound );
      }
    }
  }

  ibuf.trimToSize ();

  buddies_.ref ( buddies );
  ibounds_.ref ( ibuf.toArray() );

  updated_ = true;

  bman_->resetEvents ();

  if ( ibounds_.size() > 0 )
  {
    newSizeEvent.emit ( ibounds_.size(), *this );
  }
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void AutoBndGroup::invalidate_ ()
{
  if ( updated_ )
  {
    IdxVector  dummy;

    updated_ = false;

    ibounds_.ref ( dummy );
    buddies_.ref ( dummy );

    newSizeEvent.emit ( 0, *this );
  }
}


JIVE_END_PACKAGE( fem )
