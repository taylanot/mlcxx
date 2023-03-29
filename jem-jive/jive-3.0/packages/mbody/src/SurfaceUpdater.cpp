
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
#include <jem/util/Event.h>
#include <jive/util/Reordering.h>
#include <jive/mbody/utilities.h>
#include <jive/mbody/XBodySet.h>
#include <jive/mbody/SurfaceUpdater.h>


JIVE_BEGIN_PACKAGE( mbody )


using jive::util::GroupSet;
using jive::util::XGroupSet;


//=======================================================================
//   class SurfaceUpdater
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SurfaceUpdater::SurfaceUpdater

  ( const XBodySet&         sbodies,
    const BodySet&          rbodies,
    const Ref<ShapeTable>&  shapes ) :

    sbodies_ ( sbodies.getData() ),
    rbodies_ ( rbodies.getData() ),
    shapes_  ( shapes )

{
  using jem::util::connect;

  JEM_PRECHECK ( rbodies );

  connect ( rbodies.newSizeEvent  (), this, & Self::newSizeHandler_  );
  connect ( rbodies.newTopoEvent  (), this, & Self::newTopoHandler_  );
  connect ( rbodies.newOrderEvent (), this, & Self::newOrderHandler_ );

  updated_ = false;

  rbodies.resetEvents ();
}


SurfaceUpdater::~SurfaceUpdater ()
{}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void SurfaceUpdater::update ()
{
  using jem::util::disconnect;

  if ( updated_ )
  {
    return;
  }

  Ref<GroupSet>  rbodies = rbodies_.get ();

  if ( ! rbodies )
  {
    sbodies_ = nullptr;
    shapes_  = nullptr;
    updated_ = true;

    return;
  }

  Ref<XGroupSet>   sbodies = sbodies_.get ();
  Ref<ShapeTable>  shapes  = shapes_ .get ();

  if ( shapes == nullptr || sbodies == nullptr )
  {
    disconnect ( rbodies->newSizeEvent,
                 this, & Self::newSizeHandler_ );
    disconnect ( rbodies->newTopoEvent,
                 this, & Self::newTopoHandler_ );
    disconnect ( rbodies->newOrderEvent,
                 this, & Self::newOrderHandler_ );

    rbodies_ = nullptr;
    sbodies_ = nullptr;
    shapes_  = nullptr;
    updated_ = true;

    return;
  }

  updateSurfaces ( XBodySet( sbodies ),
                   BodySet ( rbodies ), *shapes );

  rbodies->resetEvents ();

  updated_ = true;
}


//-----------------------------------------------------------------------
//   newSizeHandler_
//-----------------------------------------------------------------------


void SurfaceUpdater::newSizeHandler_ ()
{
  updated_ = false;
}


//-----------------------------------------------------------------------
//   newTopoHandler_
//-----------------------------------------------------------------------


void SurfaceUpdater::newTopoHandler_ ()
{
  updated_ = false;
}


//-----------------------------------------------------------------------
//   newOrderHandler_
//-----------------------------------------------------------------------


void SurfaceUpdater::newOrderHandler_

  ( const Reordering&  reord )

{
  Ref<ShapeTable>  shapes  = shapes_ .get ();
  Ref<XGroupSet>   sbodies = sbodies_.get ();


  if ( updated_        &&
       shapes          &&
       sbodies         &&
       sbodies->size() == reord.oldSize() )
  {
    sbodies->reorderGroups ( reord );
  }
  else
  {
    updated_ = false;
  }
}


//=======================================================================
//   related functions
//=======================================================================


Ref<Object>               newSurfaceUpdater

  ( const XBodySet&         sbodies,
    const  BodySet&         rbodies,
    const Ref<ShapeTable>&  shapes  )

{
  return jem::newInstance<SurfaceUpdater> ( sbodies,
                                            rbodies, shapes );
}


JIVE_END_PACKAGE( mbody )
