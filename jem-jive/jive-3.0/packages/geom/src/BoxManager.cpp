
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


#include <jem/base/ClassTemplate.h>
#include <jive/util/Reordering.h>
#include <jive/geom/BoxManager.h>


JEM_DEFINE_CLASS( jive::geom::BoxManager );


JIVE_BEGIN_PACKAGE( geom )


//=======================================================================
//   class BoxManager
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


BoxManager::~BoxManager ()
{}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void BoxManager::reserve ( idx_t )
{}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void BoxManager::trimToSize ()
{}


//-----------------------------------------------------------------------
//   flushChanges
//-----------------------------------------------------------------------


void BoxManager::flushChanges ()
{}


//-----------------------------------------------------------------------
//   addBoxes
//-----------------------------------------------------------------------


idx_t BoxManager::addBoxes

  ( const Matrix&  boxes,
    idx_t          color )

{
  const idx_t  ibox = size ();
  const idx_t  n    = boxes.size (1);

  reserve ( ibox + n );

  for ( idx_t i = 0; i < n; i++ )
  {
    addBox ( boxes[i], color );
  }

  return ibox;
}


//-----------------------------------------------------------------------
//   eraseBoxes
//-----------------------------------------------------------------------


void BoxManager::eraseBoxes ( const IdxVector& iboxes )
{
  reorderBoxes ( util::makeEraseReordering( iboxes, size() ) );
}


//-----------------------------------------------------------------------
//   getBoxes
//-----------------------------------------------------------------------


void BoxManager::getBoxes ( const Matrix& boxes ) const
{
  const idx_t  n = size ();

  IdxVector    iboxes ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    iboxes[i] = i;
  }

  getSomeBoxes ( boxes, iboxes );
}


//-----------------------------------------------------------------------
//   getSomeBoxes
//-----------------------------------------------------------------------


void BoxManager::getSomeBoxes

  ( const Matrix&     boxes,
    const IdxVector&  iboxes ) const

{
  const idx_t  n = iboxes.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    getBox ( boxes[i], iboxes[i] );
  }
}


//-----------------------------------------------------------------------
//   setColors
//-----------------------------------------------------------------------


void BoxManager::setColors ( const IdxVector& colors )
{
  const idx_t  n = size ();

  IdxVector    iboxes ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    iboxes[i] = i;
  }

  setSomeColors ( iboxes, colors );
}


//-----------------------------------------------------------------------
//   updateBoxes
//-----------------------------------------------------------------------


void BoxManager::updateBoxes ( const Matrix& boxes )
{
  const idx_t  n = size ();

  IdxVector    iboxes ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    iboxes[i] = i;
  }

  updateSomeBoxes ( iboxes, boxes );
}


JIVE_END_PACKAGE( geom )
