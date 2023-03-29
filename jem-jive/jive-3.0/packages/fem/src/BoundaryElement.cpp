
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
#include <jive/fem/BoundaryElement.h>


JEM_DEFINE_CLASS( jive::fem::BoundaryElement );


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   class BoundaryElement
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


BoundaryElement::BoundaryElement

  ( const ElementSet&   e,
    const Ref<BShape>&  s ) :

    Super ( e, s ),
    shape (    s )

{
  if ( s )
  {
    normals.resize ( rank, pntCount );
  }
}


BoundaryElement::~BoundaryElement ()
{}


//-----------------------------------------------------------------------
//   getNormals
//-----------------------------------------------------------------------


const Matrix& BoundaryElement::getNormals ()
{
  JEM_PRECHECK ( this->ielem >= 0 );

  switch ( pntType_ )
  {
  case IPOINTS:

    shape->getNormals ( normals, pntWeights, xnodes );

    break;

  case VERTICES:

    shape->getVertexNormals ( normals, xnodes );

    break;

  case CUSTOM:

    if ( normals.size(1) != pntCount )
    {
      normals.resize ( rank, pntCount );
    }

    for ( idx_t i = 0; i < pntCount; i++ )
    {
      shape->evalNormal ( normals[i], upoints[i], xnodes );
    }

    break;
  }

  return normals;
}


//-----------------------------------------------------------------------
//   setupChanged_
//-----------------------------------------------------------------------


void BoundaryElement::setupChanged_ ( int flags )
{
  Super::setupChanged_ ( flags );

  if ( flags & NEW_SHAPE_ )
  {
    shape = jem::staticCast<BShape> ( Super::shape );
  }

  normals.resize ( rank, pntCount );
}


JIVE_END_PACKAGE( fem )
