
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
#include <jem/base/IllegalOperationException.h>
#include <jem/base/ClassTemplate.h>
#include <jive/geom/Shape.h>


JEM_DEFINE_CLASS ( jive::geom::Shape );


JIVE_BEGIN_PACKAGE( geom )


//=======================================================================
//   class Shape
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Shape::Shape ( const String& name ) :

  Super ( name )

{}


Shape::~Shape ()
{}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String Shape::getContext () const
{
  return NamedObject::makeContext ( "shape", myName_ );
}


//-----------------------------------------------------------------------
//   isRegular
//-----------------------------------------------------------------------


bool Shape::isRegular () const
{
  idx_t  n = nodeCount ();

  return (n == vertexCount    () &&
          n == shapeFuncCount ());
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String Shape::getGeometry () const
{
  return String ();
}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void Shape::getGlobalIntegrationPoints

  ( const Matrix&  x,
    const Matrix&  c ) const

{
  JEM_PRECHECK2 ( x.size(1) == integrationPointCount(),
                  "coordinate array has wrong shape" );

  Matrix  s = getIntegrationScheme ();

  for ( idx_t i = 0; i < s.size(1); i++ )
  {
    getGlobalPoint ( x[i], s(slice(1,END),i), c );
  }
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool Shape::getLocalPoint

  ( const Vector&,
    const Vector&,
    double,
    const Matrix& ) const

{
  throw jem::IllegalOperationException (
    JEM_FUNC,
    "getLocalPoint is not supported by this shape class -- sorry"
  );

  return false;
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* Shape::getExtByID ( ExtensionID extID ) const
{
  return nullptr;
}


JIVE_END_PACKAGE( geom )
