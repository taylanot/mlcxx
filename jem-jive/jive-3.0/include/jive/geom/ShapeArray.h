
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

#ifndef JIVE_GEOM_SHAPEARRAY_H
#define JIVE_GEOM_SHAPEARRAY_H

#include <jem/base/Collectable.h>
#include <jem/base/array/Array.h>
#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( geom )


class Shape;
class StdShape;
class StdBoundary;
class InternalShape;
class BoundaryShape;
class InterfaceShape;


//-----------------------------------------------------------------------
//   typedefs
//-----------------------------------------------------------------------


typedef StdShape            SShape;
typedef InternalShape       IShape;
typedef BoundaryShape       BShape;
typedef InterfaceShape      FShape;


typedef jem::Array
  < Ref<Shape> >            ShapeVector;
typedef jem::Array
  < Ref<SShape> >           SShapeVector;
typedef jem::Array
  < Ref<IShape> >           IShapeVector;
typedef jem::Array
  < Ref<BShape> >           BShapeVector;
typedef jem::Array
  < Ref<FShape> >           FShapeVector;
typedef jem::Array
  <Ref<StdBoundary> >       SBShapeVector;

typedef jem::Array
  < Ref<Shape>, 2 >         ShapeMatrix;
typedef jem::Array
  < Ref<SShape>, 2 >        SShapeMatrix;
typedef jem::Array
  < Ref<IShape>, 2 >        IShapeMatrix;
typedef jem::Array
  < Ref<BShape>, 2 >        BShapeMatrix;
typedef jem::Array
  < Ref<FShape>, 2 >        FShapeMatrix;


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


IShapeVector                toIShapes

  ( const ShapeVector&        shapes );

IShapeMatrix                toIShapes

  ( const ShapeMatrix&        shapes );

BShapeVector                toBShapes

  ( const ShapeVector&        shapes );

BShapeMatrix                toBShapes

  ( const ShapeMatrix&        shapes );

FShapeVector                toFShapes

  ( const ShapeVector&        shapes );

FShapeMatrix                toFShapes

  ( const ShapeMatrix&        shapes );


JIVE_END_PACKAGE( geom )

#endif
