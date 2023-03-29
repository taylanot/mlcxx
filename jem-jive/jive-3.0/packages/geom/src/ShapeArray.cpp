
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


#include <jem/base/Class.h>
#include <jive/geom/InternalShape.h>
#include <jive/geom/BoundaryShape.h>
#include <jive/geom/InterfaceShape.h>
#include <jive/geom/ShapeArray.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::Class;
using jem::Array;


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   castShapes_ ( vector version )
//-----------------------------------------------------------------------


template <class T>

  inline void                 castShapes_

  ( const Array< Ref<T> >&      lhs,
    const Array< Ref<Shape> >&  rhs )

{
  Class*       klass = T::getType ();

  const idx_t  n     = rhs.size ();


  for ( idx_t i = 0; i < n; i++ )
  {
    Shape*  s = rhs[i].get ();

    if ( s && klass->isBaseOf( s->getClass() ) )
    {
      lhs[i] = static_cast<T*> ( s );
    }
  }
}


//-----------------------------------------------------------------------
//   castShapes_ ( matrix version )
//-----------------------------------------------------------------------


template <class T>

  inline void                     castShapes_

  ( const Array< Ref<T>, 2 >&       lhs,
    const Array< Ref<Shape>, 2 >&   rhs )

{
  Class*       klass = T::getType ();

  const idx_t  m     = rhs.size (0);
  const idx_t  n     = rhs.size (1);


  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      Shape*  s = rhs(i,j).get ();

      if ( s && klass->isBaseOf( s->getClass() ) )
      {
        lhs(i,j) = static_cast<T*> ( s );
      }
    }
  }
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   toIShapes
//-----------------------------------------------------------------------


IShapeVector          toIShapes

  ( const ShapeVector&  shapes )

{
  IShapeVector  iv ( shapes.size() );

  castShapes_ ( iv, shapes );

  return iv;
}


IShapeMatrix          toIShapes

  ( const ShapeMatrix&  shapes )

{
  IShapeMatrix  im ( shapes.shape() );

  castShapes_ ( im, shapes );

  return im;
}


//-----------------------------------------------------------------------
//   toBShapes
//-----------------------------------------------------------------------


BShapeVector          toBShapes

  ( const ShapeVector&  shapes )

{
  BShapeVector  bv ( shapes.size() );

  castShapes_ ( bv, shapes );

  return bv;
}


BShapeMatrix          toBShapes

  ( const ShapeMatrix&  shapes )

{
  BShapeMatrix  bm ( shapes.shape() );

  castShapes_ ( bm, shapes );

  return bm;
}


//-----------------------------------------------------------------------
//   toFShapes
//-----------------------------------------------------------------------


FShapeVector          toFShapes

  ( const ShapeVector&  shapes )

{
  FShapeVector  bv ( shapes.size() );

  castShapes_ ( bv, shapes );

  return bv;
}


FShapeMatrix          toFShapes

  ( const ShapeMatrix&  shapes )

{
  FShapeMatrix  bm ( shapes.shape() );

  castShapes_ ( bm, shapes );

  return bm;
}


JIVE_END_PACKAGE( geom )
