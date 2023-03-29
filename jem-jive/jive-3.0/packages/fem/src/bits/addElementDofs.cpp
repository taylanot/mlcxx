
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


#include <jive/util/XDofSpace.h>
#include <jive/geom/Shape.h>
#include <jive/fem/utilities.h>
#include <jive/fem/ShapeArray.h>
#include <jive/fem/SimpleBoundary.h>
#include <jive/fem/ElementIterator.h>
#include <jive/fem/BoundaryIterator.h>


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   addElemDofTypes_
//-----------------------------------------------------------------------


void                    addElemDofTypes_

  ( XDofSpace&            dofs,
    IdxVector&            dofTypes,
    idx_t                 maxFuncs,
    const StringVector&   dofNames )

{
  const idx_t  varCount = dofNames.size ();

  String       name;

  idx_t        i, j, k;


  dofTypes.resize ( maxFuncs * varCount );

  k = 0;

  for ( i = 0; i < maxFuncs; i++ )
  {
    for ( j = 0; j < varCount; j++ )
    {
      name = String::format ( "%s[%d]", dofNames[j], i );

      dofTypes[k++] = dofs.addType ( name );
    }
  }
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   addElementDofs (given an element iterator)
//-----------------------------------------------------------------------


IdxVector               addElementDofs

  ( XDofSpace&            dofs,
    ElementIterator&      iter,
    const String&         dofName )

{
  StringVector  dofNames ( 1 );

  dofNames[0] = dofName;

  return addElementDofs ( dofs, iter, dofNames );
}


IdxVector               addElementDofs

  ( XDofSpace&            dofs,
    ElementIterator&      iter,
    const StringVector&   dofNames )

{
  ShapeVector    shapes = iter.getShapes ();

  IdxVector      dofTypes;
  IdxVector      itypes;

  const idx_t    varCount   = dofNames.size ();
  const idx_t    shapeCount = shapes  .size ();

  IdxVector      ielem ( 1 );

  SimpleElement  e;

  idx_t          i, n;


  n = 0;

  for ( i = 0; i < shapeCount; i++ )
  {
    if ( shapes[i] )
    {
      n = jem::max ( n, shapes[i]->shapeFuncCount() );
    }
  }

  addElemDofTypes_ ( dofs, dofTypes, n, dofNames );

  iter.rewind ();

  while ( iter.toNext( e ) )
  {
    if ( e.newShape )
    {
      n = varCount * e.shape->shapeFuncCount ();

      itypes.ref ( dofTypes[slice(BEGIN,n)] );
    }

    ielem[0] = e.ielem;

    dofs.addDofs ( ielem, itypes );
  }

  return dofTypes;
}


//-----------------------------------------------------------------------
//   addElementDofs (given a boundary iterator)
//-----------------------------------------------------------------------


IdxVector               addElementDofs

  ( XDofSpace&            dofs,
    BoundaryIterator&     iter,
    const String&         dofName )

{
  StringVector  dofNames ( 1 );

  dofNames[0] = dofName;

  return addElementDofs ( dofs, iter, dofNames );
}


IdxVector               addElementDofs

  ( XDofSpace&            dofs,
    BoundaryIterator&     iter,
    const StringVector&   dofNames )

{
  IShapeVector    shapes = iter.getShapes   ();

  IdxVector       dofTypes;
  IdxVector       itypes;

  const idx_t     varCount   = dofNames.size ();
  const idx_t     shapeCount = shapes  .size ();

  IdxVector       ielem ( 1 );

  SimpleBoundary  b;

  idx_t           i, n;


  n = 0;

  for ( i = 0; i < shapeCount; i++ )
  {
    if ( shapes[i] )
    {
      n = jem::max ( n, shapes[i]->shapeFuncCount() );
    }
  }

  addElemDofTypes_ ( dofs, dofTypes, n, dofNames );

  iter.rewind ();

  while ( iter.toNext( b ) )
  {
    if ( b.elem.newShape )
    {
      n = varCount * b.elem.shape->shapeFuncCount ();

      itypes.ref ( dofTypes[slice(BEGIN,n)] );
    }

    ielem[0] = b.elem.ielem;

    dofs.addDofs ( ielem, itypes );
  }

  return dofTypes;
}


JIVE_END_PACKAGE( fem )
