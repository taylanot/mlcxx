
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
#include <jem/base/RuntimeException.h>
#include <jem/base/array/utilities.h>
#include <jive/gl/GfxShape.h>


JEM_DEFINE_CLASS( jive::gl::GfxShape );


JIVE_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GfxShape
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


GfxShape::GfxShape () : type ( NONE )
{}


GfxShape::GfxShape ( TypeID tid ) : type ( tid )
{}


GfxShape::~GfxShape ()
{}


//=======================================================================
//   class GfxPoint
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GfxPoint::GfxPoint () : Super ( POINT )
{}


GfxPoint::~GfxPoint ()
{}


//=======================================================================
//   class GfxLine
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GfxLine::GfxLine () : Super ( LINE )
{}


GfxLine::~GfxLine ()
{}


//=======================================================================
//   class GfxPoint
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GfxFace::GfxFace ( const Ref<BShape>& sh ) :

  Super ( FACE ),
  shape ( sh )

{
  JEM_PRECHECK ( shape );
}


GfxFace::~GfxFace ()
{}


//-----------------------------------------------------------------------
//   check
//-----------------------------------------------------------------------


#undef  JIVE_CHECK
#define JIVE_CHECK( expr )                      \
                                                \
  if ( ! (expr) )                               \
                                                \
    checkFailed_ ( #expr, (idx_t) __LINE__ )


void GfxFace::check () const
{
  using jem::min;
  using jem::max;

  JIVE_CHECK ( shape );
  JIVE_CHECK ( edgeCount() >= 0 );

  const idx_t  ne = edgeCount ();

  JIVE_CHECK ( xedges[0] == 0 && xedges[ne] == enodes.size() );

  JIVE_CHECK ( min( xedges ) >= 0 &&
               max( xedges ) <= enodes.size() );

  if ( enodes.size() > 0 )
  {
    JIVE_CHECK ( min( enodes ) >= 0 &&
                 max( enodes ) <  nodeCount() );
  }

  if ( xfuncs.size(1) > 0 )
  {
    JIVE_CHECK ( xfuncs.size(0) == nodeCount() );
  }

  if ( triangles.size(1) > 0 )
  {
    JIVE_CHECK ( triangles.size(0) == 3 );

    JIVE_CHECK ( min( triangles ) >= 0 &&
                 max( triangles ) <  vertexCount() );
  }
}

#undef JIVE_CHECK


//-----------------------------------------------------------------------
//   setXPoint
//-----------------------------------------------------------------------


void GfxFace::setXPoint

  ( double  x,
    double  y )

{
  Vector  xp ( 2 );

  xfuncs.resize  ( nodeCount(), 1 );

  xp[0] = x;
  xp[1] = y;

  shape->evalShapeFunctions ( xfuncs[0], xp );
}


//-----------------------------------------------------------------------
//   setXPoints
//-----------------------------------------------------------------------


void GfxFace::setXPoints ( const Matrix& xp )
{
  const idx_t  nx = xp.size (1);

  xfuncs.resize ( nodeCount(), nx );

  if ( nx > 0 )
  {
    JEM_PRECHECK ( xp.size(0) == 2 );

    for ( idx_t j = 0; j < nx; j++ )
    {
      shape->evalShapeFunctions ( xfuncs[j], xp[j] );
    }
  }
}


//-----------------------------------------------------------------------
//   checkFailed_
//-----------------------------------------------------------------------


void GfxFace::checkFailed_

  ( const char*  expr,
    int          line ) const

{
  const char*  MY_NAME = "jive::gl::GfxFace";

  String       where;

  if ( ! shape )
  {
    where = MY_NAME;
  }
  else
  {
    String  geom = shape->getGeometry ();

    where = String::format ( "%s ( %s%d )",
                             MY_NAME, geom, nodeCount() );
  }

  throw jem::RuntimeException (
    where,
    String::format (
      "invalid graphics shape; runtime check `%s\' failed "
      "in source file `%s\' at line %d",
      expr,
      __FILE__,
      line
    )
  );
}


JIVE_END_PACKAGE( gl )
