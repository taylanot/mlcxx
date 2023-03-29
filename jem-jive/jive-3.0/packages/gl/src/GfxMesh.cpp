
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
#include <jive/gl/GfxMesh.h>


JEM_DEFINE_CLASS( jive::gl::GfxMesh );


JIVE_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GfxMesh
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GfxMesh::GfxMesh ()
{
  xpointCount_ = 0;
}


GfxMesh::~GfxMesh ()
{}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void GfxMesh::clear ()
{
  Topology  nilTopo;

  lineTopo = nilTopo;
  faceTopo = nilTopo;

  lineElemMap  .reshape ( 0 );
  faceElemMap  .reshape ( 0 );
  faceTypeMap  .reshape ( 0 );
  vertexNodeMap.reshape ( 0 );
  faceTypes    .reshape ( 0 );

  xpointCount_ = 0;
}


//-----------------------------------------------------------------------
//   check
//-----------------------------------------------------------------------


#undef  JIVE_CHECK
#define JIVE_CHECK( expr )                      \
                                                \
  if ( ! (expr) )                               \
                                                \
    checkFailed_ ( #expr, (idx_t) __LINE__ )


void GfxMesh::check () const
{
  IdxVector  faceOffsets = faceTopo.getRowOffsets    ();
  IdxVector  faceIndices = faceTopo.getColumnIndices ();

  JIVE_CHECK ( lineTopo.isValid() && faceTopo.isValid() );

  JIVE_CHECK ( lineTopo.size(0) == lineCount   () &&
               lineTopo.size(1) == vertexCount () &&
               faceTopo.size(0) == faceCount   () &&
               faceTopo.size(1) == vertexCount () );

  JIVE_CHECK ( faceElemMap.size() == faceTypeMap.size() );

  idx_t  n = lineCount ();

  for ( idx_t i = 0; i < n; i++ )
  {
    JIVE_CHECK ( lineElemMap[i] >= 0 );
  }

  n = faceTypes.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    JIVE_CHECK ( faceTypes[i] );

    faceTypes[i]->check ();
  }

  n = faceCount ();

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  j = faceTypeMap[i];

    JIVE_CHECK ( j >= 0 && j < faceTypes.size() );

    JIVE_CHECK ( faceElemMap[i] >= 0 );

    idx_t  k = faceOffsets[i + 1] - faceOffsets[i];

    JIVE_CHECK ( faceTypes[j]->nodeCount() == k );
  }
}

#undef JIVE_CHECK


//-----------------------------------------------------------------------
//   addXPoints
//-----------------------------------------------------------------------


idx_t GfxMesh::addXPoints ()
{
  if ( xpointCount_ == 0 )
  {
    const idx_t  nl = lineCount ();
    const idx_t  nf = faceCount ();

    for ( idx_t i = 0; i < nf; i++ )
    {
      xpointCount_ += faceTypes[faceTypeMap[i]]->xpointCount ();
    }

    if ( xpointCount_ > 0 )
    {
      idx_t  nv = vertexCount ();

      vertexNodeMap.reshape ( nv + xpointCount_ );

      vertexNodeMap[slice(nv,END)] = -1;

      nv += xpointCount_;

      lineTopo = Topology ( jem::shape ( nl, nv ),
                            lineTopo.getRowOffsets    (),
                            lineTopo.getColumnIndices () );

      faceTopo = Topology ( jem::shape ( nf, nv ),
                            faceTopo.getRowOffsets    (),
                            faceTopo.getColumnIndices () );
    }
  }

  return xpointCount_;
}


//-----------------------------------------------------------------------
//   checkFailed_
//-----------------------------------------------------------------------


void GfxMesh::checkFailed_

  ( const char*  expr,
    int          line ) const

{
  throw jem::RuntimeException (
    CLASS_NAME,
    String::format (
      "invalid graphics mesh; runtime check `%s\' failed "
      "in source file `%s\' at line %d",
      expr,
      __FILE__,
      line
    )
  );
}


JIVE_END_PACKAGE( gl )
