
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
#include <jem/base/array/utilities.h>
#include <jive/geom/ShapeTable.h>
#include <jive/geom/BoundaryShape.h>
#include <jive/fem/ElementGroup.h>
#include <jive/gl/ShapeMapper.h>
#include <jive/gl/GfxMesh.h>
#include <jive/gl/GfxMesher2D.h>


JEM_DEFINE_CLASS( jive::gl::GfxMesher2D );


JIVE_BEGIN_PACKAGE( gl )


using jem::iarray;
using jem::newInstance;
using jive::util::Topology;
using jive::geom::ShapeVector;


//=======================================================================
//   class GfxMesher2D::Work_
//=======================================================================


class GfxMesher2D::Work_
{
 public:

                          Work_

    ( const ElementGroup&   egroup,
      const ShapeTable&     shapes,
      const ShapeMapper&    mapper );

  void                    initLines

    ( GfxMesh&              mesh )     const;

  void                    initFaces

    ( GfxMesh&              mesh )     const;

  void                    initVertices

    ( GfxMesh&              mesh )     const;


 private:

  void                    initShapes_

    ( const ShapeTable&     shapes,
      const ShapeMapper&    mapper );


 private:

  ElementSet              elems_;
  IdxVector               ielems_;

  idx_t                   nodeCount_;
  idx_t                   ielemCount_;
  idx_t                   vertexCount_;
  idx_t                   shapeCount_;

  GfxShapeVector          shapes_;
  IdxVector               elemShapeMap_;
  IdxVector               nodeVertexMap_;
  IdxVector               inodes_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


GfxMesher2D::Work_::Work_

  ( const ElementGroup&  egroup,
    const ShapeTable&    stable,
    const ShapeMapper&   mapper ) :

    elems_  ( egroup.getElements() ),
    ielems_ ( egroup.getIndices()  )

{
  nodeCount_     = elems_.getNodes().size ();
  ielemCount_    = ielems_          .size ();

  elemShapeMap_  .resize ( ielemCount_ );
  inodes_        .resize ( elems_.maxElemNodeCount() );
  nodeVertexMap_ .resize ( nodeCount_ );

  elemShapeMap_  = -1;
  shapeCount_    =  0;
  nodeVertexMap_ = -1;
  vertexCount_   =  0;

  for ( idx_t  ie = 0; ie < ielemCount_; ie++ )
  {
    idx_t  n = elems_.getElemNodes ( inodes_, ielems_[ie] );

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  inode = inodes_[i];

      if ( nodeVertexMap_[inode] < 0 )
      {
        nodeVertexMap_[inode] = vertexCount_++;
      }
    }
  }

  initShapes_ ( stable, mapper );
}


//-----------------------------------------------------------------------
//   initLines
//-----------------------------------------------------------------------


void GfxMesher2D::Work_::initLines ( GfxMesh& m ) const
{
  IdxVector  lineOffsets;
  IdxVector  lineVertices;

  GfxLine*   line;

  idx_t      inode;
  idx_t      i, j, k, n;


  k = n = 0;

  for ( idx_t ie = 0; ie < ielemCount_; ie++ )
  {
    i = elemShapeMap_[ie];

    if ( i >= 0 && shapes_[i]->isLine() )
    {
      n += 1;
      k += shapes_[i]->toLine()->inodes.size ();
    }
  }

  m.lineElemMap.resize ( n );
  lineOffsets  .resize ( n + 1 );
  lineVertices .resize ( k );

  j = k = 0;

  for ( idx_t ie = 0; ie < ielemCount_; ie++ )
  {
    i = elemShapeMap_[ie];

    if ( i >= 0 && shapes_[i]->isLine() )
    {
      lineOffsets[j]   = k;
      m.lineElemMap[j] = ie;

      elems_.getElemNodes ( inodes_, ielems_[ie] );

      line = shapes_[i]->toLine ();
      n    = line->inodes.size  ();

      for ( i = 0; i < n; i++, k++ )
      {
        inode           = inodes_[line->inodes[i]];
        lineVertices[k] = nodeVertexMap_[inode];
      }

      j++;
    }
  }

  lineOffsets[j] = k;

  m.lineTopo = Topology ( jem::shape ( j, vertexCount_ ),
                          lineOffsets, lineVertices );
}


//-----------------------------------------------------------------------
//   initFaces
//-----------------------------------------------------------------------


void GfxMesher2D::Work_::initFaces ( GfxMesh& m ) const
{
  IdxVector  remap ( shapeCount_ );

  IdxVector  faceOffsets;
  IdxVector  faceVertices;

  idx_t      i, j, k, n;


  k = 0;

  for ( i = 0; i < shapeCount_; i++ )
  {
    if ( shapes_[i]->isFace() )
    {
      remap[i] =  k++;
    }
    else
    {
      remap[i] = -1;
    }
  }

  m.faceTypes.resize ( k );

  for ( i = 0; i < shapeCount_; i++ )
  {
    j = remap[i];

    if ( j >= 0 )
    {
      m.faceTypes[j] = shapes_[i]->toFace ();
    }
  }

  k = n = 0;

  for ( idx_t ie = 0; ie < ielemCount_; ie++ )
  {
    i = elemShapeMap_[ie];

    if ( i >= 0 && shapes_[i]->isFace() )
    {
      n += 1;
      k += shapes_[i]->toFace()->shape->nodeCount ();
    }
  }

  m.faceElemMap.resize ( n );
  m.faceTypeMap.resize ( n );
  faceOffsets  .resize ( n + 1 );
  faceVertices .resize ( k );

  j = k = 0;

  for ( idx_t ie = 0; ie < ielemCount_; ie++ )
  {
    i = elemShapeMap_[ie];

    if ( i >= 0 && remap[i] >= 0 )
    {
      faceOffsets  [j] = k;
      m.faceElemMap[j] = ie;
      m.faceTypeMap[j] = remap[i];

      n = elems_.getElemNodes ( inodes_, ielems_[ie] );

      for ( i = 0; i < n; i++ )
      {
        faceVertices[k++] = nodeVertexMap_[inodes_[i]];
      }

      j++;
    }
  }

  faceOffsets[j] = k;

  m.faceTopo = Topology ( jem::shape ( j, vertexCount_ ),
                          faceOffsets, faceVertices );
}


//-----------------------------------------------------------------------
//   initVertices
//-----------------------------------------------------------------------


void GfxMesher2D::Work_::initVertices ( GfxMesh& m ) const
{
  m.vertexNodeMap.resize ( vertexCount_ );

  for ( idx_t inode = 0; inode < nodeCount_; inode++ )
  {
    if ( nodeVertexMap_[inode] >= 0 )
    {
      m.vertexNodeMap[nodeVertexMap_[inode]] = inode;
    }
  }
}


//-----------------------------------------------------------------------
//   initShapes_
//-----------------------------------------------------------------------


void GfxMesher2D::Work_::initShapes_

  ( const ShapeTable&   st,
    const ShapeMapper&  mapper )

{
  ShapeVector  svec = st.getShapesOf ( elemShapeMap_, ielems_ );

  shapeCount_    = svec.size ();
  shapes_.resize ( shapeCount_ );

  for ( idx_t i = 0; i < shapeCount_; i++ )
  {
    if ( svec[i] )
    {
      shapes_[i] = mapper.toGfxShape ( *svec[i] );
    }
    else
    {
      shapes_[i] = newInstance<GfxShape> ();
    }
  }
}


//=======================================================================
//   class GfxMesher2D
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


GfxMesher2D::GfxMesher2D ( Ref<ShapeMapper> mapper ) :

  shapeMapper_ ( mapper )

{
  JEM_PRECHECK ( mapper );
}


GfxMesher2D::~GfxMesher2D ()
{}


//-----------------------------------------------------------------------
//   makeGfxMesh
//-----------------------------------------------------------------------


void GfxMesher2D::makeGfxMesh

  ( GfxMesh&             mesh,
    const ElementGroup&  elems,
    const ShapeTable&    shapes )

{
  Work_  work ( elems, shapes, *shapeMapper_ );

  work.initLines    ( mesh );
  work.initFaces    ( mesh );
  work.initVertices ( mesh );
}


JIVE_END_PACKAGE( gl )
