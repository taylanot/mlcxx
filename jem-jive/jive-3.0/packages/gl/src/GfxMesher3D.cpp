
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
#include <jive/geom/ShapeArray.h>
#include <jive/geom/InternalShape.h>
#include <jive/geom/InterfaceShape.h>
#include <jive/fem/ElementGroup.h>
#include <jive/gl/ShapeMapper.h>
#include <jive/gl/GfxMesh.h>
#include <jive/gl/GfxMesher3D.h>


JEM_DEFINE_CLASS( jive::gl::GfxMesher3D );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jive::util::Topology;
using jive::util::TopoVector;
using jive::geom::Shape;
using jive::geom::IShape;
using jive::geom::BShape;
using jive::geom::FShape;
using jive::geom::ShapeVector;


//=======================================================================
//   class GfxMesher3D::Work
//=======================================================================


class GfxMesher3D::Work_
{
 public:

                          Work_

    ( const ElementGroup&   egroup,
      const ShapeTable&     stable,
      const ShapeMapper&    mapper );

  bool                    tryInitFast

    ( GfxMesh&              mesh )           const;

  void                    initLines

    ( GfxMesh&              mesh )           const;

  void                    initFaces

    ( GfxMesh&              mesh )           const;

  inline void             initVertices

    ( GfxMesh&              mesh )           const;


 private:

  void                    initElemShapes_

    ( const ShapeTable&     st );

  void                    initSegments_   ();

  void                    addSegments_

  ( idx_t&                  iseg,
    idx_t&                  ilast,
    idx_t                   ielem,
    const IdxVector&        inodes,
    const Topology&         elemTopo );

  void                    initSegShapes_

    ( const ShapeMapper&    mapper );

  void                    killElemShapes_ ();
  void                    pruneSegments_  ();
  void                    packVertices_   ();


 private:

  ElementSet              elems_;
  IdxVector               ielems_;

  idx_t                   segCount_;
  idx_t                   vtxCount_;

  IdxVector               segOffsets_;
  IdxVector               vtxIndices_;

  IdxVector               ifaceMap_;
  IdxVector               segElemMap_;
  IdxVector               vtxNodeMap_;
  IdxVector               segShapeMap_;
  IdxVector               elemShapeMap_;

  GfxShapeVector          segShapes_;
  IntVector               segTypes_;
  ShapeVector             elemShapes_;
  IntVector               elemTypes_;
  TopoVector              elemTopos_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


GfxMesher3D::Work_::Work_

  ( const ElementGroup&  egroup,
    const ShapeTable&    stable,
    const ShapeMapper&   mapper ) :

    elems_  ( egroup.getElements() ),
    ielems_ ( egroup.getIndices () )

{
  segCount_ = 0;
  vtxCount_ = 0;

  initElemShapes_ ( stable );
  initSegments_   ();
  initSegShapes_  ( mapper );
  killElemShapes_ ();
  packVertices_   ();
  pruneSegments_  ();
  packVertices_   ();
}


//-----------------------------------------------------------------------
//   tryInitFast
//-----------------------------------------------------------------------


bool GfxMesher3D::Work_::tryInitFast ( GfxMesh& m ) const
{
  const idx_t  shapeCount = segShapes_.size ();

  IdxVector    idummy;

  idx_t        n = 0;

  for ( idx_t i = 0; i < shapeCount; i++ )
  {
    if ( segShapes_[i]->isFace() )
    {
      n++;
    }
  }

  if ( n != shapeCount )
  {
    return false;
  }

  idummy.resize ( 1 );

  idummy[0] = 0;

  m.lineTopo = Topology ( jem::shape ( 0, vtxCount_ ),
                          idummy, IdxVector() );

  m.faceTopo = Topology ( jem::shape ( segCount_, vtxCount_ ),
                          segOffsets_, vtxIndices_ );

  m.lineElemMap  .ref ( IdxVector()  );
  m.faceElemMap  .ref ( segElemMap_  );
  m.faceTypeMap  .ref ( segShapeMap_ );
  m.vertexNodeMap.ref ( vtxNodeMap_  );

  m.faceTypes.resize  ( shapeCount );

  for ( idx_t i = 0; i < shapeCount; i++ )
  {
    m.faceTypes[i] = segShapes_[i]->toFace ();
  }

  return true;
}


//-----------------------------------------------------------------------
//   initLines
//-----------------------------------------------------------------------


void GfxMesher3D::Work_::initLines ( GfxMesh& m ) const
{
  IdxVector     lineOffsets;
  IdxVector     lineVertices;

  const idx_t*  ivx;

  GfxLine*      line;

  idx_t         i, j, k, n;


  k = n = 0;

  for ( idx_t iseg = 0; iseg < segCount_; iseg++ )
  {
    i = segShapeMap_[iseg];

    if ( segShapes_[i]->isLine() )
    {
      n += 1;
      k += segShapes_[i]->toLine()->inodes.size ();
    }
  }

  m.lineElemMap.resize ( n );
  lineOffsets  .resize ( n + 1 );
  lineVertices .resize ( k );

  j = k = 0;

  for ( idx_t iseg = 0; iseg < segCount_; iseg++ )
  {
    i = segShapeMap_[iseg];

    if ( segShapes_[i]->isLine() )
    {
      lineOffsets[j]   = k;
      m.lineElemMap[j] = segElemMap_[iseg];

      line = segShapes_[i]->toLine ();
      n    = line->inodes.size     ();
      ivx  = vtxIndices_ .addr     ( segOffsets_[iseg] );

      for ( i = 0; i < n; i++ )
      {
        lineVertices[k++] = ivx[line->inodes[i]];
      }

      j++;
    }
  }

  lineOffsets[j] = k;

  m.lineTopo = Topology ( jem::shape ( j, vtxCount_ ),
                          lineOffsets, lineVertices );
}


//-----------------------------------------------------------------------
//   initFaces
//-----------------------------------------------------------------------


void GfxMesher3D::Work_::initFaces ( GfxMesh& m ) const
{
  const idx_t   shapeCount = segShapes_.size ();

  IdxVector     remap ( shapeCount );

  IdxVector     faceOffsets;
  IdxVector     faceVertices;

  const idx_t*  ivx;

  idx_t         i, j, k, n;


  k = 0;

  for ( i = 0; i < shapeCount; i++ )
  {
    if ( segShapes_[i]->isFace() )
    {
      remap[i] = k++;
    }
    else
    {
      remap[i] = -1;
    }
  }

  m.faceTypes.resize ( k );

  for ( i = 0; i < shapeCount; i++ )
  {
    j = remap[i];

    if ( j >= 0 )
    {
      m.faceTypes[j] = segShapes_[i]->toFace ();
    }
  }

  k = n = 0;

  for ( idx_t iseg = 0; iseg < segCount_; iseg++ )
  {
    i = segShapeMap_[iseg];

    if ( segShapes_[i]->isFace() )
    {
      n += 1;
      k += (segOffsets_[iseg + 1] - segOffsets_[iseg]);
    }
  }

  m.faceElemMap.resize ( n );
  m.faceTypeMap.resize ( n );
  faceOffsets  .resize ( n + 1 );
  faceVertices .resize ( k );

  j = k = 0;

  for ( idx_t iseg = 0; iseg < segCount_; iseg++ )
  {
    i = segShapeMap_[iseg];

    if ( remap[i] < 0 )
    {
      continue;
    }

    faceOffsets[j]   = k;
    m.faceElemMap[j] = segElemMap_[iseg];
    m.faceTypeMap[j] = remap[i];

    i   = segOffsets_[iseg];
    n   = segOffsets_[iseg + 1] - i;
    ivx = vtxIndices_.addr ( i );

    for ( i = 0; i < n; i++ )
    {
      faceVertices[k++] = ivx[i];
    }

    j++;
  }

  faceOffsets[j] = k;

  m.faceTopo = Topology ( jem::shape ( j, vtxCount_ ),
                          faceOffsets, faceVertices );
}


//-----------------------------------------------------------------------
//   initVertices
//-----------------------------------------------------------------------


inline void GfxMesher3D::Work_::initVertices ( GfxMesh& m ) const
{
  m.vertexNodeMap.ref ( vtxNodeMap_ );
}


//-----------------------------------------------------------------------
//   initElemShapes_
//-----------------------------------------------------------------------


void GfxMesher3D::Work_::initElemShapes_ ( const ShapeTable& st )
{
  using jem::staticCast;

  IShape*  ishape;

  idx_t    i, n;


  elemShapeMap_.resize ( ielems_.size() );
  elemShapes_  .ref    ( st.getShapesOf( elemShapeMap_, ielems_ ) );

  n = elemShapes_.size ();

  elemTypes_.resize ( n );
  elemTopos_.resize ( n );

  elemTypes_ = 0;

  for ( i = 0; i < n; i++ )
  {
    if ( ! elemShapes_[i] )
    {
      continue;
    }

    elemTypes_[i] = ShapeTable::getTypeFlags ( *elemShapes_[i] );

    if ( elemTypes_[i] & ShapeTable::INTERNAL_TYPE )
    {
      ishape = staticCast<IShape*> ( elemShapes_[i] );

      elemTopos_[i] = ishape->getBoundaryTopology ();
    }
  }

  n = ielems_.size ();

  for ( idx_t ie = 0; ie < n; ie++ )
  {
    i = elemShapeMap_[ie];

    if ( i >= 0 && elemTypes_[i] == 0 )
    {
      elemShapeMap_[ie] = -1;
    }
  }
}


//-----------------------------------------------------------------------
//   initSegments_
//-----------------------------------------------------------------------


void GfxMesher3D::Work_::initSegments_ ()
{
  const idx_t  ielemCount = ielems_.size ();

  IdxVector    inodes     ( elems_.maxElemNodeCount() );

  idx_t        ifaceCount;
  idx_t        ielem;
  idx_t        iseg;

  idx_t        i, j, k, n;


  ifaceCount = k = n = 0;

  for ( idx_t ie = 0; ie < ielemCount; ie++ )
  {
    i = elemShapeMap_[ie];

    if ( i < 0 )
    {
      continue;
    }

    if      ( elemTypes_[i] & ShapeTable::INTERNAL_TYPE )
    {
      k += elemTopos_[i].size (0);
      n += elemTopos_[i].nonZeroCount ();
    }
    else if ( elemTypes_[i] & ShapeTable::INTERFACE_TYPE )
    {
      k += 2;
      n += elemShapes_[i]->nodeCount ();

      ifaceCount++;
    }
    else
    {
      k += 1;
      n += elemShapes_[i]->nodeCount ();
    }
  }

  segCount_ = k;

  segElemMap_.resize ( segCount_ );
  segOffsets_.resize ( segCount_ + 1 );
  vtxIndices_.resize ( n );
  segElemMap_.resize ( segCount_ );

  if ( ifaceCount > 0 )
  {
    ifaceMap_.resize ( segCount_ );

    ifaceMap_ = -1;
  }

  iseg = 0;
  k    = 0;

  for ( idx_t ie = 0; ie < ielemCount; ie++ )
  {
    ielem = ielems_[ie];
    i     = elemShapeMap_[ie];

    if ( i < 0 )
    {
      continue;
    }

    n = elems_.getElemNodes ( inodes, ielem );

    if      ( elemTypes_[i] & ShapeTable::INTERNAL_TYPE )
    {
      addSegments_ ( iseg, k, ie, inodes, elemTopos_[i] );
    }
    else if ( elemTypes_[i] & ShapeTable::INTERFACE_TYPE )
    {
      segOffsets_[iseg + 0] = k;
      segOffsets_[iseg + 1] = k + n / 2;
      segElemMap_[iseg + 0] = ie;
      segElemMap_[iseg + 1] = ie;
      ifaceMap_  [iseg + 0] = iseg + 1;
      ifaceMap_  [iseg + 1] = iseg;

      iseg += 2;

      for ( j = 0; j < n; j++ )
      {
        vtxIndices_[k++] = inodes[j];
      }
    }
    else
    {
      segOffsets_[iseg] = k;
      segElemMap_[iseg] = ie;

      iseg++;

      for ( j = 0; j < n; j++ )
      {
        vtxIndices_[k++] = inodes[j];
      }
    }
  }

  segOffsets_[iseg] = k;

  vtxCount_ = 0;

  if ( vtxIndices_.size() > 0 )
  {
    vtxCount_ = 1 + jem::max ( vtxIndices_ );
  }

  vtxNodeMap_.resize ( vtxCount_ );

  vtxNodeMap_ = jem::iarray ( vtxCount_ );
}


//-----------------------------------------------------------------------
//   addSegments_
//-----------------------------------------------------------------------


void GfxMesher3D::Work_::addSegments_

  ( idx_t&            iseg,
    idx_t&            ilast,
    idx_t             ie,
    const IdxVector&  inodes,
    const Topology&   elemTopo )

{
  IdxVector    bndOffsets  = elemTopo.getRowOffsets    ();
  IdxVector    nodeIndices = elemTopo.getColumnIndices ();

  const idx_t  bndCount    = elemTopo.size (0);


  for ( idx_t ibnd = 0; ibnd < bndCount; ibnd++ )
  {
    segOffsets_[iseg] = ilast;
    segElemMap_[iseg] = ie;

    idx_t  n = bndOffsets[ibnd + 1];

    for ( idx_t i = bndOffsets[ibnd]; i < n; i++ )
    {
      vtxIndices_[ilast++] = inodes[nodeIndices[i]];
    }

    iseg++;
  }
}


//-----------------------------------------------------------------------
//   initSegShapes_
//-----------------------------------------------------------------------


void GfxMesher3D::Work_::initSegShapes_

  ( const ShapeMapper&  mapper )

{
  const idx_t  ielemCount = ielems_    .size ();
  const idx_t  shapeCount = elemShapes_.size ();

  IdxVector    remap ( shapeCount );

  Shape*       shape;
  IShape*      ishape;
  BShape*      bshape;
  FShape*      fshape;

  idx_t        iseg;
  idx_t        i, j, k, n;


  k = 0;

  for ( i = 0; i < shapeCount; i++ )
  {
    remap[i] = k;

    if      ( elemTypes_[i] & ShapeTable::INTERNAL_TYPE )
    {
      k += elemTopos_[i].size (0);
    }
    else if ( elemTypes_[i] & ShapeTable::INTERFACE_TYPE )
    {
      k += 2;
    }
    else if ( elemTypes_[i] != 0 )
    {
      k += 1;
    }
  }

  segShapes_.resize ( k );
  segTypes_ .resize ( k );

  k = 0;

  for ( i = 0; i < shapeCount; i++ )
  {
    shape = elemShapes_[i].get ();

    if      ( elemTypes_[i] & ShapeTable::INTERNAL_TYPE )
    {
      ishape = static_cast<IShape*> ( shape );
      n      = elemTopos_[i].size   (0);

      for ( j = 0; j < n; j++, k++ )
      {
        bshape        = ishape->getBoundaryShape ( j );
        segShapes_[k] = mapper .toGfxShape ( *bshape );
        segTypes_ [k] = elemTypes_[i];
      }
    }
    else if ( elemTypes_[i] & ShapeTable::INTERFACE_TYPE )
    {
      fshape = static_cast<FShape*> ( shape );
      bshape = fshape->getBShape    ();

      segShapes_[k + 0] = mapper.toGfxShape ( *bshape );
      segShapes_[k + 1] = segShapes_[k];
      segTypes_ [k + 0] = elemTypes_[i];
      segTypes_ [k + 1] = elemTypes_[i];

      k += 2;
    }
    else if ( shape )
    {
      segShapes_[k] = mapper.toGfxShape ( *shape );
      segTypes_ [k] = elemTypes_[i];

      k++;
    }
  }

  segShapeMap_.resize ( segCount_ );

  iseg = 0;

  for ( idx_t ie = 0; ie < ielemCount; ie++ )
  {
    i = elemShapeMap_[ie];

    if ( i < 0 )
    {
      continue;
    }

    if      ( elemTypes_[i] & ShapeTable::INTERNAL_TYPE )
    {
      n = elemTopos_[i].size (0);

      for ( j = 0; j < n; j++ )
      {
        segShapeMap_[iseg++] = remap[i] + j;
      }
    }
    else if ( elemTypes_[i] & ShapeTable::INTERFACE_TYPE )
    {
      segShapeMap_[iseg++] = remap[i] + 0;
      segShapeMap_[iseg++] = remap[i] + 1;
    }
    else
    {
      segShapeMap_[iseg++] = remap[i];
    }
  }
}


//-----------------------------------------------------------------------
//   killElemShapes_
//-----------------------------------------------------------------------


void GfxMesher3D::Work_::killElemShapes_ ()
{
  elemShapeMap_.ref ( IdxVector()   );
  elemShapes_  .ref ( ShapeVector() );
  elemTypes_   .ref ( IntVector()   );
  elemTopos_   .ref ( TopoVector()  );
}


//-----------------------------------------------------------------------
//   pruneSegments_
//-----------------------------------------------------------------------


void GfxMesher3D::Work_::pruneSegments_ ()
{
  using jem::numeric::findSimilarRows;

  BoolVector  keep ( segCount_ );
  IdxVector   rmap ( segCount_ );

  idx_t       iseg;
  idx_t       jseg;

  idx_t       i, j, n;


  // Find coinciding segments

  {
    Topology  segTopo ( jem::shape ( segCount_, vtxCount_ ),
                        segOffsets_,
                        vtxIndices_ );

    findSimilarRows ( rmap, segTopo );
  }

  keep = true;

  for ( iseg = 0; iseg < segCount_; iseg++ )
  {
    if      ( rmap[iseg] < 0 )
    {
      i = segShapeMap_[iseg];

      if ( (segTypes_[i] & ShapeTable::INTERFACE_TYPE) &&
           (rmap[ifaceMap_[iseg]] >= 0) )
      {
        keep[iseg] = false;
      }
    }
    else if ( rmap[iseg] > iseg )
    {
      keep[iseg] = false;
    }
    else
    {
      jseg = rmap        [iseg];
      i    = segShapeMap_[iseg];
      j    = segShapeMap_[jseg];

      if      ( segTypes_[i] & ShapeTable::INTERNAL_TYPE )
      {
        if ( segTypes_[j] & ShapeTable::INTERFACE_TYPE )
        {
          segElemMap_[iseg] = segElemMap_[jseg];
        }
        else
        {
          keep[iseg] = false;
        }
      }
      else if ( segTypes_[i] & ShapeTable::INTERFACE_TYPE )
      {
        if ( ! (segTypes_[j] & ShapeTable::INTERFACE_TYPE) )
        {
          segElemMap_[jseg] = segElemMap_[iseg];
        }

        keep[iseg] = 0;
      }
      else
      {
        if ( segTypes_[j] & ShapeTable::INTERFACE_TYPE )
        {
          segElemMap_[iseg] = segElemMap_[jseg];
        }
      }
    }
  }

  jseg = 0;
  j    = 0;

  for ( iseg = 0; iseg < segCount_; iseg++ )
  {
    if ( ! keep[iseg] )
    {
      continue;
    }

    i = segOffsets_[iseg];
    n = segOffsets_[iseg + 1];

    segOffsets_ [jseg] = j;
    segElemMap_ [jseg] = segElemMap_ [iseg];
    segShapeMap_[jseg] = segShapeMap_[iseg];

    for ( ; i < n; i++ )
    {
      vtxIndices_[j++] = vtxIndices_[i];
    }

    jseg++;
  }

  segOffsets_[jseg] = j;

  segCount_ = jseg;

  ifaceMap_   .ref     ( IdxVector() );
  segOffsets_ .reshape ( segCount_ + 1 );
  segElemMap_ .reshape ( segCount_ );
  segShapeMap_.reshape ( segCount_ );
  vtxIndices_ .reshape ( j );
}


//-----------------------------------------------------------------------
//   packVertices_
//-----------------------------------------------------------------------


void GfxMesher3D::Work_::packVertices_ ()
{
  IdxVector  iperm ( vtxCount_ );
  IdxVector  newMap;

  idx_t      ivx;
  idx_t      jvx;
  idx_t      n;


  iperm = -1_idx;
  jvx   = 0;
  n     = vtxIndices_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    ivx = vtxIndices_[i];

    if ( iperm[ivx] < 0 )
    {
      iperm[ivx] = jvx++;
    }

    vtxIndices_[i] = iperm[ivx];
  }

  newMap.resize ( jvx );

  n         = vtxCount_;
  vtxCount_ = jvx;

  for ( ivx = 0; ivx < n; ivx++ )
  {
    jvx = iperm[ivx];

    if ( jvx >= 0 )
    {
      newMap[jvx] = vtxNodeMap_[ivx];
    }
  }

  newMap.swap ( vtxNodeMap_ );
}


//=======================================================================
//   class GfxMesher3D
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


GfxMesher3D::GfxMesher3D ( Ref<ShapeMapper> mapper ) :

  shapeMapper_ ( mapper )

{
  JEM_PRECHECK ( mapper );
}


GfxMesher3D::~GfxMesher3D ()
{}


//-----------------------------------------------------------------------
//   makeGfxMesh
//-----------------------------------------------------------------------


void GfxMesher3D::makeGfxMesh

  ( GfxMesh&             mesh,
    const ElementGroup&  elems,
    const ShapeTable&    shapes )

{
  Work_  work ( elems, shapes, *shapeMapper_ );


  if ( work.tryInitFast( mesh ) )
  {
    return;
  }

  work.initLines    ( mesh );
  work.initFaces    ( mesh );
  work.initVertices ( mesh );
}


JIVE_END_PACKAGE( gl )
