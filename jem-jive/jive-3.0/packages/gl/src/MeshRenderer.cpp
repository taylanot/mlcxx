
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
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/gl/byte.h>
#include <jem/gl/render.h>
#include <jem/gl/Box.h>
#include <jem/gl/Menu.h>
#include <jem/gl/Color.h>
#include <jem/gl/Actions.h>
#include <jem/gl/Commands.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/LineSet.h>
#include <jem/gl/VertexSet.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/TriangleSet.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/matmul.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemMap.h>
#include <jive/fem/ElementGroup.h>
#include <jive/geom/ShapeTable.h>
#include <jive/geom/BoundaryShape.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/GfxMesh.h>
#include <jive/gl/GfxMesher.h>
#include <jive/gl/GfxCommands.h>
#include <jive/gl/MeshRenderer.h>


JEM_DEFINE_CLASS( jive::gl::MeshRenderer );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::IllegalInputException;
using jem::gl::Box;
using jem::gl::Menu;
using jem::gl::LineSet;
using jem::gl::TriangleSet;
using jive::util::joinNames;
using jive::util::Topology;


//=======================================================================
//   class MeshRenderer::AppData_
//=======================================================================


class MeshRenderer::AppData_ : public Collectable
{
 public:

  typedef AppData_          Self;

  inline                    AppData_

    ( const ElementGroup&     group,
      const Ref<ShapeTable>&  shapes,
      const Ref<GfxMesher>&   mesher );


 public:

  ElementGroup              egroup;
  ElementSet                elems;
  NodeSet                   nodes;
  Ref<ShapeTable>           shapes;
  Ref<GfxMesher>            mesher;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline MeshRenderer::AppData_::AppData_

  ( const ElementGroup&     g,
    const Ref<ShapeTable>&  s,
    const Ref<GfxMesher>&   m ) :

    egroup  ( g ),
    elems   ( g.getElements () ),
    nodes   ( elems.getNodes() ),
    shapes  ( s ),
    mesher  ( m )

{}


//=======================================================================
//   class MeshRenderer::OwnData_
//=======================================================================


class MeshRenderer::OwnData_ : public Collectable
{
 public:

  typedef OwnData_          Self;

  static const char*        POINT_SIZE_NAME;
  static const char*        LINE_WIDTH_NAME;
  static const char*        SHADE_MODEL_NAME;
  static const char*        COLOR_NAME;
  static const char*        MATERIAL_NAME;
  static const char*        SPECULAR_NAME;
  static const char*        SHININESS_NAME;
  static const char*        EDGE_COLOR_NAME;
  static const char*        NODE_COLOR_NAME;
  static const char*        SHOW_EDGES_NAME;
  static const char*        SHOW_LINES_NAME;
  static const char*        SHOW_FACES_NAME;

  static const int          NEW_MESH    = 1 << 0;
  static const int          NEW_COORDS  = 1 << 1;
  static const int          NEW_NORMALS = 1 << 2;

  static const int          SHOW_EDGES  = 1 << 0;
  static const int          SHOW_FACES  = 1 << 1;
  static const int          SHOW_LINES  = 1 << 2;
  static const int          SMOOTH_MODE = 1 << 3;


                            OwnData_

    ( const AppData_&         appdat );

  inline void               redraw

    ( RenderContext&          rtx );

  inline void               redrawFast

    ( RenderContext&          rtx,
      double                  lod );

  void                      initMesh

    ( const String&           myName,
      const AppData_&         appdat );

  void                      initElemIDs

    ( const AppData_&         appdat );

  void                      initVertices

    ( const AppData_&         appdat );

  void                      initTriangles

    ( const String&           myName );

  void                      initLines

    ( const String&           myName );

  void                      initEdges

    ( const String&           myName );

  void                      setNodeColors

    ( const GLfloatVector&    colors,
      const Ref<ColorMap>&    cmap );

  void                      setElemColors

    ( const GLubyteMatrix&    colors );

  void                      setNodeCoords

    ( const Matrix&           coords );

  void                      setNodeCoords

    ( idx_t                   idim,
      const Vector&           coords );

  void                      addToNodeCoords

    ( const Matrix&           delta,
      const AppData_&         appdat );

  void                      addToNodeCoords

    ( idx_t                   idim,
      const Vector&           delta,
      const AppData_&         appdat );

  idx_t                     mapTriangToElem

    ( idx_t                   itr )            const;

  void                      updateVertices

    ( const AppData_&         appdat );

  void                      updateNormals   ();

  void                      updateXPoints

    ( const GLfloatVector&    data )           const;

  void                      updateXPoints3

    ( const GLfloatMatrix&    data )           const;

  void                      meshChanged     ();
  void                      geomChanged     ();

  void                      changeOptions

    ( int                     newOptions );

  void                      copyOptions

    ( const Properties&       props );

  void                      getOptions

    ( const Properties&       props )          const;


 public:

  IdxVector                 elemIDs;

  Ref<GfxMesh>              mesh;
  Ref<VertexSet>            vertices;
  Ref<TriangleSet>          triangles;
  Ref<LineSet>              lines;
  Ref<LineSet>              edges;

  Color                     color;
  Color                     specular;
  Color                     edgeColor;
  Color                     nodeColor;

  bool                      trackGeom;
  int                       dirtyBits;
  int                       options;

  GLfloat                   lineWidth;
  GLfloat                   pointSize;
  GLfloat                   shininess;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MeshRenderer::OwnData_::POINT_SIZE_NAME  = "pointSize";
const char*  MeshRenderer::OwnData_::LINE_WIDTH_NAME  = "lineWidth";
const char*  MeshRenderer::OwnData_::SHADE_MODEL_NAME = "shadeModel";
const char*  MeshRenderer::OwnData_::COLOR_NAME       = "color";
const char*  MeshRenderer::OwnData_::MATERIAL_NAME    = "material";
const char*  MeshRenderer::OwnData_::SPECULAR_NAME    = "specular";
const char*  MeshRenderer::OwnData_::SHININESS_NAME   = "shininess";
const char*  MeshRenderer::OwnData_::EDGE_COLOR_NAME  = "edgeColor";
const char*  MeshRenderer::OwnData_::NODE_COLOR_NAME  = "nodeColor";
const char*  MeshRenderer::OwnData_::SHOW_EDGES_NAME  = "showEdges";
const char*  MeshRenderer::OwnData_::SHOW_LINES_NAME  = "showLines";
const char*  MeshRenderer::OwnData_::SHOW_FACES_NAME  = "showFaces";


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


MeshRenderer::OwnData_::OwnData_ ( const AppData_& appdat )
{
  using jem::util::connect;

  options    = SHOW_EDGES | SHOW_LINES | SHOW_FACES;
  trackGeom  = true;
  dirtyBits  = 0;
  lineWidth  = 1.0F;
  pointSize  = 1.0F;

  mesh       = newInstance<GfxMesh>   ();
  vertices   = newInstance<VertexSet> ();

  color      = Color::gray  ();
  edgeColor  = Color::black ();
  nodeColor  = edgeColor;

  if ( appdat.egroup.size() > 100000_idx )
  {
    options &= ~SHOW_EDGES;
  }
  else
  {
    options |=  SHOW_EDGES;
  }

  connect ( appdat.egroup.newSizeEvent (),
            this, & Self::meshChanged );

  connect ( appdat.egroup.newOrderEvent (),
            this, & Self::meshChanged );

  connect ( appdat.nodes.newOrderEvent (),
            this, & Self::meshChanged );

  connect ( appdat.nodes.newRankEvent (),
            this, & Self::meshChanged );

  connect ( appdat.nodes.newGeomEvent (),
            this, & Self::geomChanged );
}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


inline void MeshRenderer::OwnData_::redraw ( RenderContext& rtx )
{
  if ( lineWidth > 0.0 )
  {
    if ( lines && ! lines->hidden )
    {
      lines->redraw ( rtx );
    }

    if ( edges && ! edges->hidden )
    {
      GLContext&  gtx  = rtx.getGLContext ();
      Color       col0 = gtx.getColor     ();

      gtx.setColor  ( edgeColor );
      edges->redraw ( rtx );
      gtx.setColor  ( col0 );
    }
  }

  if ( triangles && ! triangles->hidden )
  {
    if ( (options & SMOOTH_MODE) && (dirtyBits & NEW_NORMALS) )
    {
      updateNormals ();
    }

    triangles->redraw ( rtx );
  }
}


//-----------------------------------------------------------------------
//   redrawFast
//-----------------------------------------------------------------------


inline void MeshRenderer::OwnData_::redrawFast

  ( RenderContext&  rtx,
    double          lod )

{
  if ( (lineWidth > 0.0) && (lod > 0.2) )
  {
    if ( lines && ! lines->hidden )
    {
      lines->redrawFast ( rtx, lod );
    }

    if ( edges && ! edges->hidden )
    {
      GLContext&  gtx  = rtx.getGLContext ();
      Color       col0 = gtx.getColor     ();

      gtx.setColor      ( edgeColor );
      edges->redrawFast ( rtx, lod );
      gtx.setColor      ( col0 );
    }
  }

  if ( triangles && ! triangles->hidden )
  {
    if ( (options & SMOOTH_MODE) && (dirtyBits & NEW_NORMALS) )
    {
      updateNormals ();
    }

    triangles->redrawFast ( rtx, lod );
  }
}


//-----------------------------------------------------------------------
//   initMesh
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::initMesh

  ( const String&    myName,
    const AppData_&  appdat )

{
  using jem::io::Writer;

  Writer&  out = jem::System::info ( myName );

  idx_t    triCount  = 0;
  idx_t    edgeCount = 0;


  print ( out, myName, " : building graphics mesh ...\n" );

  triangles = nullptr;
  edges     = nullptr;
  lines     = nullptr;

  mesh    ->clear   ();
  vertices->reshape ( 3, 0 );

  appdat.mesher->makeGfxMesh

    ( *mesh, appdat.egroup, *appdat.shapes );

  mesh->check      ();
  mesh->addXPoints ();

  initElemIDs  ( appdat );
  initVertices ( appdat );

  if ( mesh->faceCount() > 0 )
  {
    initTriangles ( myName );
    initEdges     ( myName );

    triCount  = triangles->size ();
    edgeCount = edges    ->size ();
  }

  if ( mesh->lineCount() )
  {
    initLines ( myName );
  }

  print ( out, myName, " : ready; graphics mesh contains:\n" );

  print ( out, "\n  vertices  ...... ", vertices ->size() );
  print ( out, "\n  faces     ...... ", mesh->faceCount() );
  print ( out, "\n  triangles ...... ", triCount          );
  print ( out, "\n  lines     ...... ", mesh->lineCount() );
  print ( out, "\n  edges     ...... ", edgeCount         );

  print ( out, "\n\n" );

  appdat.egroup.resetEvents ();
  appdat.nodes .resetEvents ();

  trackGeom = true;
  dirtyBits = NEW_NORMALS;
}


//-----------------------------------------------------------------------
//   initElemIDs
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::initElemIDs

  ( const AppData_& appdat )

{
  IdxVector  ielems = appdat.egroup.getIndices  ();

  elemIDs.resize ( ielems.size() );

  appdat.elems.getData()->

    getItemMap()->getItemIDs ( elemIDs, ielems );
}


//-----------------------------------------------------------------------
//   initVertices
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::initVertices

  ( const AppData_&  appdat )

{
  GLfloatMatrix  coords ( 3, mesh->vertexCount() );

  coords = 0.0F;

  vertices->setCoords ( coords );

  updateVertices ( appdat );
}


//-----------------------------------------------------------------------
//   initTriangles
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::initTriangles

  ( const String&  myName )

{
  const idx_t  faceCount   = mesh->faceCount ();

  IdxVector    faceOffsets = mesh->faceTopo.getRowOffsets    ();
  IdxVector    faceIndices = mesh->faceTopo.getColumnIndices ();

  IdxMatrix    tri;

  GfxFace*     face;

  idx_t        itr, ntr;
  idx_t        ixp, nxp;


  ntr = 0;

  for ( idx_t iface = 0; iface < faceCount; iface++ )
  {
    face = mesh->faceTypes[mesh->faceTypeMap[iface]].get ();
    ntr += face->triangCount ();
  }

  tri.resize ( 3, ntr );

  itr = 0;
  ixp = mesh->firstXPoint ();

  for ( idx_t iface = 0; iface < faceCount; iface++ )
  {
    idx_t  k = faceOffsets[iface];

    face = mesh->faceTypes[mesh->faceTypeMap[iface]].get ();
    ntr  = face->triangCount ();
    nxp  = face->xpointCount ();

    if ( nxp == 0 )
    {
      for ( idx_t j = 0; j < ntr; j++, itr++ )
      {
        tri(0,itr) = faceIndices[k + face->triangles(0,j)];
        tri(1,itr) = faceIndices[k + face->triangles(1,j)];
        tri(2,itr) = faceIndices[k + face->triangles(2,j)];
      }
    }
    else
    {
      idx_t  nnd = face->nodeCount ();

      for ( idx_t j = 0; j < ntr; j++, itr++ )
      {
        for ( idx_t i = 0; i < 3; i++ )
        {
          idx_t  ivx = face->triangles(i,j);

          if ( ivx < nnd )
          {
            tri(i,itr) = faceIndices[k + ivx];
          }
          else
          {
            tri(i,itr) = ixp + ivx - nnd;
          }
        }
      }
    }

    ixp += nxp;
  }

  JEM_ASSERT ( ixp == mesh->vertexCount() );

  triangles = newInstance<TriangleSet> (
    joinNames ( myName, "faces" ),
    vertices
  );

  triangles->setTopology ( tri );

  triangles->hidden = ! (options & SHOW_FACES);
}


//-----------------------------------------------------------------------
//   initLines
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::initLines

  ( const String&  myName )

{
  lines = newInstance<LineSet> ( joinNames ( myName, "lines" ),
                                 vertices );

  lines->setTopology ( mesh->lineTopo.getRowOffsets    (),
                       mesh->lineTopo.getColumnIndices () );

  lines->hidden = ! (options & SHOW_LINES);
}


//-----------------------------------------------------------------------
//   initEdges
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::initEdges

  ( const String&  myName )

{
  using jem::numeric::findSimilarRows;

  const idx_t  faceCount   = mesh->faceCount ();

  IdxVector    faceOffsets = mesh->faceTopo.getRowOffsets    ();
  IdxVector    faceIndices = mesh->faceTopo.getColumnIndices ();

  IdxVector    edgeOffsets;
  IdxVector    edgeIndices;
  IdxVector    edgeBuddies;

  GfxFace*     face;

  idx_t        edgeCount;
  idx_t        iedge;
  idx_t        i, j, k, n;


  k = n = 0;

  for ( idx_t iface = 0; iface < faceCount; iface++ )
  {
    face = mesh->faceTypes[mesh->faceTypeMap[iface]].get ();
    n   += face->edgeCount   ();
    k   += face->enodes.size ();
  }

  edgeOffsets.resize ( n + 1 );
  edgeBuddies.resize ( n );
  edgeIndices.resize ( k );

  edgeCount      = n;
  edgeOffsets[0] = 0;

  iedge = j = 0;

  for ( idx_t iface = 0; iface < faceCount; iface++ )
  {
    face = mesh->faceTypes[mesh->faceTypeMap[iface]].get ();
    n    = face->edgeCount ();
    k    = edgeOffsets[iedge];

    for ( i = 0; i < n; i++, iedge++ )
    {
      edgeOffsets[iedge + 1] = k + face->xedges[i + 1];
    }

    n = face->enodes.size ();
    k = faceOffsets[iface];

    for ( i = 0; i < n; i++, j++ )
    {
      edgeIndices[j] = faceIndices[k + face->enodes[i]];
    }
  }

  n = mesh->vertexCount ();

  findSimilarRows (
    edgeBuddies,
    Topology ( jem::shape ( edgeCount, n ),
               edgeOffsets, edgeIndices )
  );

  j = k = 0;

  for ( iedge = 0; iedge < edgeCount; iedge++ )
  {
    if ( edgeBuddies[iedge] >= iedge )
    {
      continue;
    }

    i = edgeOffsets[iedge];
    n = edgeOffsets[iedge + 1];

    edgeOffsets[j + 1] = edgeOffsets[j] + (n - i);

    for ( ; i < n; i++, k++ )
    {
      edgeIndices[k] = edgeIndices[i];
    }

    j++;
  }

  edgeOffsets.reshape ( j + 1 );
  edgeIndices.reshape ( k );

  edges = newInstance<LineSet> (
    joinNames ( myName, "edges" ),
    vertices
  );

  edges->setTopology ( edgeOffsets, edgeIndices );

  edges->hidden = ! (options & SHOW_EDGES);
}


//-----------------------------------------------------------------------
//   setNodeColors
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::setNodeColors

  ( const GLfloatVector&  nodeColors,
    const Ref<ColorMap>&  colorMap )

{
  const idx_t    vertexCount  = mesh->vertexCount ();
  const idx_t    firstXPoint  = mesh->firstXPoint ();

  GLfloatVector  vertexColors ( vertexCount );


  for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
  {
    idx_t  inode = mesh->vertexNodeMap[ivx];

    vertexColors[ivx] = nodeColors[inode];
  }

  if ( mesh->xpointCount() > 0 )
  {
    updateXPoints ( vertexColors );
  }

  if ( triangles )
  {
    triangles->setVertexColors ( vertexColors, colorMap );
  }

  if ( lines )
  {
    lines    ->setVertexColors ( vertexColors, colorMap );
  }
}


//-----------------------------------------------------------------------
//   setElemColors
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::setElemColors

  ( const GLubyteMatrix&  elemColors )

{
  JEM_PRECHECK ( mesh );

  const idx_t  colorCount = elemColors.size (0);

  if ( triangles )
  {
    const idx_t    faceCount   = mesh->faceCount ();
    const idx_t    triangCount = triangles->size ();

    GLubyteMatrix  triangColors;

    GfxFace*       face;

    idx_t          itr;
    idx_t          ie;
    idx_t          n;


    if ( triangles->getColorMode() == TriangleSet::TRIANG_COLORS )
    {
      triangColors.ref    ( triangles->getTriangColors() );
    }
    else
    {
      triangColors.resize ( colorCount, triangCount );
    }

    itr = 0;

    for ( idx_t iface = 0; iface < faceCount; iface++ )
    {
      face = mesh->faceTypes[mesh->faceTypeMap[iface]].get ();
      ie   = mesh->faceElemMap[iface];
      n    = itr + face->triangCount ();

      for ( ; itr < n; itr++ )
      {
        for ( idx_t i = 0; i < colorCount; i++ )
        {
          triangColors(i,itr) = elemColors(i,ie);
        }
      }
    }

    triangles->setTriangColors ( triangColors );
  }

  if ( lines )
  {
    const idx_t    lineCount = lines->size ();

    GLubyteMatrix  lineColors;


    if ( lines->getColorMode() == LineSet::LINE_COLORS )
    {
      lineColors.ref    ( lines->getLineColors() );
    }
    else
    {
      lineColors.resize ( colorCount, lineCount );
    }

    for ( idx_t iline = 0; iline < lineCount; iline++ )
    {
      idx_t  ie = mesh->lineElemMap[iline];

      for ( idx_t i = 0; i < colorCount; i++ )
      {
        lineColors(i,iline) = elemColors(i,ie);
      }
    }

    lines->setLineColors ( lineColors );
  }
}


//-----------------------------------------------------------------------
//   setNodeCoords
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::setNodeCoords

  ( const Matrix&  nodeCoords )

{
  JEM_PRECHECK ( mesh );

  const idx_t    rank         = nodeCoords.size     (0);
  const idx_t    firstXPoint  = mesh->firstXPoint   ();

  GLfloatMatrix  vertexCoords = vertices->getCoords ();


  if      ( rank == 1_idx )
  {
    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      idx_t  inode = mesh->vertexNodeMap[ivx];

      vertexCoords(0,ivx) = (GLfloat) nodeCoords(0,inode);
    }
  }
  else if ( rank == 2_idx )
  {
    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      idx_t  inode = mesh->vertexNodeMap[ivx];

      vertexCoords(0,ivx) = (GLfloat) nodeCoords(0,inode);
      vertexCoords(1,ivx) = (GLfloat) nodeCoords(1,inode);
    }
  }
  else if ( rank >= 3_idx )
  {
    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      idx_t  inode = mesh->vertexNodeMap[ivx];

      vertexCoords(0,ivx) = (GLfloat) nodeCoords(0,inode);
      vertexCoords(1,ivx) = (GLfloat) nodeCoords(1,inode);
      vertexCoords(2,ivx) = (GLfloat) nodeCoords(2,inode);
    }
  }

  if ( mesh->xpointCount() > 0 )
  {
    updateXPoints3 ( vertexCoords );
  }

  vertices->newCoords ();

  dirtyBits &= ~NEW_COORDS;
  dirtyBits |=  NEW_NORMALS;
}


void MeshRenderer::OwnData_::setNodeCoords

  ( idx_t          idim,
    const Vector&  nodeCoords )

{
  JEM_PRECHECK ( mesh );

  const idx_t    firstXPoint  = mesh->firstXPoint   ();

  GLfloatMatrix  vertexCoords = vertices->getCoords ();


  for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
  {
    idx_t  inode = mesh->vertexNodeMap[ivx];

    vertexCoords(idim,ivx) = (GLfloat) nodeCoords[inode];
  }

  if ( mesh->xpointCount() > 0 )
  {
    updateXPoints ( vertexCoords(idim,ALL) );
  }

  vertices->newCoords ();

  dirtyBits &= ~NEW_COORDS;
  dirtyBits |=  NEW_NORMALS;
}


//-----------------------------------------------------------------------
//   addToNodeCoords
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::addToNodeCoords

  ( const Matrix&    delta,
    const AppData_&  appdat )

{
  JEM_PRECHECK ( mesh );

  const idx_t    rank         = appdat.nodes.rank   ();
  const idx_t    firstXPoint  = mesh->firstXPoint   ();

  GLfloatMatrix  vertexCoords = vertices->getCoords ();

  Matrix         nodeCoords   ( rank, firstXPoint );

  idx_t          n1, n2;


  appdat.nodes.getSomeCoords

    ( nodeCoords, mesh->vertexNodeMap[slice(BEGIN,firstXPoint)] );

  n1 = jem::min ( rank,  delta.size(0) );
  n2 = jem::min ( 3_idx, delta.size(0) );

  if      ( n1 == 1_idx )
  {
    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      idx_t  inode = mesh->vertexNodeMap[ivx];

      vertexCoords(0,ivx) = (GLfloat)

        (nodeCoords(0,ivx) + delta(0,inode));
    }
  }
  else if ( n1 == 2_idx )
  {
    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      idx_t  inode = mesh->vertexNodeMap[ivx];

      vertexCoords(0,ivx) = (GLfloat)
        (nodeCoords(0,ivx) + delta(0,inode));
      vertexCoords(1,ivx) = (GLfloat)
        (nodeCoords(1,ivx) + delta(1,inode));
    }
  }
  else if ( n1 >= 3_idx )
  {
    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      idx_t  inode = mesh->vertexNodeMap[ivx];

      vertexCoords(0,ivx) = (GLfloat)
        (nodeCoords(0,ivx) + delta(0,inode));
      vertexCoords(1,ivx) = (GLfloat)
        (nodeCoords(1,ivx) + delta(1,inode));
      vertexCoords(2,ivx) = (GLfloat)
        (nodeCoords(2,ivx) + delta(2,inode));
    }
  }

  for ( idx_t i = n1; i < n2; i++ )
  {
    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      idx_t  inode = mesh->vertexNodeMap[ivx];

      vertexCoords(i,ivx) = (GLfloat) delta(i,inode);
    }
  }

  if ( mesh->xpointCount() > 0 )
  {
    updateXPoints3 ( vertexCoords );
  }

  vertices->newCoords ();

  dirtyBits &= ~NEW_COORDS;
  dirtyBits |=  NEW_NORMALS;
}


void MeshRenderer::OwnData_::addToNodeCoords

  ( idx_t            idim,
    const Vector&    delta,
    const AppData_&  appdat )

{
  JEM_PRECHECK ( mesh );

  const idx_t    rank         = appdat.nodes.rank   ();
  const idx_t    firstXPoint  = mesh->firstXPoint   ();

  GLfloatMatrix  vertexCoords = vertices->getCoords ();


  if ( idim < rank )
  {
    Matrix  nodeCoords ( rank, firstXPoint );

    appdat.nodes.getSomeCoords

      ( nodeCoords, mesh->vertexNodeMap[slice(BEGIN,firstXPoint)] );

    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      idx_t  inode = mesh->vertexNodeMap[ivx];

      vertexCoords(idim,ivx) = (GLfloat)

        (nodeCoords(idim,ivx) + delta[inode]);
    }
  }
  else
  {
    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      idx_t  inode = mesh->vertexNodeMap[ivx];

      vertexCoords(idim,ivx) = (GLfloat) delta[inode];
    }
  }

  if ( mesh->xpointCount() > 0 )
  {
    updateXPoints ( vertexCoords(idim,ALL) );
  }

  vertices->newCoords ();

  dirtyBits &= ~NEW_COORDS;
  dirtyBits |=  NEW_NORMALS;
}


//-----------------------------------------------------------------------
//   mapTriangToElem
//-----------------------------------------------------------------------


idx_t MeshRenderer::OwnData_::mapTriangToElem ( idx_t itr ) const
{
  JEM_PRECHECK ( mesh );

  const idx_t  faceCount = mesh->faceCount ();

  GfxFace*     face;

  idx_t        jtr, ntr;


  // Take a shortcut if possible

  if ( faceCount == triangles->size() )
  {
    return mesh->faceElemMap[itr];
  }

  jtr = 0;

  for ( idx_t iface = 0; iface < faceCount; iface++ )
  {
    face = mesh->faceTypes[mesh->faceTypeMap[iface]].get ();
    ntr  = face->triangCount ();

    if ( itr >= jtr && itr < (jtr + ntr) )
    {
      return mesh->faceElemMap[iface];
    }

    jtr += ntr;
  }

  throw jem::Error (
    JEM_FUNC,
    String::format (
      "failed to map triangle %d to its corresponding element", itr
    )
  );

  return -1;
}


//-----------------------------------------------------------------------
//   updateVertices
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::updateVertices

  ( const AppData_&  appdat )

{
  JEM_PRECHECK ( mesh );

  const idx_t    rank         = appdat.nodes.rank   ();
  const idx_t    firstXPoint  = mesh->firstXPoint   ();

  GLfloatMatrix  vertexCoords = vertices->getCoords ();

  Matrix         nodeCoords   ( rank, firstXPoint );


  appdat.nodes.getSomeCoords

    ( nodeCoords, mesh->vertexNodeMap[slice(BEGIN,firstXPoint)] );

  if      ( rank == 0_idx )
  {
    vertexCoords = 0.0F;
  }
  else if ( rank == 1_idx )
  {
    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      vertexCoords(0,ivx) = (GLfloat) nodeCoords(0,ivx);
      vertexCoords(1,ivx) = 0.0F;
      vertexCoords(2,ivx) = 0.0F;
    }
  }
  else if ( rank == 2_idx )
  {
    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      vertexCoords(0,ivx) = (GLfloat) nodeCoords(0,ivx);
      vertexCoords(1,ivx) = (GLfloat) nodeCoords(1,ivx);
      vertexCoords(2,ivx) = 0.0F;
    }
  }
  else if ( rank >= 3_idx )
  {
    for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
    {
      vertexCoords(0,ivx) = (GLfloat) nodeCoords(0,ivx);
      vertexCoords(1,ivx) = (GLfloat) nodeCoords(1,ivx);
      vertexCoords(2,ivx) = (GLfloat) nodeCoords(2,ivx);
    }
  }

  if ( mesh->xpointCount() > 0 )
  {
    updateXPoints3 ( vertexCoords );
  }

  vertices->newCoords ();

  dirtyBits &= ~NEW_COORDS;
  dirtyBits |=  NEW_NORMALS;
}


//-----------------------------------------------------------------------
//   updateNormals
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::updateNormals ()
{
  using jem::System;
  using jem::gl::toByte;
  using jem::numeric::matmul;

  JEM_PRECHECK ( triangles );

  // Set auto normals if all faces are triangles

  if ( mesh->faceCount() == triangles->size() )
  {
    triangles->setAutoNormals ();

    return;
  }

  const idx_t    faceCount    = mesh->faceCount ();

  IdxVector      faceOffsets  = mesh->faceTopo.getRowOffsets    ();
  IdxVector      faceIndices  = mesh->faceTopo.getColumnIndices ();

  GLfloatMatrix  vertexCoords = vertices->getCoords ();

  Matrix         faceCoords;
  Matrix         faceNormals;
  GLbyteMatrix   triangNormals;

  GfxFace*       face;

  idx_t          ierr;
  idx_t          itr, ntr;
  idx_t          nnd, nxp;


  if ( triangles->getNormalMode() != TriangleSet::TRIANG_NORMALS )
  {
    triangNormals.resize ( 3, 3 * triangles->size() );
  }
  else
  {
    triangNormals.ref    ( triangles->getNormals() );
  }

  ierr = -1;
  itr  =  0;

  for ( idx_t iface = 0; iface < faceCount; iface++ )
  {
    idx_t  k = faceOffsets[iface];

    face = mesh->faceTypes[mesh->faceTypeMap[iface]].get ();

    nnd  = faceOffsets[iface + 1] - k;
    ntr  = face->triangCount ();
    nxp  = face->xpointCount ();

    if ( faceCoords.size(1) != nnd )
    {
      faceCoords .resize ( 3, nnd );
      faceNormals.resize ( 3, nnd + nxp );
    }

    // Get the node coordinates.

    for ( idx_t j = 0; j < nnd; j++ )
    {
      idx_t  ivx = faceIndices[k + j];

      faceCoords(0,j) = vertexCoords(0,ivx);
      faceCoords(1,j) = vertexCoords(1,ivx);
      faceCoords(2,j) = vertexCoords(2,ivx);
    }

    // Compute the node normals.

    try
    {
      face->shape->getVertexNormals
        ( faceNormals[slice(BEGIN,nnd)], faceCoords );
    }
    catch ( const jem::ArithmeticException& )
    {
      ierr = mesh->faceElemMap[iface];

      faceNormals        = 0.0;
      faceNormals(2,ALL) = 1.0;
    }

    // Interpolate the node normals at the extra points.

    if ( nxp > 0 )
    {
      matmul ( faceNormals[slice(nnd,END)],
               faceNormals[slice(BEGIN,nnd)], face->xfuncs );
    }

    for ( idx_t j = 0; j < ntr; j++ )
    {
      for ( idx_t i = 0; i < 3; i++, itr++ )
      {
        idx_t  ivx = face->triangles(i,j);

        triangNormals(0,itr) = toByte ( faceNormals(0,ivx) );
        triangNormals(1,itr) = toByte ( faceNormals(1,ivx) );
        triangNormals(2,itr) = toByte ( faceNormals(2,ivx) );
      }
    }
  }

  triangles->setTriangNormals ( triangNormals );

  dirtyBits &= ~NEW_NORMALS;

  if ( ierr >= 0 )
  {
    print ( System::warn(), "element ", elemIDs[ierr],
            " has zero normal vectors\n\n" );
  }
}


//-----------------------------------------------------------------------
//   updateXPoints
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::updateXPoints

  ( const GLfloatVector&  data ) const

{
  const idx_t  faceCount   = mesh->faceCount ();

  IdxVector    faceOffsets = mesh->faceTopo.getRowOffsets    ();
  IdxVector    faceIndices = mesh->faceTopo.getColumnIndices ();

  GfxFace*     face;

  idx_t        ixp, nxp;
  idx_t        nnd;


  ixp = mesh->firstXPoint ();

  for ( idx_t iface = 0; iface < faceCount; iface++ )
  {
    idx_t  k = faceOffsets[iface];

    face = mesh->faceTypes[mesh->faceTypeMap[iface]].get ();

    nnd  = faceOffsets[iface + 1] - k;
    nxp  = face->xpointCount ();

    for ( idx_t j = 0; j < nxp; j++, ixp++ )
    {
      double  x = 0.0;

      for ( idx_t i = 0; i < nnd; i++ )
      {
        idx_t  ivx = faceIndices[k + i];

        x += face->xfuncs(i,j) * data[ivx];
      }

      data[ixp] = (GLfloat) x;
    }
  }
}


//-----------------------------------------------------------------------
//   updateXPoints3
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::updateXPoints3

  ( const GLfloatMatrix&  data ) const

{
  JEM_PRECHECK ( data.size(0) == 3 );

  const idx_t  faceCount   = mesh->faceCount ();

  IdxVector    faceOffsets = mesh->faceTopo.getRowOffsets    ();
  IdxVector    faceIndices = mesh->faceTopo.getColumnIndices ();

  GfxFace*     face;

  idx_t        ixp, nxp;
  idx_t        nnd;


  ixp = mesh->firstXPoint ();

  for ( idx_t iface = 0; iface < faceCount; iface++ )
  {
    idx_t  k = faceOffsets[iface];

    face = mesh->faceTypes[mesh->faceTypeMap[iface]].get ();

    nnd  = faceOffsets[iface + 1] - k;
    nxp  = face->xpointCount ();

    for ( idx_t j = 0; j < nxp; j++, ixp++ )
    {
      double  x = 0.0;
      double  y = 0.0;
      double  z = 0.0;

      for ( idx_t i = 0; i < nnd; i++ )
      {
        idx_t   ivx = faceIndices[k + i];
        double  xf  = face->xfuncs(i,j);

        x += xf * data(0,ivx);
        y += xf * data(1,ivx);
        z += xf * data(2,ivx);
      }

      data(0,ixp) = (GLfloat) x;
      data(1,ixp) = (GLfloat) y;
      data(2,ixp) = (GLfloat) z;
    }
  }
}


//-----------------------------------------------------------------------
//   meshChanged
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::meshChanged ()
{
  dirtyBits = NEW_MESH;
}


//-----------------------------------------------------------------------
//   geomChanged
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::geomChanged ()
{
  if ( trackGeom )
  {
    dirtyBits |= NEW_COORDS;
  }
}


//-----------------------------------------------------------------------
//   changeOptions
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::changeOptions ( int newOptions )
{
  int  changes = options ^ newOptions;

  if ( changes & SMOOTH_MODE )
  {
    if ( newOptions & SMOOTH_MODE )
    {
      dirtyBits |=  NEW_NORMALS;
    }
    else
    {
      dirtyBits &= ~NEW_NORMALS;

      if ( triangles )
      {
        triangles->setAutoNormals ();
      }
    }
  }

  if ( (changes & SHOW_EDGES) && edges )
  {
    edges->hidden = ! (newOptions & SHOW_EDGES);
  }

  if ( (changes & SHOW_LINES) && lines )
  {
    lines->hidden = ! (newOptions & SHOW_LINES);
  }

  if ( (changes & SHOW_FACES) && triangles )
  {
    triangles->hidden = ! (newOptions & SHOW_FACES);
  }

  options = newOptions;
}


//-----------------------------------------------------------------------
//   copyOptions
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::copyOptions ( const Properties& props )
{
  const int    OPT_BITS [3] = { SHOW_EDGES,
                                SHOW_LINES,
                                SHOW_FACES };

  const char*  OPT_NAMES[3] = { SHOW_EDGES_NAME,
                                SHOW_LINES_NAME,
                                SHOW_FACES_NAME };

  Properties   matProps     = props.findProps ( MATERIAL_NAME );
  int          newOptions   = options;

  String       s;
  double       d;


  if ( props.find( d, POINT_SIZE_NAME, 0.0, 100.0 ) )
  {
    pointSize = (GLfloat) d;
  }

  if ( props.find( d, LINE_WIDTH_NAME, 0.0, 100.0 ) )
  {
    lineWidth = (GLfloat) d;
  }

  if ( props.find( s, SHADE_MODEL_NAME ) )
  {
    if      ( s.equalsIgnoreCase( "SMOOTH" ) )
    {
      newOptions |=  SMOOTH_MODE;
    }
    else if ( s.equalsIgnoreCase( "FLAT" ) )
    {
      newOptions &= ~SMOOTH_MODE;
    }
    else
    {
      props.propertyError (
        SHADE_MODEL_NAME,
        String::format (
          "invalid shade model: `%s\'; should be either "
          "`Flat\' or `Smooth\'",
          s
        )
      );
    }
  }

  for ( int i = 0; i < 3; i++ )
  {
    bool  yesno;

    if ( props.find( yesno, OPT_NAMES[i] ) )
    {
      if ( yesno )
      {
        newOptions |=  OPT_BITS[i];
      }
      else
      {
        newOptions &= ~OPT_BITS[i];
      }
    }
  }

  color    .configure ( COLOR_NAME,      props );
  edgeColor.configure ( EDGE_COLOR_NAME, props );
  nodeColor.configure ( NODE_COLOR_NAME, props );
  specular .configure ( SPECULAR_NAME,   matProps );

  if ( matProps.find( d, SHININESS_NAME, 0.0, 128.0 ) )
  {
    shininess = (GLfloat) d;
  }

  changeOptions ( newOptions );
}


//-----------------------------------------------------------------------
//   copyOptions
//-----------------------------------------------------------------------


void MeshRenderer::OwnData_::getOptions

  ( const Properties& props ) const

{
  const int    OPT_BITS [3] = { SHOW_EDGES,
                                SHOW_LINES,
                                SHOW_FACES };

  const char*  OPT_NAMES[3] = { SHOW_EDGES_NAME,
                                SHOW_LINES_NAME,
                                SHOW_FACES_NAME };

  Properties   matProps     = props.makeProps ( MATERIAL_NAME );
  String       s;


  if ( options & SMOOTH_MODE )
  {
    s = "Smooth";
  }
  else
  {
    s = "Flat";
  }

  for ( int i = 0; i < 3; i++ )
  {
    if ( options & OPT_BITS[i] )
    {
      bool  yesno = ((options & OPT_BITS[i]) != 0);

      props.set ( OPT_NAMES[i], yesno );
    }
  }

  props   .set ( POINT_SIZE_NAME,  (double) pointSize );
  props   .set ( LINE_WIDTH_NAME,  (double) lineWidth );
  props   .set ( SHADE_MODEL_NAME, s );
  matProps.set ( SHININESS_NAME,   (double) shininess );

  color    .getConfig ( COLOR_NAME,      props );
  edgeColor.getConfig ( EDGE_COLOR_NAME, props );
  nodeColor.getConfig ( NODE_COLOR_NAME, props );
  specular .getConfig ( SPECULAR_NAME,   matProps );
}


//=======================================================================
//   class MeshRenderer::DrawScope_
//=======================================================================


class MeshRenderer::DrawScope_
{
 public:

  inline                    DrawScope_

    ( RenderContext&          rtx,
      const OwnData_&         data );

  inline                   ~DrawScope_  ();


 private:

  GLContext&                gtx_;
  GLenum                    shadeModel_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline MeshRenderer::DrawScope_::DrawScope_

  ( RenderContext&   rtx,
    const OwnData_&  data ) :

    gtx_ ( rtx.getGLContext() )

{
  using jem::gl::GLFuncs1;
  using jem::gl::GLVersion;

  shadeModel_ = GL_SMOOTH;

  gtx_.pushAttribs  ( GLContext::COLOR      |
                      GLContext::SPECULAR   |
                      GLContext::POINT_SIZE |
                      GLContext::LINE_WIDTH );

  gtx_.setColor     ( data.color );
  gtx_.setSpecular  ( data.specular );
  gtx_.setShininess ( data.shininess );
  gtx_.setPointSize ( data.pointSize );
  gtx_.setLineWidth ( data.lineWidth );

  if ( ! gtx_.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLFuncs1*  funcs = gtx_.findFunctions<GLFuncs1> ();

    if ( funcs )
    {
      GLint  model;

      funcs->glGetParameter ( GL_SHADE_MODEL, model );

      shadeModel_ = (GLenum) model;

      if ( data.options & OwnData_::SMOOTH_MODE )
      {
        funcs->glShadeModel ( GL_SMOOTH );
      }
      else
      {
        funcs->glShadeModel ( GL_FLAT );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


inline MeshRenderer::DrawScope_::~DrawScope_ ()
{
  using jem::gl::GLFuncs1;
  using jem::gl::GLVersion;

  gtx_.popAttribs ();

  if ( ! gtx_.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLFuncs1*  funcs = gtx_.findFunctions<GLFuncs1> ();

    if ( funcs )
    {
      funcs->glShadeModel ( shadeModel_ );
    }
  }
}


//=======================================================================
//   class MeshRenderer
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MeshRenderer::MeshRenderer

  ( const String&           name,
    const AppContext&       apx,
    const ElementGroup&     group,
    const Ref<ShapeTable>&  shapes,
    const Ref<GfxMesher>&   mesher ) :

    Super ( name )

{
  JEM_PRECHECK ( group && shapes && mesher );

  checkShapes_ ( *shapes, group.getElements() );

  appData_ = newInstance<AppData_> ( group, shapes, mesher );
  ownData_ = newInstance<OwnData_> ( *appData_[apx]  );

  ownData_->initMesh ( myName_, *appData_[apx] );
}


MeshRenderer::~MeshRenderer ()
{}


//-----------------------------------------------------------------------
//   getBBox
//-----------------------------------------------------------------------


void MeshRenderer::getBBox ( Box& box ) const
{
  if ( ownData_->vertices )
  {
    ownData_->vertices->getBBox ( box );
  }
  else
  {
    box.makeEmpty ();
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MeshRenderer::configure ( const Properties& props )
{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    ownData_->copyOptions ( myProps );

    Super::configure ( props );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MeshRenderer::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  ownData_->getOptions ( myProps );

  Super::getConfig ( props );
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool MeshRenderer::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  using jem::gl::GLActions;
  using jem::gl::GLActionParams;

  if ( ! hidden && ownData_->mesh )
  {
    if ( action == GLActions::REDRAW_FAST )
    {
      double  lod = 1.0;

      params.find ( lod, GLActionParams::DETAIL_LEVEL );

      redrawFast ( rtx, lod );

      return true;
    }

    if ( action == GLActions::REDRAW )
    {
      redraw ( rtx );

      return true;
    }

    if ( action == GLActions::DRAW_CDATA )
    {
      drawCData ( rtx );

      return true;
    }
  }

  if ( action == GLActions::EXE_COMMAND )
  {
    String  cmd;

    params.get ( cmd, GLActionParams::COMMAND );

    return exeCommand ( rtx, cmd, params );
  }

  return false;
}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


void MeshRenderer::redraw ( RenderContext& rtx )
{
  if ( ownData_->mesh )
  {
    DrawScope_  scope ( rtx, *ownData_ );

    ownData_->redraw ( rtx );
  }
}


//-----------------------------------------------------------------------
//   redrawFast
//-----------------------------------------------------------------------


void MeshRenderer::redrawFast

  ( RenderContext&  rtx,
    double          lod )

{
  using jem::gl::renderBoxEdges;

  OwnData_&  d = * ownData_;

  if ( d.mesh )
  {
    DrawScope_  scope ( rtx, d );

    if ( lod < 0.1 )
    {
      Box  box;

      d.vertices->getBBox ( box );

      renderBoxEdges ( rtx.getGLContext(), box );
    }

    d.redrawFast ( rtx, lod );
  }
}


//-----------------------------------------------------------------------
//   drawCData
//-----------------------------------------------------------------------


void MeshRenderer::drawCData ( RenderContext& rtx )
{
  using jem::gl::AttribScope;

  OwnData_&  d = * ownData_;

  if ( d.triangles && ! d.triangles->hidden )
  {
    GLContext&   gtx     = rtx.getGLContext ();
    AttribScope  attribs ( gtx, GLContext::LIGHTING );

    gtx.setLighting     ( false );
    d.triangles->redraw ( rtx );
  }
}


//-----------------------------------------------------------------------
//   exeCommand
//-----------------------------------------------------------------------


bool MeshRenderer::exeCommand

  ( RenderContext&     rtx,
    const String&      cmd,
    const Properties&  params )

{
  OwnData_&  d       = * ownData_;
  int        options = d.options;

  if ( cmd == GfxCommands::SHOW_MESH )
  {
    hidden = false;
    return   true;
  }

  if ( cmd == GfxCommands::HIDE_MESH )
  {
    hidden = true;
    return   true;
  }

  if ( cmd == GfxCommands::SHOW_EDGES )
  {
    options |= OwnData_::SHOW_EDGES;

    d.changeOptions ( options );

    return true;
  }

  if ( cmd == GfxCommands::HIDE_EDGES )
  {
    options &= ~OwnData_::SHOW_EDGES;

    d.changeOptions ( options );

    return true;
  }

  if ( cmd == GfxCommands::SHOW_LINES )
  {
    options |= OwnData_::SHOW_LINES;

    d.changeOptions ( options );

    return true;
  }

  if ( cmd == GfxCommands::HIDE_LINES )
  {
    options &= ~OwnData_::SHOW_LINES;

    d.changeOptions ( options );

    return true;
  }

  if ( cmd == GfxCommands::SHOW_FACES )
  {
    options |= OwnData_::SHOW_FACES;

    d.changeOptions ( options );

    return true;
  }

  if ( cmd == GfxCommands::HIDE_FACES )
  {
    options &= ~OwnData_::SHOW_FACES;

    d.changeOptions ( options );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   listOptions
//-----------------------------------------------------------------------


void MeshRenderer::listOptions ( Menu& menu ) const
{
  OwnData_&  d = * ownData_;

  menu.addOption ( GfxCommands::HIDE_MESH,
                   GfxCommands::SHOW_MESH, hidden );

  if ( d.edges )
  {
    menu.addOption ( GfxCommands::HIDE_EDGES,
                     GfxCommands::SHOW_EDGES, d.edges->hidden );
  }

  if ( d.lines )
  {
    menu.addOption ( GfxCommands::HIDE_LINES,
                     GfxCommands::SHOW_LINES, d.lines->hidden );
  }

  if ( d.triangles )
  {
    menu.addOption ( GfxCommands::HIDE_FACES,
                     GfxCommands::SHOW_FACES, d.triangles->hidden );
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void MeshRenderer::update ( const AppContext& apx )
{
  OwnData_&  d = * ownData_;

  updateMesh_ ( apx );

  if ( d.edges && (d.options & OwnData_::SHOW_EDGES) )
  {
    d.initEdges ( myName_ );
  }

  if ( d.trackGeom && (d.dirtyBits & OwnData_::NEW_COORDS) )
  {
    d.updateVertices ( *appData_[apx] );
  }
}


//-----------------------------------------------------------------------
//   clearColors
//-----------------------------------------------------------------------


void MeshRenderer::clearColors ()
{
  if ( ownData_->triangles )
  {
    ownData_->triangles->setNoColors ();
  }
}


//-----------------------------------------------------------------------
//   setNodeColors
//-----------------------------------------------------------------------


void MeshRenderer::setNodeColors

  ( const GLfloatVector&  colors,
    const Ref<ColorMap>&  colorMap,
    const AppContext&     apx )

{
  JEM_PRECHECK ( colorMap );

  OwnData_&  d = * ownData_;
  const idx_t  n = appData_[apx]->nodes.size ();


  if ( colors.size() != n )
  {
    util::sizeError ( JEM_FUNC,
                      "node color vector", colors.size(), n );
  }

  updateMesh_ ( apx );

  if ( d.lines || d.triangles )
  {
    d.setNodeColors ( colors, colorMap );
  }
}


//-----------------------------------------------------------------------
//   setElemColors
//-----------------------------------------------------------------------


void MeshRenderer::setElemColors

  ( const GLubyteMatrix&  colors,
    const AppContext&     apx )

{
  JEM_PRECHECK ( colors.size(0) >= 3 &&
                 colors.size(0) <= 4 );

  OwnData_&    d = * ownData_;
  const idx_t  n = appData_[apx]->egroup.size ();


  if ( colors.size(1) != n )
  {
    util::shapeError ( JEM_FUNC, "element color matrix",
                       colors.shape(),
                       jem::shape( colors.size(0), n ) );
  }

  updateMesh_ ( apx );

  if ( d.lines || d.triangles )
  {
    d.setElemColors ( colors );
  }
}


//-----------------------------------------------------------------------
//   trackMeshGeom
//-----------------------------------------------------------------------


void MeshRenderer::trackMeshGeom ()
{
  OwnData_&  d = * ownData_;

  if ( ! d.trackGeom )
  {
    d.trackGeom  = true;
    d.dirtyBits |= OwnData_::NEW_COORDS;
  }
}


//-----------------------------------------------------------------------
//   setNodeCoords
//-----------------------------------------------------------------------


void MeshRenderer::setNodeCoords

  ( const Matrix&      coords,
    const AppContext&  apx )

{
  JEM_PRECHECK ( coords.size(0) >= 2 && coords.size(0) <= 3 );

  OwnData_&    d = * ownData_;
  const idx_t  n = appData_[apx]->nodes.size ();


  if ( coords.size(1) != n )
  {
    util::sizeError ( JEM_FUNC, "node coordinate matrix",
                      coords.size(1), n );
  }

  updateMesh_     ( apx );
  d.setNodeCoords ( coords );
}


void MeshRenderer::setNodeCoords

  ( idx_t              idim,
    const Vector&      coords,
    const AppContext&  apx )

{
  JEM_PRECHECK ( idim >= 0 && idim < 3 );

  OwnData_&    d = * ownData_;
  const idx_t  n = appData_[apx]->nodes.size ();


  if ( coords.size() != n )
  {
    util::sizeError ( JEM_FUNC, "node coordinate vector",
                      coords.size(), n );
  }

  updateMesh_     ( apx );
  d.setNodeCoords ( idim, coords );
}


//-----------------------------------------------------------------------
//   addToNodeCoords
//-----------------------------------------------------------------------


void MeshRenderer::addToNodeCoords

  ( const Matrix&      delta,
    const AppContext&  apx )

{
  JEM_PRECHECK ( delta.size(0) >= 2 && delta.size(0) <= 3 );

  OwnData_&    d = * ownData_;
  const idx_t  n = appData_[apx]->nodes.size ();


  if ( delta.size(1) != n )
  {
    util::sizeError ( JEM_FUNC, "node displacement matrix",
                      delta.size(1), n );
  }

  updateMesh_       ( apx );
  d.addToNodeCoords ( delta, *appData_[apx] );
}


void MeshRenderer::addToNodeCoords

  ( idx_t              idim,
    const Vector&      delta,
    const AppContext&  apx )

{
  JEM_PRECHECK ( idim >= 0 && idim < 3 );

  OwnData_&    d = * ownData_;
  const idx_t  n = appData_[apx]->nodes.size ();


  if ( delta.size() != n )
  {
    util::sizeError ( JEM_FUNC, "node displacement vector",
                      delta.size(), n );
  }

  updateMesh_       ( apx );
  d.addToNodeCoords ( idim, delta, *appData_[apx] );
}


//-----------------------------------------------------------------------
//   findElement
//-----------------------------------------------------------------------


idx_t MeshRenderer::findElement

  ( idx_t&  elemID,
    double  x,
    double  y,
    double  z ) const

{
  using jem::gl::Box;

  OwnData_&  d = * ownData_;

  if ( d.mesh == nullptr || d.triangles == nullptr )
  {
    return -1;
  }

  Box      box;
  GLfloat  dist;
  idx_t    itr;


  d.triangles->findTriangle ( itr, dist,
                              (GLfloat) x,
                              (GLfloat) y,
                              (GLfloat) z );

  if ( itr < 0 )
  {
    return -1;
  }

  d.triangles->getBBox ( box );

  if ( box.isEmpty() || (GLdouble) dist > 0.01 * box.size() )
  {
    return -1;
  }
  else
  {
    idx_t  ielem = d.mapTriangToElem ( itr );

    if ( ielem >= 0 )
    {
      elemID = d.elemIDs[ielem];
    }

    return ielem;
  }
}


//-----------------------------------------------------------------------
//   getVertices
//-----------------------------------------------------------------------


VertexSet* MeshRenderer::getVertices () const
{
  return ownData_->vertices.get ();
}


//-----------------------------------------------------------------------
//   getVxNodeMap
//-----------------------------------------------------------------------


IdxVector MeshRenderer::getVxNodeMap () const
{
  return ownData_->mesh->vertexNodeMap;
}


//-----------------------------------------------------------------------
//   getColor
//-----------------------------------------------------------------------


jem::gl::Color MeshRenderer::getColor () const
{
  return ownData_->color;
}


//-----------------------------------------------------------------------
//   getNodeVxMap
//-----------------------------------------------------------------------


IdxVector MeshRenderer::getNodeVxMap ( const AppContext& apx ) const
{
  IdxVector    vxNodeMap   = ownData_->mesh->vertexNodeMap;

  IdxVector    nodeVxMap   ( appData_[apx]->nodes.size() );

  const idx_t  firstXPoint = ownData_->mesh->firstXPoint ();


  nodeVxMap = -1;

  for ( idx_t ivx = 0; ivx < firstXPoint; ivx++ )
  {
    nodeVxMap[vxNodeMap[ivx]] = ivx;
  }

  return nodeVxMap;
}


//-----------------------------------------------------------------------
//   getElemIndices
//-----------------------------------------------------------------------


IdxVector MeshRenderer::getElemIndices

  ( const AppContext&  apx ) const

{
  return appData_[apx]->egroup.getIndices ();
}


//-----------------------------------------------------------------------
//   getElemGroup
//-----------------------------------------------------------------------


ElementGroup MeshRenderer::getElemGroup

  ( const AppContext&  apx ) const

{
  return appData_[apx]->egroup;
}


//-----------------------------------------------------------------------
//   getElements
//-----------------------------------------------------------------------


ElementSet MeshRenderer::getElements

  ( const AppContext&  apx ) const

{
  return appData_[apx]->elems;
}


//-----------------------------------------------------------------------
//   getNodes
//-----------------------------------------------------------------------


NodeSet MeshRenderer::getNodes ( const AppContext& apx ) const
{
  return appData_[apx]->nodes;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<MeshRenderer> MeshRenderer::get

  ( const Properties&  viewdat,
    const String&      context )

{
  Ref<Self>  self;

  if ( ! viewdat.find( self, Viewdat::MESH_RENDERER ) )
  {
    throw IllegalInputException (
      context,
      "no mesh renderer has been defined; "
      "have you enabled the MeshView plugin?"
    );
  }

  return self;
}


//-----------------------------------------------------------------------
//   updateMesh_
//-----------------------------------------------------------------------


bool MeshRenderer::updateMesh_ ( const AppContext& apx )
{
  if ( ownData_->dirtyBits & OwnData_::NEW_MESH )
  {
    ownData_->initMesh ( myName_, *appData_[apx] );
    newMeshEvent.emit  ( *this );

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   checkShapes_
//-----------------------------------------------------------------------


void MeshRenderer::checkShapes_

  ( const ShapeTable&  shapes,
    const ElementSet&  elems ) const

{
  if ( shapes.getItems() != elems.getData() )
  {
    String  shapeTableName = shapes.getName ();
    String  elemSetName    = elems .getName ();

    throw jem::IllegalArgumentException (
      getContext (),
      String::format (
        "shape table `%s\' is not associated with "
        "element set `%s\'",
        shapeTableName,
        elemSetName
      )
    );
  }
}


JIVE_END_PACKAGE( gl )
