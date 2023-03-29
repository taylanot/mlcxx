
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/Object.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/base/array/intrinsics.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/gl/byte.h>
#include <jem/gl/error.h>
#include <jem/gl/render.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Box.h>
#include <jem/gl/ColorMap.h>
#include <jem/gl/Commands.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions3_3.h>
#include <jem/gl/Menu.h>
#include <jem/gl/Rect.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/ShaderProg.h>
#include <jem/gl/Transform.h>
#include <jem/gl/Vertex.h>
#include <jem/gl/VertexSet.h>
#include <jem/gl/VertexBufferObject.h>
#include <jem/gl/TriangleSet.h>
#include "ShaderNames.h"
#include "ShaderSource.h"
#include "VertexArray.h"


JEM_DEFINE_CLASS( jem::gl::TriangleSet );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class TriangleSet::Work_
//=======================================================================


class TriangleSet::Work_
{
 public:

  explicit inline         Work_

    ( const TriangleSet&    ts );


 public:

  const idx_t             trCount;
  const idx_t*            triangs;
  const GLfloat*          vcoords;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline TriangleSet::Work_::Work_ ( const TriangleSet& ts ) :

  trCount ( ts.triangles_.size(1) )

{
  triangs = ts.triangles_.addr    ();
  vcoords = ts.vertices_->getData ();
}


//=======================================================================
//   class TriangleSet::Work2D_
//=======================================================================


class TriangleSet::Work2D_ : public Work_
{
 public:

  explicit inline         Work2D_

    ( const TriangleSet&    ts );

  inline void             getTrCenter

    ( GLfloat*              c )       const;

  GLdouble                distanceTo

    ( const GLfloat*        x )       const;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline TriangleSet::Work2D_::Work2D_ ( const TriangleSet& ts ) :

  Work_ ( ts )

{}


//-----------------------------------------------------------------------
//   getTrCenter
//-----------------------------------------------------------------------


inline void TriangleSet::Work2D_::getTrCenter ( GLfloat* c ) const
{
  idx_t  k = 2 * triangs[0];

  c[0]  = vcoords[k + 0];
  c[1]  = vcoords[k + 1];

  k     = 2 * triangs[1];

  c[0] += vcoords[k + 0];
  c[1] += vcoords[k + 1];

  k     = 2 * triangs[2];

  c[0] += vcoords[k + 0];
  c[1] += vcoords[k + 1];

  c[0] /= 3.0F;
  c[1] /= 3.0F;
}


//-----------------------------------------------------------------------
//   distanceTo
//-----------------------------------------------------------------------


GLdouble TriangleSet::Work2D_::distanceTo ( const GLfloat* x ) const
{
  GLdouble  gx[2][2];
  GLdouble  dx[2];
  GLdouble  u [2];
  GLdouble  tmp;
  GLdouble  det;
  idx_t      i, j, k;


  i        = 2 * triangs[0];
  j        = 2 * triangs[1];
  k        = 2 * triangs[2];

  tmp      = vcoords[i + 0];
  gx[0][0] = vcoords[j + 0] - tmp;
  gx[0][1] = vcoords[k + 0] - tmp;
  dx[0]    = x[0]           - tmp;

  tmp      = vcoords[i + 1];
  gx[1][0] = vcoords[j + 1] - tmp;
  gx[1][1] = vcoords[k + 1] - tmp;
  dx[1]    = x[1]           - tmp;


  det = gx[0][0] * gx[1][1] - gx[0][1] * gx[1][0];

  if ( isTiny( det ) )
  {
    return -1.0;
  }

  tmp  = 1.0 / det;

  u[0] = tmp * (gx[1][1] * dx[0] - gx[0][1] * dx[1]);
  u[1] = tmp * (gx[0][0] * dx[1] - gx[1][0] * dx[0]);

  if ( u[0] < 0.0 )
  {
    u[0] = 0.0;
  }
  if ( u[1] < 0.0 )
  {
    u[1] = 0.0;
  }

  tmp = u[0] + u[1];

  if ( tmp > 1.0 )
  {
    tmp   = 1.0 / tmp;
    u[0] *= tmp;
    u[1] *= tmp;
  }

  dx[0] = dx[0] - u[0] * gx[0][0] - u[1] * gx[0][1];
  dx[1] = dx[1] - u[0] * gx[1][0] - u[1] * gx[1][1];

  tmp   = dx[0] * dx[0] + dx[1] * dx[1];

  return tmp;
}


//=======================================================================
//   class TriangleSet::Work3D_
//=======================================================================


class TriangleSet::Work3D_ : public Work_
{
 public:

  explicit inline         Work3D_

    ( const TriangleSet&    ts );

  inline void             getTrCenter

    ( GLfloat*              c )              const;

  GLdouble                distanceTo

    ( const GLfloat*        x )              const;

  void                    getZOrder

    ( GLContext&            gtx,
      const IdxVector&      iperm );


 public:

  const idx_t             nrCount;
  const GLbyte*           normals;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline TriangleSet::Work3D_::Work3D_ ( const TriangleSet& ts ) :

    Work_   ( ts ),
    nrCount ( ts.normals_.size(1) )

{
  normals = ts.normals_.addr ();
}


//-----------------------------------------------------------------------
//   getTrCenter
//-----------------------------------------------------------------------


inline void TriangleSet::Work3D_::getTrCenter ( GLfloat* c ) const
{
  idx_t  k = 3 * triangs[0];

  c[0]  = vcoords[k + 0];
  c[1]  = vcoords[k + 1];
  c[2]  = vcoords[k + 2];

  k     = 3 * triangs[1];

  c[0] += vcoords[k + 0];
  c[1] += vcoords[k + 1];
  c[2] += vcoords[k + 2];

  k     = 3 * triangs[2];

  c[0] += vcoords[k + 0];
  c[1] += vcoords[k + 1];
  c[2] += vcoords[k + 2];

  c[0] /= 3.0F;
  c[1] /= 3.0F;
  c[2] /= 3.0F;
}


//-----------------------------------------------------------------------
//   distanceTo
//-----------------------------------------------------------------------


GLdouble TriangleSet::Work3D_::distanceTo ( const GLfloat* x ) const
{
  GLdouble  px[2][2];
  GLdouble  dx[3];
  GLdouble  s [3];
  GLdouble  t [3];
  GLdouble  u [2];
  GLdouble  v [2];

  double    tmp;
  double    det;
  idx_t     i, j, k;


  i        = 3 * triangs[0];
  j        = 3 * triangs[1];
  k        = 3 * triangs[2];

  tmp      = vcoords[i + 0];
  s[0]     = vcoords[j + 0] - tmp;
  t[0]     = vcoords[k + 0] - tmp;
  dx[0]    = x[0]           - tmp;

  tmp      = vcoords[i + 1];
  s[1]     = vcoords[j + 1] - tmp;
  t[1]     = vcoords[k + 1] - tmp;
  dx[1]    = x[1]           - tmp;

  tmp      = vcoords[i + 2];
  s[2]     = vcoords[j + 2] - tmp;
  t[2]     = vcoords[k + 2] - tmp;
  dx[2]    = x[2]           - tmp;

  px[0][0] = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
  px[0][1] = s[0] * t[0] + s[1] * t[1] + s[2] * t[2];
  px[1][0] = px[0][1];
  px[1][1] = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];

  v[0]     = s[0] * dx[0] + s[1] * dx[1] + s[2] * dx[2];
  v[1]     = t[0] * dx[0] + t[1] * dx[1] + t[2] * dx[2];

  det      = px[0][0] * px[1][1] - px[0][1] * px[1][0];

  if ( isTiny( det ) )
  {
    return -1.0;
  }

  tmp  = 1.0 / det;

  u[0] = tmp * (px[1][1] * v[0] - px[0][1] * v[1]);
  u[1] = tmp * (px[0][0] * v[1] - px[1][0] * v[0]);

  if ( u[0] < 0.0 )
  {
    u[0] = 0.0;
  }
  if ( u[1] < 0.0 )
  {
    u[1] = 0.0;
  }

  tmp = u[0] + u[1];

  if ( tmp > 1.0 )
  {
    tmp   = 1.0 / tmp;
    u[0] *= tmp;
    u[1] *= tmp;
  }

  dx[0] = dx[0] - u[0] * s[0] - u[1] * t[0];
  dx[1] = dx[1] - u[0] * s[1] - u[1] * t[1];
  dx[2] = dx[2] - u[0] * s[2] - u[1] * t[2];

  tmp   = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

  return tmp;
}


//-----------------------------------------------------------------------
//   getZOrder
//-----------------------------------------------------------------------


void TriangleSet::Work3D_::getZOrder

  ( GLContext&        gtx,
    const IdxVector&  iperm )

{
  const idx_t*    triangs0 = triangs;

  GLfloatVector   z        ( trCount );

  GLdouble        mx[16];
  GLfloat         t0, t1, t2;
  GLfloat         c[3];

  gtx.getTransform ( mx );

  t0 = (GLfloat) mx[ 2];
  t1 = (GLfloat) mx[ 6];
  t2 = (GLfloat) mx[10];

  for ( idx_t i = 0; i < trCount; i++ )
  {
    getTrCenter ( c );

    z[i]     = t0 * c[0] + t1 * c[1] + t2 * c[2];
    iperm[i] = i;

    triangs += 3;
  }

  triangs = triangs0;

  sort ( iperm, z );
}


//=======================================================================
//   class TriangleSet::Buffer_
//=======================================================================


class TriangleSet::Buffer_ : public Collectable
{
 public:

  static const int        POSITIONS  = 1 << 0;
  static const int        NORMALS    = 1 << 1;
  static const int        COLORS     = 1 << 2;
  static const int        TEX_COORDS = 1 << 3;


                          Buffer_       ();

  inline bool             isValid       () const noexcept;
  inline void             setValid      ();
  inline void             invalidate    ();

  inline void             drawArrays

    ( GLContext&            gtx,
      int                   what = 0 );

  void                    drawArrays

    ( GLContext&            gtx,
      const Ref<ColorMap>&  map,
      int                   what = 0 );

  inline GLfloat*         getPositions  () const;
  inline GLfloat*         getNormals    () const;
  inline GLfloat*         getColors     () const;
  inline GLfloat*         getTexCoords  () const;

  inline GLfloat*         allocPositions

    ( idx_t                 size );

  inline GLfloat*         allocNormals

    ( idx_t                 size );

  inline GLfloat*         allocColors

    ( idx_t                 size );

  inline GLfloat*         allocTexCoords

    ( idx_t                 size );

  inline
    const IdxVector&      allocZOrder

    ( idx_t                 size );


 private:

  Ref<VBO>                positions_;
  Ref<VBO>                normals_;
  Ref<VBO>                colors_;
  Ref<VBO>                texCoords_;
  IdxVector               zperm_;

  bool                    valid_;
  bool                    synced_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


TriangleSet::Buffer_::Buffer_ ()
{
  positions_  = newInstance<VBO> ();
  normals_    = newInstance<VBO> ();
  colors_     = newInstance<VBO> ();
  texCoords_  = newInstance<VBO> ();
  valid_      = false;
  synced_     = false;
}


//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


inline bool TriangleSet::Buffer_::isValid () const noexcept
{
  return valid_;
}


//-----------------------------------------------------------------------
//   setValid
//-----------------------------------------------------------------------


inline void TriangleSet::Buffer_::setValid ()
{
  valid_ = true;
}


//-----------------------------------------------------------------------
//   invalidate
//-----------------------------------------------------------------------


inline void TriangleSet::Buffer_::invalidate ()
{
  valid_  = false;
  synced_ = false;
}


//-----------------------------------------------------------------------
//   drawArrays
//-----------------------------------------------------------------------


inline void TriangleSet::Buffer_::drawArrays

  ( GLContext&  gtx,
    int         what )

{
  drawArrays ( gtx, nullptr, what );
}


void TriangleSet::Buffer_::drawArrays

  ( GLContext&            gtx,
    const Ref<ColorMap>&  cmap,
    int                   what )

{
  GLFuncs3_3Core&  funcs    = gtx.getFunctions<GLFuncs3_3Core> ();
  ShaderProg*      shader   = nullptr;
  GLfloat          alphaMin = gtx.getAlphaTest ();

  VertexArray      vao      ( gtx );

  String           name;
  GLsizei          count;
  GLfloat          mvp[16];


  if ( cmap )
  {
    what |= TEX_COORDS;
  }

  if ( what & NORMALS )
  {
    if      ( what & COLORS )
    {
      name = ShaderNames::SHADE_MULTI_COLOR;
    }
    else if ( what & TEX_COORDS )
    {
      name = ShaderNames::SHADE_TEX1D;
    }
    else
    {
      name = ShaderNames::SHADE_CONST_COLOR;
    }
  }
  else
  {
    if      ( what & COLORS )
    {
      name = ShaderNames::FLAT_MULTI_COLOR;
    }
    else if ( what & TEX_COORDS )
    {
      name = ShaderNames::FLAT_TEX1D;
    }
    else
    {
      name = ShaderNames::FLAT_CONST_COLOR;
    }
  }

  shader = & ShaderSource::getProgram ( gtx, name );

  if ( ! shader->bind( gtx ) )
  {
    return;
  }

  gtx.getModelViewProj  ( mvp );
  shader->setUniform    ( gtx, ShaderUnis::ALPHA_MIN, alphaMin );
  shader->setUniform4   ( gtx, ShaderUnis::COLOR, gtx.getColor() );
  shader->setUniMatrix4 ( gtx, ShaderUnis::MVP, mvp );

  if ( what & NORMALS )
  {
    GLfloat  mv  [16];
    GLfloat  proj[16];
    GLfloat  norm[9];

    ShaderDraw::uploadLights ( gtx, *shader );

    gtx.getTransform      ( mv,   GLContext::MODEL_VIEW );
    gtx.getTransform      ( proj, GLContext::PROJECTION );
    gtx.getNormalMatrix   ( norm );

    shader->setUniMatrix4 ( gtx, ShaderUnis::MV,       mv   );
    shader->setUniMatrix4 ( gtx, ShaderUnis::PROJ_MAT, proj );
    shader->setUniMatrix3 ( gtx, ShaderUnis::NORM_MAT, norm );
  }

  if ( what & TEX_COORDS )
  {
    JEM_PRECHECK ( cmap );

    funcs  .glActiveTexture ( GL_TEXTURE0 );
    funcs  .glBindTexture   ( GL_TEXTURE_1D, cmap->getTexID() );
    shader->setUniform      ( gtx, ShaderUnis::TEXTURE_1D, 0 );
  }

  if ( ! synced_ )
  {
    positions_  ->update ( gtx );

    if ( what & NORMALS )
    {
      normals_  ->update ( gtx );
    }

    if ( what & COLORS )
    {
      colors_   ->update ( gtx );
    }

    if ( what & TEX_COORDS )
    {
      texCoords_->update ( gtx );
    }

    if ( valid_ )
    {
      synced_ = true;
    }
  }

  vao.addAttrib   ( positions_->getBufferID( gtx ), 3, GL_FLOAT );

  if ( what & NORMALS )
  {
    vao.addAttrib ( normals_  ->getBufferID( gtx ), 3, GL_FLOAT );
  }

  if ( what & COLORS )
  {
    vao.addAttrib ( colors_   ->getBufferID( gtx ), 4, GL_FLOAT );
  }

  if ( what & TEX_COORDS )
  {
    vao.addAttrib ( texCoords_->getBufferID( gtx ), 1, GL_FLOAT );
  }

  count = (GLsizei) jem::min ( (idx_t) maxOf( count ),
                               positions_->size() / 3_idx );

  funcs.glDrawArrays ( GL_TRIANGLES, 0, count );
  shader->unbind     ( gtx );
}


//-----------------------------------------------------------------------
//   get ...
//-----------------------------------------------------------------------


inline GLfloat* TriangleSet::Buffer_::getPositions () const
{
  return positions_->getData().addr ();
}


inline GLfloat* TriangleSet::Buffer_::getNormals   () const
{
  return normals_  ->getData().addr ();
}


inline GLfloat* TriangleSet::Buffer_::getColors    () const
{
  return colors_   ->getData().addr ();
}


inline GLfloat* TriangleSet::Buffer_::getTexCoords () const
{
  return texCoords_->getData().addr ();
}


//-----------------------------------------------------------------------
//   alloc ...
//-----------------------------------------------------------------------


inline GLfloat* TriangleSet::Buffer_::allocPositions ( idx_t size )
{
  invalidate         ();
  positions_->resize ( size );

  return positions_->getData().addr ();
}


inline GLfloat* TriangleSet::Buffer_::allocNormals   ( idx_t size )
{
  invalidate       ();
  normals_->resize ( size );

  return normals_->getData().addr ();
}


inline GLfloat* TriangleSet::Buffer_::allocColors    ( idx_t size )
{
  invalidate      ();
  colors_->resize ( size );

  return colors_->getData().addr ();
}


inline GLfloat* TriangleSet::Buffer_::allocTexCoords ( idx_t size )
{
  invalidate         ();
  texCoords_->resize ( size );

  return texCoords_->getData().addr ();
}


inline const IdxVector&
  TriangleSet::Buffer_::allocZOrder ( idx_t size )
{
  if ( zperm_.size() != size )
  {
    zperm_.resize ( size );
  }

  return zperm_;
}


//=======================================================================
//   class TriangleSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


TriangleSet::TriangleSet

  ( const String&          name,
    const Ref<VertexSet>&  vx ) :

    Super      ( name ),
    vertices_  (   vx ),
    triangles_ ( 3, 0 )

{
  JEM_PRECHECK ( vx );

  if ( vx->size() > 0 && (vx->rank() < 2 || vx->rank() > 3) )
  {
    shapeError ( JEM_FUNC, "vertex array", vx->shape() );
  }

  colorMode_  = NO_COLORS;
  normalMode_ = AUTO_NORMALS;
  opaque_     = true;
  checked_    = true;
  updated_    = false;
  buffer_     = jem::newInstance<Buffer_> ();

  connectHandlers_ ();
}


TriangleSet::~TriangleSet ()
{}


//-----------------------------------------------------------------------
//   getBBox
//-----------------------------------------------------------------------


void TriangleSet::getBBox ( Box& box ) const
{
  // TO DO: get actual bbox of triangles (not all vertices have to
  //        be used).

  vertices_->getBBox ( box );
}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


void TriangleSet::redraw ( RenderContext& rtx )
{
  if ( this->size() == 0 )
  {
    return;
  }

  GLContext&   gtx       = rtx.getGLContext ();
  const idx_t  rank      = vertices_->rank  ();

  EnableScope  polyScope ( gtx, GL_POLYGON_OFFSET_FILL );
  AttribScope  attribs   ( gtx, GLContext::COLOR |
                                GLContext::LIGHTING );

  if ( ! checked_ )
  {
    checkData_ ( getContext() );
  }

  if ( normalMode_ == AUTO_NORMALS && ! updated_ )
  {
    updateNormals_ ();
  }

  switch ( colorMode_ )
  {
  case NO_COLORS:

    if ( isOpaque( gtx.getColor()[3] ) )
    {
      DisableScope  blendScope ( gtx, GL_BLEND );

      if ( rank == 2 )
      {
        drawNoColors2D_ ( gtx );
      }
      else
      {
        drawNoColors3D_ ( gtx );
      }
    }
    else
    {
      if ( rank == 2 )
      {
        drawNoColors2D_  ( gtx );
      }
      else
      {
        blendNoColors3D_ ( gtx );
      }
    }

    break;

  case TRIANG_COLORS:

    if ( opaque_ )
    {
      DisableScope  blendScope ( gtx, GL_BLEND );

      if ( rank == 2 )
      {
        drawTrColors2D_ ( gtx );
      }
      else
      {
        drawTrColors3D_ ( gtx );
      }
    }
    else
    {
      if ( rank == 2 )
      {
        drawTrColors2D_  ( gtx );
      }
      else
      {
        blendTrColors3D_ ( gtx );
      }
    }

    break;

  case VERTEX_COLORS:

    colorMap_->enable  ( gtx );

    if ( rank == 2 )
    {
      drawVxColors2D_  ( gtx );
    }
    else
    {
      if ( opaque_ )
      {
        drawVxColors3D_  ( gtx );
      }
      else
      {
        blendVxColors3D_ ( gtx );
      }
    }

    colorMap_->disable ( gtx );

    break;
  }
}


//-----------------------------------------------------------------------
//   redrawFast
//-----------------------------------------------------------------------


void TriangleSet::redrawFast

  ( RenderContext&  rtx,
    double          lod )

{
  if ( lod <= 0.0 )
  {
    return;
  }

  if ( lod >= 1.0 )
  {
    Self::redraw ( rtx ); return;
  }

  GLContext&    gtx        = rtx.getGLContext ();
  const idx_t   rank       = vertices_->rank  ();

  DisableScope  blendScope ( gtx, GL_BLEND );
  AttribScope   attribs    ( gtx, GLContext::COLOR |
                                  GLContext::POINT_SIZE );

  if ( ! checked_ )
  {
    checkData_ ( getContext() );
  }

  if ( normalMode_ == AUTO_NORMALS && ! updated_ )
  {
    updateNormals_ ();
  }

  if ( gtx.getPointSize() <= 0.0F )
  {
    return;
  }

  switch ( colorMode_ )
  {
  case NO_COLORS:

    if ( rank == 2 )
    {
      plotNoColors2D_  ( gtx, lod );
    }
    else
    {
      plotNoColors3D_  ( gtx, lod );
    }

    break;

  case TRIANG_COLORS:

    if ( rank == 2 )
    {
      plotTrColors2D_ ( gtx, lod );
    }
    else
    {
      plotTrColors3D_ ( gtx, lod );
    }

    break;

  case VERTEX_COLORS:

    colorMap_->enable  ( gtx );

    if ( rank == 2 )
    {
      plotVxColors2D_  ( gtx, lod );
    }
    else
    {
      plotVxColors3D_  ( gtx, lod );
    }

    colorMap_->disable ( gtx );

    break;
  }
}


//-----------------------------------------------------------------------
//   exeCommand
//-----------------------------------------------------------------------


bool TriangleSet::exeCommand

  ( RenderContext&     rtx,
    const String&      cmd,
    const Properties&  params )

{
  if ( cmd == Commands::HIDE_FACES )
  {
    hidden = true;
    return   true;
  }

  if ( cmd == Commands::SHOW_FACES )
  {
    hidden = false;
    return   true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   listOptions
//-----------------------------------------------------------------------


void TriangleSet::listOptions ( Menu& menu ) const
{
  menu.addOption ( Commands::HIDE_FACES,
                   Commands::SHOW_FACES, hidden );
}


//-----------------------------------------------------------------------
//   getTopology
//-----------------------------------------------------------------------


IdxMatrix TriangleSet::getTopology () const
{
  return triangles_;
}


//-----------------------------------------------------------------------
//   setTopology
//-----------------------------------------------------------------------


void TriangleSet::setTopology ( const IdxMatrix& topo )
{
  JEM_PRECHECK ( topo.size(0) == 3 );

  triangles_.ref ( makeContiguous( topo ) );

  buffer_->invalidate ();
  setNoColors         ();
  setAutoNormals      ();

  checked_ = false;
  updated_ = false;

  checkData_ ( JEM_FUNC );
}


//-----------------------------------------------------------------------
//   setNoColors
//-----------------------------------------------------------------------


void TriangleSet::setNoColors ()
{
  if ( colorMode_ != NO_COLORS )
  {
    buffer_->invalidate ();

    trColors_.ref ( GLubyteMatrix() );
    vxColors_.ref ( GLfloatVector() );

    opaque_    = true;
    colorMap_  = nullptr;
    colorMode_ = NO_COLORS;
  }
}


//-----------------------------------------------------------------------
//   setTriangColors
//-----------------------------------------------------------------------


void TriangleSet::setTriangColors ( const GLubyteMatrix& colors )
{
  JEM_PRECHECK ( colors.size(0) >= 3 &&
                 colors.size(0) <= 4 );

  const idx_t  n    = colors.size (1);

  GLubyte     alpha = toUbyte ( Color::OPAQUE_ALPHA );


  buffer_->invalidate ();

  if ( n != triangles_.size(1) )
  {
    shapeError ( JEM_FUNC,
                 "triangle color matrix", colors.shape() );
  }

  trColors_.ref ( makeContiguous( colors ) );
  vxColors_.ref ( GLfloatVector() );

  if ( colors.size(0) == 4 && n > 0 )
  {
    opaque_ = testall ( colors(3,ALL) >= alpha );
  }
  else
  {
    opaque_ = false;
  }

  colorMap_  = nullptr;
  colorMode_ = TRIANG_COLORS;
}


//-----------------------------------------------------------------------
//   setVertexColors
//-----------------------------------------------------------------------


void TriangleSet::setVertexColors

  ( const GLfloatVector&  colors,
    const Ref<ColorMap>&  colorMap )

{
  JEM_PRECHECK ( colorMap );

  if ( colors.size() != vertices_->size() )
  {
    sizeError ( JEM_FUNC,
                "vertex color vector", colors.size() );
  }

  buffer_->invalidate ();

  vxColors_.ref ( makeContiguous( colors ) );
  trColors_.ref ( GLubyteMatrix() );

  opaque_    = isOpaque ( colorMap->getMinAlpha() );
  colorMap_  = colorMap;
  colorMode_ = VERTEX_COLORS;
}


//-----------------------------------------------------------------------
//   setAutoNormals
//-----------------------------------------------------------------------


void TriangleSet::setAutoNormals ()
{
  if ( normalMode_ != AUTO_NORMALS )
  {
    buffer_->invalidate ();

    normals_.ref ( GLbyteMatrix() );

    normalMode_ = AUTO_NORMALS;
    updated_    = false;
  }
}


//-----------------------------------------------------------------------
//   setTriangNormals
//-----------------------------------------------------------------------


void TriangleSet::setTriangNormals ( const GLbyteMatrix& normals )
{
  const idx_t  k = triangles_.size (1);
  const idx_t  n = normals   .size (1);

  buffer_->invalidate ();

  if ( normals.size(0) != 3 || (n != k && n != 3 * k) )
  {
    shapeError ( JEM_FUNC,
                 "triangle normal matrix", normals.shape() );
  }

  normals_.ref ( makeContiguous( normals ) );

  normalMode_ = TRIANG_NORMALS;
}


//-----------------------------------------------------------------------
//   setVertexNormals
//-----------------------------------------------------------------------


void TriangleSet::setVertexNormals ( const GLbyteMatrix& normals )
{
  if ( normals.size(0) != 3 ||
       normals.size(1) != vertices_->size() )
  {
    shapeError ( JEM_FUNC,
                 "vertex normal matrix", normals.shape() );
  }

  buffer_->invalidate ();

  normals_.ref ( makeContiguous( normals ) );

  normalMode_ = VERTEX_NORMALS;
}


//-----------------------------------------------------------------------
//   findTriangle
//-----------------------------------------------------------------------


void TriangleSet::findTriangle

  ( idx_t&    index,
    GLfloat&  dist,
    GLfloat   x,
    GLfloat   y,
    GLfloat   z ) const

{
  if ( ! checked_ )
  {
    const_cast<Self*>(this)->checkData_ ( getContext() );
  }

  GLfloat  pos[3] = { x, y, z };


  index = -1_idx;
  dist  = 0.0F;

  if ( vertices_->rank() == 2 )
  {
    findTriangle2D_ ( index, dist, pos );
  }
  else
  {
    findTriangle3D_ ( index, dist, pos );
  }
}


//-----------------------------------------------------------------------
//   drawNoColors2D_
//-----------------------------------------------------------------------


void TriangleSet::drawNoColors2D_ ( GLContext& gtx ) const
{
  Work2D_  w ( *this );

  if ( ! buffer_->isValid() )
  {
    const idx_t  vcount  = 9 * w.trCount;
    GLfloat*     vertPos = buffer_->allocPositions ( vcount );

    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      const idx_t  addr0 = 2 * w.triangs[0];
      const idx_t  addr1 = 2 * w.triangs[1];
      const idx_t  addr2 = 2 * w.triangs[2];

      vertPos[0] = w.vcoords[addr0 + 0];
      vertPos[1] = w.vcoords[addr0 + 1];
      vertPos[2] = 0.0F;

      vertPos[3] = w.vcoords[addr1 + 0];
      vertPos[4] = w.vcoords[addr1 + 1];
      vertPos[5] = 0.0F;

      vertPos[6] = w.vcoords[addr2 + 0];
      vertPos[7] = w.vcoords[addr2 + 1];
      vertPos[8] = 0.0F;

      vertPos   += 9;
      w.triangs += 3;
    }

    buffer_->setValid ();
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    buffer_->drawArrays ( gtx );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      const GLfloat*  vertPos = buffer_->getPositions ();

      funcs->glBegin  ( GL_TRIANGLES  );
      funcs->glNormal ( 0.0, 0.0, 1.0 );

      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        funcs->glVertex2 ( vertPos + 0);
        funcs->glVertex2 ( vertPos + 3 );
        funcs->glVertex2 ( vertPos + 6 );

        vertPos += 9;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawNoColors3D_
//-----------------------------------------------------------------------


void TriangleSet::drawNoColors3D_ ( GLContext& gtx ) const
{
  Work3D_  w ( *this );

  if ( ! buffer_->isValid() )
  {
    const idx_t  vcount  = 9 * w.trCount;

    GLfloat*     vertPos = buffer_->allocPositions ( vcount );
    GLfloat*     normals = buffer_->allocNormals   ( vcount );

    if ( normalMode_ == VERTEX_NORMALS )
    {
      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        const idx_t  addr0 = 3 * w.triangs[0];
        const idx_t  addr1 = 3 * w.triangs[1];
        const idx_t  addr2 = 3 * w.triangs[2];

        vertPos[0] = w.vcoords[addr0 + 0];
        vertPos[1] = w.vcoords[addr0 + 1];
        vertPos[2] = w.vcoords[addr0 + 2];

        vertPos[3] = w.vcoords[addr1 + 0];
        vertPos[4] = w.vcoords[addr1 + 1];
        vertPos[5] = w.vcoords[addr1 + 2];

        vertPos[6] = w.vcoords[addr2 + 0];
        vertPos[7] = w.vcoords[addr2 + 1];
        vertPos[8] = w.vcoords[addr2 + 2];

        normals[0] = toFloat ( w.normals[addr0 + 0] );
        normals[1] = toFloat ( w.normals[addr0 + 1] );
        normals[2] = toFloat ( w.normals[addr0 + 2] );

        normals[3] = toFloat ( w.normals[addr1 + 0] );
        normals[4] = toFloat ( w.normals[addr1 + 1] );
        normals[5] = toFloat ( w.normals[addr1 + 2] );

        normals[6] = toFloat ( w.normals[addr2 + 0] );
        normals[7] = toFloat ( w.normals[addr2 + 1] );
        normals[8] = toFloat ( w.normals[addr2 + 2] );

        vertPos   += 9;
        normals   += 9;
        w.triangs += 3;
      }
    }
    else if ( w.nrCount == w.trCount )
    {
      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        const idx_t  addr0 = 3 * w.triangs[0];
        const idx_t  addr1 = 3 * w.triangs[1];
        const idx_t  addr2 = 3 * w.triangs[2];

        vertPos[0] = w.vcoords[addr0 + 0];
        vertPos[1] = w.vcoords[addr0 + 1];
        vertPos[2] = w.vcoords[addr0 + 2];

        vertPos[3] = w.vcoords[addr1 + 0];
        vertPos[4] = w.vcoords[addr1 + 1];
        vertPos[5] = w.vcoords[addr1 + 2];

        vertPos[6] = w.vcoords[addr2 + 0];
        vertPos[7] = w.vcoords[addr2 + 1];
        vertPos[8] = w.vcoords[addr2 + 2];

        normals[0] = toFloat ( w.normals[0] );
        normals[1] = toFloat ( w.normals[1] );
        normals[2] = toFloat ( w.normals[2] );

        normals[3] = toFloat ( w.normals[0] );
        normals[4] = toFloat ( w.normals[1] );
        normals[5] = toFloat ( w.normals[2] );

        normals[6] = toFloat ( w.normals[0] );
        normals[7] = toFloat ( w.normals[1] );
        normals[8] = toFloat ( w.normals[2] );

        vertPos   += 9;
        normals   += 9;
        w.triangs += 3;
        w.normals += 3;
      }
    }
    else
    {
      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        const idx_t  addr0 = 3 * w.triangs[0];
        const idx_t  addr1 = 3 * w.triangs[1];
        const idx_t  addr2 = 3 * w.triangs[2];

        vertPos[0] = w.vcoords[addr0 + 0];
        vertPos[1] = w.vcoords[addr0 + 1];
        vertPos[2] = w.vcoords[addr0 + 2];

        vertPos[3] = w.vcoords[addr1 + 0];
        vertPos[4] = w.vcoords[addr1 + 1];
        vertPos[5] = w.vcoords[addr1 + 2];

        vertPos[6] = w.vcoords[addr2 + 0];
        vertPos[7] = w.vcoords[addr2 + 1];
        vertPos[8] = w.vcoords[addr2 + 2];

        normals[0] = toFloat ( w.normals[0] );
        normals[1] = toFloat ( w.normals[1] );
        normals[2] = toFloat ( w.normals[2] );

        normals[3] = toFloat ( w.normals[3] );
        normals[4] = toFloat ( w.normals[4] );
        normals[5] = toFloat ( w.normals[5] );

        normals[6] = toFloat ( w.normals[6] );
        normals[7] = toFloat ( w.normals[7] );
        normals[8] = toFloat ( w.normals[8] );

        vertPos   += 9;
        normals   += 9;
        w.triangs += 3;
        w.normals += 9;
      }
    }

    buffer_->setValid ();
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    buffer_->drawArrays ( gtx, Buffer_::NORMALS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      const GLfloat*  vertPos = buffer_->getPositions ();
      const GLfloat*  normals = buffer_->getNormals   ();

      funcs->glBegin ( GL_TRIANGLES  );

      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        funcs->glNormal3 ( normals + 0 );
        funcs->glVertex3 ( vertPos + 0 );
        funcs->glNormal3 ( normals + 3 );
        funcs->glVertex3 ( vertPos + 3 );
        funcs->glNormal3 ( normals + 6 );
        funcs->glVertex3 ( vertPos + 6 );

        vertPos += 9;
        normals += 9;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawTrColors2D_
//-----------------------------------------------------------------------


void TriangleSet::drawTrColors2D_ ( GLContext& gtx ) const
{
  Work2D_  w ( *this );

  if ( ! buffer_->isValid() )
  {
    const GLubyte*  colors  = trColors_.addr ();
    const idx_t     cst     = trColors_.size (0);
    const idx_t     vcount  =  9 * w.trCount;
    const idx_t     ccount  = 12 * w.trCount;

    GLfloat*        vertPos = buffer_->allocPositions ( vcount );
    GLfloat*        vertCol = buffer_->allocColors    ( ccount );

    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      const idx_t  addr0 = 2 * w.triangs[0];
      const idx_t  addr1 = 2 * w.triangs[1];
      const idx_t  addr2 = 2 * w.triangs[2];

      vertPos[ 0] = w.vcoords[addr0 + 0];
      vertPos[ 1] = w.vcoords[addr0 + 1];
      vertPos[ 2] = 0.0F;

      vertPos[ 3] = w.vcoords[addr1 + 0];
      vertPos[ 4] = w.vcoords[addr1 + 1];
      vertPos[ 5] = 0.0F;

      vertPos[ 6] = w.vcoords[addr2 + 0];
      vertPos[ 7] = w.vcoords[addr2 + 1];
      vertPos[ 8] = 0.0F;

      vertCol[ 0] = colors[0];
      vertCol[ 1] = colors[1];
      vertCol[ 2] = colors[2];
      vertCol[ 3] = (cst == 3) ? 1.0F : colors[3];

      vertCol[ 4] = colors[0];
      vertCol[ 5] = colors[1];
      vertCol[ 6] = colors[2];
      vertCol[ 7] = (cst == 3) ? 1.0F : colors[3];

      vertCol[ 8] = colors[0];
      vertCol[ 9] = colors[1];
      vertCol[10] = colors[2];
      vertCol[11] = (cst == 3) ? 1.0F : colors[3];

      vertPos    += 9;
      vertCol    += 12;
      w.triangs  += 3;
      colors     += cst;
    }

    buffer_->setValid ();
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    buffer_->drawArrays ( gtx, Buffer_::COLORS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      const GLfloat*  vertPos = buffer_->getPositions ();
      const GLfloat*  vertCol = buffer_->getColors    ();

      funcs->glBegin  ( GL_TRIANGLES  );
      funcs->glNormal ( 0.0, 0.0, 1.0 );

      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        funcs->glColor4  ( vertCol + 0 );
        funcs->glVertex2 ( vertPos + 0 );
        funcs->glVertex2 ( vertPos + 3 );
        funcs->glVertex2 ( vertPos + 6 );

        vertPos += 9;
        vertCol += 12;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawTrColors3D_
//-----------------------------------------------------------------------


void TriangleSet::drawTrColors3D_ ( GLContext& gtx ) const
{
  Work3D_  w ( *this );

  if ( ! buffer_->isValid() )
  {
    const GLubyte*  colors  = trColors_.addr ();
    const idx_t     cst     = trColors_.size (0);
    const idx_t     vcount  =  9 * w.trCount;
    const idx_t     ccount  = 12 * w.trCount;

    GLfloat*        vertPos = buffer_->allocPositions ( vcount );
    GLfloat*        normals = buffer_->allocNormals   ( vcount );
    GLfloat*        vertCol = buffer_->allocColors    ( ccount );

    if ( normalMode_ == VERTEX_NORMALS )
    {
      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        const idx_t  addr0 = 3 * w.triangs[0];
        const idx_t  addr1 = 3 * w.triangs[1];
        const idx_t  addr2 = 3 * w.triangs[2];

        vertPos[ 0] = w.vcoords[addr0 + 0];
        vertPos[ 1] = w.vcoords[addr0 + 1];
        vertPos[ 2] = w.vcoords[addr0 + 2];

        vertPos[ 3] = w.vcoords[addr1 + 0];
        vertPos[ 4] = w.vcoords[addr1 + 1];
        vertPos[ 5] = w.vcoords[addr1 + 2];

        vertPos[ 6] = w.vcoords[addr2 + 0];
        vertPos[ 7] = w.vcoords[addr2 + 1];
        vertPos[ 8] = w.vcoords[addr2 + 2];

        normals[ 0] = toFloat ( w.normals[addr0 + 0] );
        normals[ 1] = toFloat ( w.normals[addr0 + 1] );
        normals[ 2] = toFloat ( w.normals[addr0 + 2] );

        normals[ 3] = toFloat ( w.normals[addr1 + 0] );
        normals[ 4] = toFloat ( w.normals[addr1 + 1] );
        normals[ 5] = toFloat ( w.normals[addr1 + 2] );

        normals[ 6] = toFloat ( w.normals[addr2 + 0] );
        normals[ 7] = toFloat ( w.normals[addr2 + 1] );
        normals[ 8] = toFloat ( w.normals[addr2 + 2] );

        vertCol[ 0] = colors[0];
        vertCol[ 1] = colors[1];
        vertCol[ 2] = colors[2];
        vertCol[ 3] = (cst == 3) ? 1.0F : colors[3];

        vertCol[ 4] = colors[0];
        vertCol[ 5] = colors[1];
        vertCol[ 6] = colors[2];
        vertCol[ 7] = (cst == 3) ? 1.0F : colors[3];

        vertCol[ 8] = colors[0];
        vertCol[ 9] = colors[1];
        vertCol[10] = colors[2];
        vertCol[11] = (cst == 3) ? 1.0F : colors[3];

        vertPos    += 9;
        normals    += 9;
        vertCol    += 12;
        w.triangs  += 3;
        colors     += cst;
      }
    }
    else if ( w.nrCount == w.trCount )
    {
      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        const idx_t  addr0 = 3 * w.triangs[0];
        const idx_t  addr1 = 3 * w.triangs[1];
        const idx_t  addr2 = 3 * w.triangs[2];

        vertPos[ 0] = w.vcoords[addr0 + 0];
        vertPos[ 1] = w.vcoords[addr0 + 1];
        vertPos[ 2] = w.vcoords[addr0 + 2];

        vertPos[ 3] = w.vcoords[addr1 + 0];
        vertPos[ 4] = w.vcoords[addr1 + 1];
        vertPos[ 5] = w.vcoords[addr1 + 2];

        vertPos[ 6] = w.vcoords[addr2 + 0];
        vertPos[ 7] = w.vcoords[addr2 + 1];
        vertPos[ 8] = w.vcoords[addr2 + 2];

        normals[ 0] = toFloat ( w.normals[0] );
        normals[ 1] = toFloat ( w.normals[1] );
        normals[ 2] = toFloat ( w.normals[2] );

        normals[ 3] = toFloat ( w.normals[0] );
        normals[ 4] = toFloat ( w.normals[1] );
        normals[ 5] = toFloat ( w.normals[2] );

        normals[ 6] = toFloat ( w.normals[0] );
        normals[ 7] = toFloat ( w.normals[1] );
        normals[ 8] = toFloat ( w.normals[2] );

        vertCol[ 0] = colors[0];
        vertCol[ 1] = colors[1];
        vertCol[ 2] = colors[2];
        vertCol[ 3] = (cst == 3) ? 1.0F : colors[3];

        vertCol[ 4] = colors[0];
        vertCol[ 5] = colors[1];
        vertCol[ 6] = colors[2];
        vertCol[ 7] = (cst == 3) ? 1.0F : colors[3];

        vertCol[ 8] = colors[0];
        vertCol[ 9] = colors[1];
        vertCol[10] = colors[2];
        vertCol[11] = (cst == 3) ? 1.0F : colors[3];

        vertPos    += 9;
        normals    += 9;
        vertCol    += 12;
        w.triangs  += 3;
        w.normals  += 3;
        colors     += cst;
      }
    }
    else
    {
      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        const idx_t  addr0 = 3 * w.triangs[0];
        const idx_t  addr1 = 3 * w.triangs[1];
        const idx_t  addr2 = 3 * w.triangs[2];

        vertPos[ 0] = w.vcoords[addr0 + 0];
        vertPos[ 1] = w.vcoords[addr0 + 1];
        vertPos[ 2] = w.vcoords[addr0 + 2];

        vertPos[ 3] = w.vcoords[addr1 + 0];
        vertPos[ 4] = w.vcoords[addr1 + 1];
        vertPos[ 5] = w.vcoords[addr1 + 2];

        vertPos[ 6] = w.vcoords[addr2 + 0];
        vertPos[ 7] = w.vcoords[addr2 + 1];
        vertPos[ 8] = w.vcoords[addr2 + 2];

        normals[ 0] = toFloat ( w.normals[0] );
        normals[ 1] = toFloat ( w.normals[1] );
        normals[ 2] = toFloat ( w.normals[2] );

        normals[ 3] = toFloat ( w.normals[3] );
        normals[ 4] = toFloat ( w.normals[4] );
        normals[ 5] = toFloat ( w.normals[5] );

        normals[ 6] = toFloat ( w.normals[6] );
        normals[ 7] = toFloat ( w.normals[7] );
        normals[ 8] = toFloat ( w.normals[8] );

        vertCol[ 0] = colors[0];
        vertCol[ 1] = colors[1];
        vertCol[ 2] = colors[2];
        vertCol[ 3] = (cst == 3) ? 1.0F : colors[3];

        vertCol[ 4] = colors[0];
        vertCol[ 5] = colors[1];
        vertCol[ 6] = colors[2];
        vertCol[ 7] = (cst == 3) ? 1.0F : colors[3];

        vertCol[ 8] = colors[0];
        vertCol[ 9] = colors[1];
        vertCol[10] = colors[2];
        vertCol[11] = (cst == 3) ? 1.0F : colors[3];

        vertPos    += 9;
        normals    += 9;
        vertCol    += 12;
        w.triangs  += 3;
        w.normals  += 9;
        colors     += cst;
      }
    }

    buffer_->setValid ();
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    buffer_->drawArrays ( gtx, Buffer_::NORMALS | Buffer_::COLORS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      const GLfloat*  vertPos = buffer_->getPositions ();
      const GLfloat*  normals = buffer_->getNormals   ();
      const GLfloat*  vertCol = buffer_->getColors    ();

      funcs->glBegin  ( GL_TRIANGLES  );

      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        funcs->glColor4  ( vertCol + 0 );
        funcs->glNormal3 ( normals + 0 );
        funcs->glVertex3 ( vertPos + 0 );
        funcs->glNormal3 ( normals + 3 );
        funcs->glVertex3 ( vertPos + 3 );
        funcs->glNormal3 ( normals + 6 );
        funcs->glVertex3 ( vertPos + 6 );

        vertPos +=  9;
        normals +=  9;
        vertCol += 12;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawVxColors2D_
//-----------------------------------------------------------------------


void TriangleSet::drawVxColors2D_ ( GLContext& gtx ) const
{
  Work2D_  w ( *this );

  if ( ! buffer_->isValid() )
  {
    const idx_t     vcount  = 9 * w.trCount;
    const idx_t     tcount  = 3 * w.trCount;

    const GLfloat*  colors  = vxColors_.addr ();
    GLfloat*        vertPos = buffer_->allocPositions ( vcount );
    GLfloat*        texCrds = buffer_->allocTexCoords ( tcount );

    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      const idx_t  addr0 = 2 * w.triangs[0];
      const idx_t  addr1 = 2 * w.triangs[1];
      const idx_t  addr2 = 2 * w.triangs[2];

      vertPos[0] = w.vcoords[addr0 + 0];
      vertPos[1] = w.vcoords[addr0 + 1];
      vertPos[2] = 0.0F;

      vertPos[3] = w.vcoords[addr1 + 0];
      vertPos[4] = w.vcoords[addr1 + 1];
      vertPos[5] = 0.0F;

      vertPos[6] = w.vcoords[addr2 + 0];
      vertPos[7] = w.vcoords[addr2 + 1];
      vertPos[8] = 0.0F;

      texCrds[0] = colors[w.triangs[0]];
      texCrds[1] = colors[w.triangs[1]];
      texCrds[2] = colors[w.triangs[2]];

      vertPos   += 9;
      texCrds   += 3;
      w.triangs += 3;
    }

    buffer_->setValid ();
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    buffer_->drawArrays ( gtx, colorMap_ );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      const GLfloat*  vertPos = buffer_->getPositions ();
      const GLfloat*  texCrds = buffer_->getTexCoords ();

      funcs->glBegin  ( GL_TRIANGLES  );
      funcs->glNormal ( 0.0, 0.0, 1.0 );

      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        funcs->glTexCoord ( texCrds[0] );
        funcs->glVertex2  ( vertPos );
        funcs->glTexCoord ( texCrds[1] );
        funcs->glVertex2  ( vertPos + 3 );
        funcs->glTexCoord ( texCrds[2]);
        funcs->glVertex2  ( vertPos + 6 );

        vertPos += 9;
        texCrds += 3;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawVxColors3D_
//-----------------------------------------------------------------------


void TriangleSet::drawVxColors3D_ ( GLContext& gtx ) const
{
  Work3D_  w ( *this );

  if ( ! buffer_->isValid() )
  {
    const idx_t     vcount  = 9 * w.trCount;
    const idx_t     tcount  = 3 * w.trCount;

    const GLfloat*  colors  = vxColors_.addr ();
    GLfloat*        vertPos = buffer_->allocPositions ( vcount );
    GLfloat*        normals = buffer_->allocNormals   ( vcount );
    GLfloat*        texCrds = buffer_->allocTexCoords ( tcount );

    if ( normalMode_ == VERTEX_NORMALS )
    {
      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        const idx_t  addr0 = 3 * w.triangs[0];
        const idx_t  addr1 = 3 * w.triangs[1];
        const idx_t  addr2 = 3 * w.triangs[2];

        vertPos[0] = w.vcoords[addr0 + 0];
        vertPos[1] = w.vcoords[addr0 + 1];
        vertPos[2] = w.vcoords[addr0 + 2];

        vertPos[3] = w.vcoords[addr1 + 0];
        vertPos[4] = w.vcoords[addr1 + 1];
        vertPos[5] = w.vcoords[addr1 + 2];

        vertPos[6] = w.vcoords[addr2 + 0];
        vertPos[7] = w.vcoords[addr2 + 1];
        vertPos[8] = w.vcoords[addr2 + 2];

        normals[0] = toFloat ( w.normals[addr0 + 0] );
        normals[1] = toFloat ( w.normals[addr0 + 1] );
        normals[2] = toFloat ( w.normals[addr0 + 2] );

        normals[3] = toFloat ( w.normals[addr1 + 0] );
        normals[4] = toFloat ( w.normals[addr1 + 1] );
        normals[5] = toFloat ( w.normals[addr1 + 2] );

        normals[6] = toFloat ( w.normals[addr2 + 0] );
        normals[7] = toFloat ( w.normals[addr2 + 1] );
        normals[8] = toFloat ( w.normals[addr2 + 2] );

        texCrds[0] = colors[w.triangs[0]];
        texCrds[1] = colors[w.triangs[1]];
        texCrds[2] = colors[w.triangs[2]];

        vertPos   += 9;
        normals   += 9;
        texCrds   += 3;
        w.triangs += 3;
      }
    }
    else if ( w.nrCount == w.trCount )
    {
      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        const idx_t  addr0 = 3 * w.triangs[0];
        const idx_t  addr1 = 3 * w.triangs[1];
        const idx_t  addr2 = 3 * w.triangs[2];

        vertPos[0] = w.vcoords[addr0 + 0];
        vertPos[1] = w.vcoords[addr0 + 1];
        vertPos[2] = w.vcoords[addr0 + 2];

        vertPos[3] = w.vcoords[addr1 + 0];
        vertPos[4] = w.vcoords[addr1 + 1];
        vertPos[5] = w.vcoords[addr1 + 2];

        vertPos[6] = w.vcoords[addr2 + 0];
        vertPos[7] = w.vcoords[addr2 + 1];
        vertPos[8] = w.vcoords[addr2 + 2];

        normals[0] = toFloat ( w.normals[0] );
        normals[1] = toFloat ( w.normals[1] );
        normals[2] = toFloat ( w.normals[2] );

        normals[3] = toFloat ( w.normals[0] );
        normals[4] = toFloat ( w.normals[1] );
        normals[5] = toFloat ( w.normals[2] );

        normals[6] = toFloat ( w.normals[0] );
        normals[7] = toFloat ( w.normals[1] );
        normals[8] = toFloat ( w.normals[2] );

        texCrds[0] = colors[w.triangs[0]];
        texCrds[1] = colors[w.triangs[1]];
        texCrds[2] = colors[w.triangs[2]];

        vertPos   += 9;
        normals   += 9;
        texCrds   += 3;
        w.triangs += 3;
        w.normals += 3;
      }
    }
    else
    {
      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        const idx_t  addr0 = 3 * w.triangs[0];
        const idx_t  addr1 = 3 * w.triangs[1];
        const idx_t  addr2 = 3 * w.triangs[2];

        vertPos[0] = w.vcoords[addr0 + 0];
        vertPos[1] = w.vcoords[addr0 + 1];
        vertPos[2] = w.vcoords[addr0 + 2];

        vertPos[3] = w.vcoords[addr1 + 0];
        vertPos[4] = w.vcoords[addr1 + 1];
        vertPos[5] = w.vcoords[addr1 + 2];

        vertPos[6] = w.vcoords[addr2 + 0];
        vertPos[7] = w.vcoords[addr2 + 1];
        vertPos[8] = w.vcoords[addr2 + 2];

        normals[0] = toFloat ( w.normals[0] );
        normals[1] = toFloat ( w.normals[1] );
        normals[2] = toFloat ( w.normals[2] );

        normals[3] = toFloat ( w.normals[3] );
        normals[4] = toFloat ( w.normals[4] );
        normals[5] = toFloat ( w.normals[5] );

        normals[6] = toFloat ( w.normals[6] );
        normals[7] = toFloat ( w.normals[7] );
        normals[8] = toFloat ( w.normals[8] );

        texCrds[0] = colors[w.triangs[0]];
        texCrds[1] = colors[w.triangs[1]];
        texCrds[2] = colors[w.triangs[2]];

        vertPos   += 9;
        normals   += 9;
        texCrds   += 3;
        w.triangs += 3;
        w.normals += 9;
      }
    }

    buffer_->setValid ();
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    buffer_->drawArrays ( gtx, colorMap_, Buffer_::NORMALS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      const GLfloat*  vertPos = buffer_->getPositions ();
      const GLfloat*  normals = buffer_->getNormals   ();
      const GLfloat*  texCrds = buffer_->getTexCoords ();

      funcs->glBegin ( GL_TRIANGLES  );

      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        funcs->glTexCoord ( texCrds[0] );
        funcs->glNormal3  ( normals );
        funcs->glVertex3  ( vertPos );
        funcs->glTexCoord ( texCrds[1] );
        funcs->glNormal3  ( normals + 3 );
        funcs->glVertex3  ( vertPos + 3 );
        funcs->glTexCoord ( texCrds[2] );
        funcs->glNormal3  ( normals + 6 );
        funcs->glVertex3  ( vertPos + 6 );

        vertPos   += 9;
        normals   += 9;
        texCrds   += 3;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   blendNoColors3D_
//-----------------------------------------------------------------------


void TriangleSet::blendNoColors3D_ ( GLContext& gtx ) const
{
  Work3D_       w       ( *this );

  const idx_t   vcount  = 9 * w.trCount;
  GLfloat*      vertPos = buffer_->allocPositions ( vcount );
  GLfloat*      normals = buffer_->allocNormals   ( vcount );
  IdxVector     iperm   = buffer_->allocZOrder ( w.trCount );

  GLboolean     zmask;


  beginBlend_ ( gtx, zmask );
  w.getZOrder ( gtx, iperm );

  if ( normalMode_ == VERTEX_NORMALS )
  {
    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      idx_t        j     = 3 * iperm[i];
      const idx_t  addr0 = 3 * w.triangs[j + 0];
      const idx_t  addr1 = 3 * w.triangs[j + 1];
      const idx_t  addr2 = 3 * w.triangs[j + 2];

      vertPos[0] = w.vcoords[addr0 + 0];
      vertPos[1] = w.vcoords[addr0 + 1];
      vertPos[2] = w.vcoords[addr0 + 2];

      vertPos[3] = w.vcoords[addr1 + 0];
      vertPos[4] = w.vcoords[addr1 + 1];
      vertPos[5] = w.vcoords[addr1 + 2];

      vertPos[6] = w.vcoords[addr2 + 0];
      vertPos[7] = w.vcoords[addr2 + 1];
      vertPos[8] = w.vcoords[addr2 + 2];

      normals[0] = toFloat ( w.normals[addr0 + 0] );
      normals[1] = toFloat ( w.normals[addr0 + 1] );
      normals[2] = toFloat ( w.normals[addr0 + 2] );

      normals[3] = toFloat ( w.normals[addr1 + 0] );
      normals[4] = toFloat ( w.normals[addr1 + 1] );
      normals[5] = toFloat ( w.normals[addr1 + 2] );

      normals[6] = toFloat ( w.normals[addr2 + 0] );
      normals[7] = toFloat ( w.normals[addr2 + 1] );
      normals[8] = toFloat ( w.normals[addr2 + 2] );

      vertPos   += 9;
      normals   += 9;
    }
  }
  else if ( w.nrCount == w.trCount )
  {
    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      idx_t        j     = 3 * iperm[i];
      const idx_t  addr0 = 3 * w.triangs[j + 0];
      const idx_t  addr1 = 3 * w.triangs[j + 1];
      const idx_t  addr2 = 3 * w.triangs[j + 2];

      vertPos[0] = w.vcoords[addr0 + 0];
      vertPos[1] = w.vcoords[addr0 + 1];
      vertPos[2] = w.vcoords[addr0 + 2];

      vertPos[3] = w.vcoords[addr1 + 0];
      vertPos[4] = w.vcoords[addr1 + 1];
      vertPos[5] = w.vcoords[addr1 + 2];

      vertPos[6] = w.vcoords[addr2 + 0];
      vertPos[7] = w.vcoords[addr2 + 1];
      vertPos[8] = w.vcoords[addr2 + 2];

      normals[0] = toFloat ( w.normals[j + 0] );
      normals[1] = toFloat ( w.normals[j + 1] );
      normals[2] = toFloat ( w.normals[j + 2] );

      normals[3] = toFloat ( w.normals[j + 0] );
      normals[4] = toFloat ( w.normals[j + 1] );
      normals[5] = toFloat ( w.normals[j + 2] );

      normals[6] = toFloat ( w.normals[j + 0] );
      normals[7] = toFloat ( w.normals[j + 1] );
      normals[8] = toFloat ( w.normals[j + 2] );

      vertPos   += 9;
      normals   += 9;
    }
  }
  else
  {
    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      idx_t        j     = 3 * iperm[i];
      const idx_t  addr0 = 3 * w.triangs[j + 0];
      const idx_t  addr1 = 3 * w.triangs[j + 1];
      const idx_t  addr2 = 3 * w.triangs[j + 2];

      vertPos[0] = w.vcoords[addr0 + 0];
      vertPos[1] = w.vcoords[addr0 + 1];
      vertPos[2] = w.vcoords[addr0 + 2];

      vertPos[3] = w.vcoords[addr1 + 0];
      vertPos[4] = w.vcoords[addr1 + 1];
      vertPos[5] = w.vcoords[addr1 + 2];

      vertPos[6] = w.vcoords[addr2 + 0];
      vertPos[7] = w.vcoords[addr2 + 1];
      vertPos[8] = w.vcoords[addr2 + 2];

      normals[0] = toFloat ( w.normals[3 * j + 0] );
      normals[1] = toFloat ( w.normals[3 * j + 1] );
      normals[2] = toFloat ( w.normals[3 * j + 2] );

      normals[3] = toFloat ( w.normals[3 * j + 3] );
      normals[4] = toFloat ( w.normals[3 * j + 4] );
      normals[5] = toFloat ( w.normals[3 * j + 5] );

      normals[6] = toFloat ( w.normals[3 * j + 6] );
      normals[7] = toFloat ( w.normals[3 * j + 7] );
      normals[8] = toFloat ( w.normals[3 * j + 8] );

      vertPos   += 9;
      normals   += 9;
    }
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    buffer_->drawArrays ( gtx, Buffer_::NORMALS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      vertPos = buffer_->getPositions ();
      normals = buffer_->getNormals   ();

      funcs->glBegin     ( GL_TRIANGLES  );

      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        funcs->glNormal3 ( normals );
        funcs->glVertex3 ( vertPos );
        funcs->glNormal3 ( normals + 3 );
        funcs->glVertex3 ( vertPos + 3 );
        funcs->glNormal3 ( normals + 6 );
        funcs->glVertex3 ( vertPos + 6 );

        vertPos += 9;
        normals += 9;
      }

      funcs->glEnd ();
    }
  }

  endBlend_ ( gtx, zmask );
}


//-----------------------------------------------------------------------
//   blendTrColors3D_
//-----------------------------------------------------------------------


void TriangleSet::blendTrColors3D_ ( GLContext& gtx ) const
{
  Work3D_         w       ( *this );

  const GLubyte*  colors  = trColors_.addr ();
  const idx_t     cst     = trColors_.size (0);
  const idx_t     vcount  =  9 * w.trCount;
  const idx_t     ccount  = 12 * w.trCount;
  GLfloat*        vertPos = buffer_->allocPositions ( vcount );
  GLfloat*        normals = buffer_->allocNormals   ( vcount );
  GLfloat*        vertCol = buffer_->allocColors    ( ccount );
  IdxVector       iperm   = buffer_->allocZOrder ( w.trCount );

  Color           color;
  GLboolean       zmask;


  beginBlend_ ( gtx, zmask );
  w.getZOrder ( gtx, iperm );

  if ( normalMode_ == VERTEX_NORMALS )
  {
    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      idx_t           j       = 3 * iperm[i];
      const idx_t     addr0   = 3 * w.triangs[j + 0];
      const idx_t     addr1   = 3 * w.triangs[j + 1];
      const idx_t     addr2   = 3 * w.triangs[j + 2];
      const GLubyte*  addrCol = colors + (cst * iperm[i]);

      vertPos[ 0] = w.vcoords[addr0 + 0];
      vertPos[ 1] = w.vcoords[addr0 + 1];
      vertPos[ 2] = w.vcoords[addr0 + 2];

      vertPos[ 3] = w.vcoords[addr1 + 0];
      vertPos[ 4] = w.vcoords[addr1 + 1];
      vertPos[ 5] = w.vcoords[addr1 + 2];

      vertPos[ 6] = w.vcoords[addr2 + 0];
      vertPos[ 7] = w.vcoords[addr2 + 1];
      vertPos[ 8] = w.vcoords[addr2 + 2];

      normals[ 0] = toFloat ( w.normals[addr0 + 0] );
      normals[ 1] = toFloat ( w.normals[addr0 + 1] );
      normals[ 2] = toFloat ( w.normals[addr0 + 2] );

      normals[ 3] = toFloat ( w.normals[addr1 + 0] );
      normals[ 4] = toFloat ( w.normals[addr1 + 1] );
      normals[ 5] = toFloat ( w.normals[addr1 + 2] );

      normals[ 6] = toFloat ( w.normals[addr2 + 0] );
      normals[ 7] = toFloat ( w.normals[addr2 + 1] );
      normals[ 8] = toFloat ( w.normals[addr2 + 2] );

      vertCol[ 0] = addrCol[0];
      vertCol[ 1] = addrCol[1];
      vertCol[ 2] = addrCol[2];
      vertCol[ 3] = (cst == 3) ? 1.0F : addrCol[3];

      vertCol[ 4] = addrCol[0];
      vertCol[ 5] = addrCol[1];
      vertCol[ 6] = addrCol[2];
      vertCol[ 7] = (cst == 3) ? 1.0F : addrCol[3];

      vertCol[ 8] = addrCol[0];
      vertCol[ 9] = addrCol[1];
      vertCol[10] = addrCol[2];
      vertCol[11] = (cst == 3) ? 1.0F : addrCol[3];

      vertPos    += 9;
      normals    += 9;
      vertCol    += 12;
    }
  }
  else if ( w.nrCount == w.trCount )
  {
    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      idx_t           j       = 3 * iperm[i];
      const idx_t     addr0   = 3 * w.triangs[j + 0];
      const idx_t     addr1   = 3 * w.triangs[j + 1];
      const idx_t     addr2   = 3 * w.triangs[j + 2];
      const GLubyte*  addrCol = colors + (cst * iperm[i]);

      vertPos[ 0] = w.vcoords[addr0 + 0];
      vertPos[ 1] = w.vcoords[addr0 + 1];
      vertPos[ 2] = w.vcoords[addr0 + 2];

      vertPos[ 3] = w.vcoords[addr1 + 0];
      vertPos[ 4] = w.vcoords[addr1 + 1];
      vertPos[ 5] = w.vcoords[addr1 + 2];

      vertPos[ 6] = w.vcoords[addr2 + 0];
      vertPos[ 7] = w.vcoords[addr2 + 1];
      vertPos[ 8] = w.vcoords[addr2 + 2];

      normals[ 0] = toFloat ( w.normals[j + 0] );
      normals[ 1] = toFloat ( w.normals[j + 1] );
      normals[ 2] = toFloat ( w.normals[j + 2] );

      normals[ 3] = toFloat ( w.normals[j + 0] );
      normals[ 4] = toFloat ( w.normals[j + 1] );
      normals[ 5] = toFloat ( w.normals[j + 2] );

      normals[ 6] = toFloat ( w.normals[j + 0] );
      normals[ 7] = toFloat ( w.normals[j + 1] );
      normals[ 8] = toFloat ( w.normals[j + 2] );

      vertCol[ 0] = addrCol[0];
      vertCol[ 1] = addrCol[1];
      vertCol[ 2] = addrCol[2];
      vertCol[ 3] = (cst == 3) ? 1.0F : addrCol[3];

      vertCol[ 4] = addrCol[0];
      vertCol[ 5] = addrCol[1];
      vertCol[ 6] = addrCol[2];
      vertCol[ 7] = (cst == 3) ? 1.0F : addrCol[3];

      vertCol[ 8] = addrCol[0];
      vertCol[ 9] = addrCol[1];
      vertCol[10] = addrCol[2];
      vertCol[11] = (cst == 3) ? 1.0F : addrCol[3];

      vertPos    += 9;
      normals    += 9;
      vertCol    += 12;
    }
  }
  else
  {
    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      idx_t           j       = 3 * iperm[i];
      const idx_t     addr0   = 3 * w.triangs[j + 0];
      const idx_t     addr1   = 3 * w.triangs[j + 1];
      const idx_t     addr2   = 3 * w.triangs[j + 2];
      const GLubyte*  addrCol = colors + (cst * iperm[i]);

      vertPos[ 0] = w.vcoords[addr0 + 0];
      vertPos[ 1] = w.vcoords[addr0 + 1];
      vertPos[ 2] = w.vcoords[addr0 + 2];

      vertPos[ 3] = w.vcoords[addr1 + 0];
      vertPos[ 4] = w.vcoords[addr1 + 1];
      vertPos[ 5] = w.vcoords[addr1 + 2];

      vertPos[ 6] = w.vcoords[addr2 + 0];
      vertPos[ 7] = w.vcoords[addr2 + 1];
      vertPos[ 8] = w.vcoords[addr2 + 2];

      normals[ 0] = toFloat ( w.normals[3 * j + 0] );
      normals[ 1] = toFloat ( w.normals[3 * j + 1] );
      normals[ 2] = toFloat ( w.normals[3 * j + 2] );

      normals[ 3] = toFloat ( w.normals[3 * j + 3] );
      normals[ 4] = toFloat ( w.normals[3 * j + 4] );
      normals[ 5] = toFloat ( w.normals[3 * j + 5] );

      normals[ 6] = toFloat ( w.normals[3 * j + 6] );
      normals[ 7] = toFloat ( w.normals[3 * j + 7] );
      normals[ 8] = toFloat ( w.normals[3 * j + 8] );

      vertCol[ 0] = addrCol[0];
      vertCol[ 1] = addrCol[1];
      vertCol[ 2] = addrCol[2];
      vertCol[ 3] = (cst == 3) ? 1.0F : addrCol[3];

      vertCol[ 4] = addrCol[0];
      vertCol[ 5] = addrCol[1];
      vertCol[ 6] = addrCol[2];
      vertCol[ 7] = (cst == 3) ? 1.0F : addrCol[3];

      vertCol[ 8] = addrCol[0];
      vertCol[ 9] = addrCol[1];
      vertCol[10] = addrCol[2];
      vertCol[11] = (cst == 3) ? 1.0F : addrCol[3];

      vertPos    += 9;
      normals    += 9;
      vertCol    += 12;
    }
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    buffer_->drawArrays ( gtx, Buffer_::NORMALS | Buffer_::COLORS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      vertPos = buffer_->getPositions ();
      normals = buffer_->getNormals   ();
      vertCol = buffer_->getColors    ();

      funcs->glBegin  ( GL_TRIANGLES  );

      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        funcs->glColor4  ( vertCol );
        funcs->glNormal3 ( normals );
        funcs->glVertex3 ( vertPos );
        funcs->glNormal3 ( normals + 3 );
        funcs->glVertex3 ( vertPos + 3 );
        funcs->glNormal3 ( normals + 6 );
        funcs->glVertex3 ( vertPos + 6 );

        vertPos += 9;
        normals += 9;
        vertCol += 12;
      }

      funcs->glEnd ();
    }
  }

  endBlend_ ( gtx, zmask );
}


//-----------------------------------------------------------------------
//   blendVxColors3D_
//-----------------------------------------------------------------------


void TriangleSet::blendVxColors3D_ ( GLContext& gtx ) const
{
  Work3D_         w       ( *this );

  const GLfloat*  colors  = vxColors_.addr ();
  const idx_t     vcount  = 9 * w.trCount;
  const idx_t     tcount  = 3 * w.trCount;
  GLfloat*        vertPos = buffer_->allocPositions ( vcount );
  GLfloat*        normals = buffer_->allocNormals   ( vcount );
  GLfloat*        texCrds = buffer_->allocTexCoords ( tcount );
  IdxVector       iperm   = buffer_->allocZOrder ( w.trCount );

  GLboolean       zmask;


  beginBlend_ ( gtx, zmask );
  w.getZOrder ( gtx, iperm );

  if ( normalMode_ == VERTEX_NORMALS )
  {
    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      idx_t        j     = 3 * iperm[i];
      const idx_t  addr0 = 3 * w.triangs[j + 0];
      const idx_t  addr1 = 3 * w.triangs[j + 1];
      const idx_t  addr2 = 3 * w.triangs[j + 2];

      vertPos[0] = w.vcoords[addr0 + 0];
      vertPos[1] = w.vcoords[addr0 + 1];
      vertPos[2] = w.vcoords[addr0 + 2];

      vertPos[3] = w.vcoords[addr1 + 0];
      vertPos[4] = w.vcoords[addr1 + 1];
      vertPos[5] = w.vcoords[addr1 + 2];

      vertPos[6] = w.vcoords[addr2 + 0];
      vertPos[7] = w.vcoords[addr2 + 1];
      vertPos[8] = w.vcoords[addr2 + 2];

      normals[0] = toFloat ( w.normals[addr0 + 0] );
      normals[1] = toFloat ( w.normals[addr0 + 1] );
      normals[2] = toFloat ( w.normals[addr0 + 2] );

      normals[3] = toFloat ( w.normals[addr1 + 0] );
      normals[4] = toFloat ( w.normals[addr1 + 1] );
      normals[5] = toFloat ( w.normals[addr1 + 2] );

      normals[6] = toFloat ( w.normals[addr2 + 0] );
      normals[7] = toFloat ( w.normals[addr2 + 1] );
      normals[8] = toFloat ( w.normals[addr2 + 2] );

      texCrds[0] = colors[w.triangs[j + 0]];
      texCrds[1] = colors[w.triangs[j + 1]];
      texCrds[2] = colors[w.triangs[j + 2]];

      vertPos   += 9;
      normals   += 9;
      texCrds   += 9;
    }
  }
  else if ( w.nrCount == w.trCount )
  {
    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      idx_t        j     = 3 * iperm[i];
      const idx_t  addr0 = 3 * w.triangs[j + 0];
      const idx_t  addr1 = 3 * w.triangs[j + 1];
      const idx_t  addr2 = 3 * w.triangs[j + 2];

      vertPos[0] = w.vcoords[addr0 + 0];
      vertPos[1] = w.vcoords[addr0 + 1];
      vertPos[2] = w.vcoords[addr0 + 2];

      vertPos[3] = w.vcoords[addr1 + 0];
      vertPos[4] = w.vcoords[addr1 + 1];
      vertPos[5] = w.vcoords[addr1 + 2];

      vertPos[6] = w.vcoords[addr2 + 0];
      vertPos[7] = w.vcoords[addr2 + 1];
      vertPos[8] = w.vcoords[addr2 + 2];

      normals[0] = toFloat ( w.normals[j + 0] );
      normals[1] = toFloat ( w.normals[j + 1] );
      normals[2] = toFloat ( w.normals[j + 2] );

      normals[3] = toFloat ( w.normals[j + 0] );
      normals[4] = toFloat ( w.normals[j + 1] );
      normals[5] = toFloat ( w.normals[j + 2] );

      normals[6] = toFloat ( w.normals[j + 0] );
      normals[7] = toFloat ( w.normals[j + 1] );
      normals[8] = toFloat ( w.normals[j + 2] );

      texCrds[0] = colors[w.triangs[j + 0]];
      texCrds[1] = colors[w.triangs[j + 1]];
      texCrds[2] = colors[w.triangs[j + 2]];

      vertPos   += 9;
      normals   += 9;
      texCrds   += 9;
    }
  }
  else
  {
    for ( idx_t i = 0; i < w.trCount; i++ )
    {
      idx_t        j     = 3 * iperm[i];
      const idx_t  addr0 = 3 * w.triangs[j + 0];
      const idx_t  addr1 = 3 * w.triangs[j + 1];
      const idx_t  addr2 = 3 * w.triangs[j + 2];

      vertPos[0] = w.vcoords[addr0 + 0];
      vertPos[1] = w.vcoords[addr0 + 1];
      vertPos[2] = w.vcoords[addr0 + 2];

      vertPos[3] = w.vcoords[addr1 + 0];
      vertPos[4] = w.vcoords[addr1 + 1];
      vertPos[5] = w.vcoords[addr1 + 2];

      vertPos[6] = w.vcoords[addr2 + 0];
      vertPos[7] = w.vcoords[addr2 + 1];
      vertPos[8] = w.vcoords[addr2 + 2];

      normals[0] = toFloat ( w.normals[3 * j + 0] );
      normals[1] = toFloat ( w.normals[3 * j + 1] );
      normals[2] = toFloat ( w.normals[3 * j + 2] );

      normals[3] = toFloat ( w.normals[3 * j + 3] );
      normals[4] = toFloat ( w.normals[3 * j + 4] );
      normals[5] = toFloat ( w.normals[3 * j + 5] );

      normals[6] = toFloat ( w.normals[3 * j + 6] );
      normals[7] = toFloat ( w.normals[3 * j + 7] );
      normals[8] = toFloat ( w.normals[3 * j + 8] );

      texCrds[0] = colors[w.triangs[j + 0]];
      texCrds[1] = colors[w.triangs[j + 1]];
      texCrds[2] = colors[w.triangs[j + 2]];

      vertPos   += 9;
      normals   += 9;
      texCrds   += 9;
    }
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    buffer_->drawArrays ( gtx, colorMap_, Buffer_::NORMALS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      vertPos = buffer_->getPositions ();
      normals = buffer_->getNormals   ();
      texCrds = buffer_->getTexCoords ();

      funcs->glBegin  ( GL_TRIANGLES  );

      for ( idx_t i = 0; i < w.trCount; i++ )
      {
        funcs->glTexCoord ( texCrds[0] );
        funcs->glNormal3  ( normals );
        funcs->glVertex3  ( vertPos );
        funcs->glTexCoord ( texCrds[1] );
        funcs->glNormal3  ( normals + 3 );
        funcs->glVertex3  ( vertPos + 3 );
        funcs->glTexCoord ( texCrds[2] );
        funcs->glNormal3  ( normals + 6 );
        funcs->glVertex3  ( vertPos + 6 );

        vertPos += 9;
        normals += 9;
        texCrds += 9;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   plotNoColors2D_
//-----------------------------------------------------------------------


void TriangleSet::plotNoColors2D_

  ( GLContext&  gtx,
    double      lod ) const

{
  using std::ceil;

  Work2D_        w       ( *this );

  const double   st      = 1.0 / lod;
  const idx_t*   tr      = w.triangs;
  const idx_t    size    = (idx_t) ceil ( (double) w.trCount / st );
  const idx_t    vcount  = 3 * size;
  GLfloat*       vertPos = buffer_->allocPositions ( vcount );

  GLfloat        c[2];


  for ( idx_t i = 0, j = 0; j < w.trCount;
        i++, j = (idx_t) (st * (double) i) )
  {
    w.triangs  = tr + (3 * j);

    w.getTrCenter ( c );

    vertPos[0] = c[0];
    vertPos[1] = c[1];
    vertPos[2] = 0.0F;

    vertPos   += 3;
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, vcount,
                       buffer_->getPositions (),
                       gtx     .getColor     (),
                       ShaderDraw::POINTS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      vertPos = buffer_->getPositions ();

      funcs->glBegin  ( GL_POINTS );
      funcs->glNormal ( 0.0, 0.0, 1.0 );

      for ( idx_t i = 0; i < size; i++ )
      {
        funcs->glVertex2 ( vertPos );

        vertPos += 3;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   plotNoColors3D_
//-----------------------------------------------------------------------


void TriangleSet::plotNoColors3D_

( GLContext&  gtx,
  double      lod ) const

{
  using std::ceil;

  Work3D_        w       ( *this );

  const double   st      = 1.0 / lod;
  const idx_t*   tr      = w.triangs;
  const idx_t    size    = (idx_t) ceil ( (double) w.trCount / st );
  const idx_t    vcount  = 3 * size;
  GLfloat*       vertPos = buffer_->allocPositions ( vcount );
  GLfloat*       normals = buffer_->allocNormals   ( vcount );

  GLfloat        c[3];


  if ( normalMode_ == VERTEX_NORMALS )
  {
    for ( idx_t i = 0, j = 0; j < w.trCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      w.triangs = tr + (3 * j);

      w.getTrCenter ( c );

      vertPos[0] = c[0];
      vertPos[1] = c[1];
      vertPos[2] = c[2];

      normals[0] = toFloat ( w.normals[3 * w.triangs[j] + 0] );
      normals[1] = toFloat ( w.normals[3 * w.triangs[j] + 1] );
      normals[2] = toFloat ( w.normals[3 * w.triangs[j] + 2] );

      vertPos   += 3;
      normals   += 3;
    }
  }
  else if ( w.nrCount == w.trCount )
  {
    for ( idx_t i = 0, j = 0; j < w.trCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      w.triangs = tr + (3 * j);

      w.getTrCenter ( c );

      vertPos[0] = c[0];
      vertPos[1] = c[1];
      vertPos[2] = c[2];

      normals[0] = toFloat ( w.normals[3 * j + 0] );
      normals[1] = toFloat ( w.normals[3 * j + 1] );
      normals[2] = toFloat ( w.normals[3 * j + 2] );

      vertPos   += 3;
      normals   += 3;
    }
  }
  else
  {
    for ( idx_t i = 0, j = 0; j < w.trCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      w.triangs = tr + (3 * j);

      w.getTrCenter ( c );

      vertPos[0] = c[0];
      vertPos[1] = c[1];
      vertPos[2] = c[2];

      normals[0] = toFloat ( w.normals[9 * j + 0] );
      normals[1] = toFloat ( w.normals[9 * j + 1] );
      normals[2] = toFloat ( w.normals[9 * j + 2] );

      vertPos   += 3;
      normals   += 3;
    }
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, vcount,
                       buffer_->getPositions(),
                       buffer_->getNormals  (),
                       gtx.getColor         (),
                       ShaderDraw::POINTS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      vertPos = buffer_->getPositions ();
      normals = buffer_->getNormals   ();

      funcs->glBegin ( GL_POINTS );

      for ( idx_t i = 0; i < size; i++ )
      {
        funcs->glNormal3 ( normals );
        funcs->glVertex3 ( vertPos );

        vertPos += 3;
        normals += 3;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   plotTrColors2D_
//-----------------------------------------------------------------------


void TriangleSet::plotTrColors2D_

  ( GLContext&  gtx,
    double      lod ) const

{
  using std::ceil;

  Work2D_         w       ( *this );

  const GLubyte*  colors  = trColors_.addr ();
  const idx_t     cst     = trColors_.size (0);
  const double    st      = 1.0 / lod;
  const idx_t*    tr      = w.triangs;
  const idx_t     size    = (idx_t) ceil ( (double) w.trCount / st );
  const idx_t     vcount  = 3 * size;
  const idx_t     ccount  = 4 * size;
  GLfloat*        vertPos = buffer_->allocPositions ( vcount );
  GLfloat*        vertCol = buffer_->allocColors    ( ccount );

  GLfloat         c[2];
  Color           color;


  for ( idx_t i = 0, j = 0; j < w.trCount;
        i++, j = (idx_t) (st * (double) i) )
  {
    w.triangs = tr + (3 * j);

    w.getTrCenter ( c );

    vertPos[0] = c[0];
    vertPos[1] = c[1];
    vertPos[2] = 0.0F;

    vertCol[0] = colors[cst*j + 0];
    vertCol[1] = colors[cst*j + 1];
    vertCol[2] = colors[cst*j + 2];
    vertCol[3] = (cst == 3) ? 1.0F : colors[cst*j + 3];

    vertPos   += 3;
    vertCol   += 4;
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, vcount,
                       buffer_->getPositions(),
                       buffer_->getColors   (),
                       ShaderDraw::POINTS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      vertPos = buffer_->getPositions ();
      vertCol = buffer_->getColors    ();

      funcs->glBegin  ( GL_POINTS );
      funcs->glNormal ( 0.0, 0.0, 1.0 );

      for ( idx_t i = 0; i < size; i++ )
      {
        funcs->glColor4  ( vertCol );
        funcs->glVertex2 ( vertPos );

        vertPos += 3;
        vertCol += 4;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   plotTrColors3D_
//-----------------------------------------------------------------------


void TriangleSet::plotTrColors3D_

  ( GLContext&  gtx,
    double      lod ) const

{
  using std::ceil;

  Work3D_         w       ( *this );

  const GLubyte*  colors  = trColors_.addr ();
  const idx_t     cst     = trColors_.size (0);
  const double    st      = 1.0 / lod;
  const idx_t*    tr      = w.triangs;
  const idx_t     size    = (idx_t) ceil ( (double) w.trCount / st );
  const idx_t     vcount  = 3 * size;
  const idx_t     ccount  = 4 * size;
  GLfloat*        vertPos = buffer_->allocPositions ( vcount );
  GLfloat*        normals = buffer_->allocNormals   ( vcount );
  GLfloat*        vertCol = buffer_->allocColors    ( ccount );

  GLfloat         c[3];
  Color           color;


  if ( normalMode_ == VERTEX_NORMALS )
  {
    for ( idx_t i = 0, j = 0; j < w.trCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      w.triangs = tr + (3 * j);

      w.getTrCenter ( c );

      vertPos[0] = c[0];
      vertPos[1] = c[1];
      vertPos[2] = c[2];

      normals[0] = toFloat ( w.normals[3 * w.triangs[j] + 0] );
      normals[1] = toFloat ( w.normals[3 * w.triangs[j] + 1] );
      normals[2] = toFloat ( w.normals[3 * w.triangs[j] + 2] );

      vertCol[0] = colors[cst*j + 0];
      vertCol[1] = colors[cst*j + 1];
      vertCol[2] = colors[cst*j + 2];
      vertCol[3] = (cst == 3) ? 1.0F : colors[cst * j + 3];

      vertPos   += 3;
      normals   += 3;
      vertCol   += 4;
    }
  }
  else if ( w.nrCount == w.trCount )
  {
    for ( idx_t i = 0, j = 0; j < w.trCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      w.triangs = tr + (3 * j);

      w.getTrCenter ( c );

      vertPos[0] = c[0];
      vertPos[1] = c[1];
      vertPos[2] = c[2];

      normals[0] = toFloat ( w.normals[3 * j + 0] );
      normals[1] = toFloat ( w.normals[3 * j + 1] );
      normals[2] = toFloat ( w.normals[3 * j + 2] );

      vertCol[0] = colors[cst*j + 0];
      vertCol[1] = colors[cst*j + 1];
      vertCol[2] = colors[cst*j + 2];
      vertCol[3] = (cst == 3) ? 1.0F : colors[cst * j + 3];

      vertPos   += 3;
      normals   += 3;
      vertCol   += 4;
    }
  }
  else
  {
    for ( idx_t i = 0, j = 0; j < w.trCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      w.triangs = tr + (3 * j);

      w.getTrCenter ( c );

      vertPos[0] = c[0];
      vertPos[1] = c[1];
      vertPos[2] = c[2];

      normals[0] = toFloat ( w.normals[9 * j + 0] );
      normals[1] = toFloat ( w.normals[9 * j + 1] );
      normals[2] = toFloat ( w.normals[9 * j + 2] );

      vertCol[0] = colors[cst*j + 0];
      vertCol[1] = colors[cst*j + 1];
      vertCol[2] = colors[cst*j + 2];
      vertCol[3] = (cst == 3) ? 1.0F : colors[cst * j + 3];

      vertPos   += 3;
      normals   += 3;
      vertCol   += 4;
    }
  }


  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, vcount,
                       buffer_->getPositions(),
                       buffer_->getNormals  (),
                       buffer_->getColors   (),
                       ShaderDraw::POINTS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      vertPos = buffer_->getPositions ();
      normals = buffer_->getNormals   ();
      vertCol = buffer_->getColors    ();

      funcs->glBegin ( GL_POINTS );

      for ( idx_t i = 0; i < size; i++ )
      {
        funcs->glColor4  ( vertCol );
        funcs->glNormal3 ( normals );
        funcs->glVertex3 ( vertPos );

        vertPos += 3;
        normals += 3;
        vertCol += 4;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   plotVxColors2D_
//-----------------------------------------------------------------------


void TriangleSet::plotVxColors2D_

  ( GLContext&  gtx,
    double      lod ) const

{
  using std::ceil;

  Work2D_         w      ( *this );

  const GLfloat*  colors = vxColors_.addr ();
  const double    st     = 1.0 / lod;
  const idx_t*    tr     = w.triangs;
  const idx_t     size   = (idx_t) ceil ( (double) w.trCount / st );
  const idx_t     vcount  = 3 * size;
  const idx_t     tcount  =     size;
  GLfloat*        vertPos = buffer_->allocPositions ( vcount );
  GLfloat*        texCrds = buffer_->allocTexCoords ( tcount );

  GLfloat         c[2];


  for ( idx_t i = 0, j = 0; j < w.trCount;
        i++, j = (idx_t) (st * (double) i) )
  {
    w.triangs = tr + (3 * j);

    w.getTrCenter ( c );

    vertPos[0] = c[0];
    vertPos[1] = c[1];
    vertPos[2] = 0.0F;
    texCrds[0] = colors[w.triangs[0]];

    vertPos   += 3;
    texCrds   += 1;
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::drawTex1D ( gtx, vcount,
                            buffer_->getPositions(),
                            buffer_->getTexCoords(),
                            colorMap_->getTexID  (),
                            ShaderDraw::POINTS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      vertPos = buffer_->getPositions ();
      texCrds = buffer_->getTexCoords ();

      funcs->glBegin  ( GL_POINTS );
      funcs->glNormal ( 0.0, 0.0, 1.0 );

      for ( idx_t i = 0; i < size; i++ )
      {
        funcs->glTexCoord ( texCrds[0] );
        funcs->glVertex2  ( vertPos );

        vertPos += 3;
        texCrds += 1;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   plotVxColors3D_
//-----------------------------------------------------------------------


void TriangleSet::plotVxColors3D_

  ( GLContext&  gtx,
    double      lod ) const

{
  using std::ceil;

  Work3D_         w       ( *this );

  const GLfloat*  colors  = vxColors_.addr ();
  const double    st      = 1.0 / lod;
  const idx_t*    tr      = w.triangs;
  const idx_t     size    = (idx_t) ceil ( (double) w.trCount / st );
  const idx_t     vcount  = 3 * size;
  const idx_t     tcount  =     size;
  GLfloat*        vertPos = buffer_->allocPositions ( vcount );
  GLfloat*        normals = buffer_->allocNormals   ( vcount );
  GLfloat*        texCrds = buffer_->allocTexCoords ( tcount );

  GLfloat         c[3];

  if ( normalMode_ == VERTEX_NORMALS )
  {
    for ( idx_t i = 0, j = 0; j < w.trCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      w.triangs = tr + (3 * j);

      w.getTrCenter ( c );

      vertPos[0] = c[0];
      vertPos[1] = c[1];
      vertPos[2] = c[2];

      normals[0] = toFloat ( w.normals[3 * w.triangs[j] + 0] );
      normals[1] = toFloat ( w.normals[3 * w.triangs[j] + 1] );
      normals[2] = toFloat ( w.normals[3 * w.triangs[j] + 2] );

      texCrds[0] = colors[w.triangs[0]];

      vertPos   += 3;
      normals   += 3;
      texCrds   += 1;
    }
  }
  else if ( w.nrCount == w.trCount )
  {
    for ( idx_t i = 0, j = 0; j < w.trCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      w.triangs = tr + (3 * j);

      w.getTrCenter ( c );

      vertPos[0] = c[0];
      vertPos[1] = c[1];
      vertPos[2] = c[2];

      normals[0] = toFloat ( w.normals[3 * j + 0] );
      normals[1] = toFloat ( w.normals[3 * j + 1] );
      normals[2] = toFloat ( w.normals[3 * j + 2] );

      texCrds[0] = colors[w.triangs[0]];

      vertPos   += 3;
      normals   += 3;
      texCrds   += 1;
    }
  }
  else
  {
    for ( idx_t i = 0, j = 0; j < w.trCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      w.triangs = tr + (3 * j);

      w.getTrCenter ( c );

      vertPos[0] = c[0];
      vertPos[1] = c[1];
      vertPos[2] = c[2];

      normals[0] = toFloat ( w.normals[9 * j + 0] );
      normals[1] = toFloat ( w.normals[9 * j + 1] );
      normals[2] = toFloat ( w.normals[9 * j + 2] );

      texCrds[0] = colors[w.triangs[0]];

      vertPos   += 3;
      normals   += 3;
      texCrds   += 1;
    }
  }

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::drawTex1D ( gtx, vcount,
                            buffer_->getPositions(),
                            buffer_->getNormals  (),
                            buffer_->getTexCoords(),
                            colorMap_->getTexID  (),
                            ShaderDraw::POINTS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      vertPos = buffer_->getPositions ();
      normals = buffer_->getNormals   ();
      texCrds = buffer_->getTexCoords ();

      funcs->glBegin ( GL_POINTS );

      for ( idx_t i = 0; i < size; i++ )
      {
        funcs->glTexCoord ( texCrds[0] );
        funcs->glNormal3  ( normals );
        funcs->glVertex3  ( vertPos );

        vertPos += 3;
        normals += 3;
        texCrds += 1;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   beginBlend_
//-----------------------------------------------------------------------


inline void TriangleSet::beginBlend_

  ( GLContext&  gtx,
    GLboolean&  zmask ) const

{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

  if ( funcs )
  {
    funcs->glGetParameter ( GL_DEPTH_WRITEMASK, & zmask );

    if ( zmask == GL_TRUE )
    {
      funcs->glDepthMask ( GL_FALSE );
    }
  }
}


//-----------------------------------------------------------------------
//   endBlend_
//-----------------------------------------------------------------------


void TriangleSet::endBlend_

  ( GLContext&  gtx,
    GLboolean   zmask ) const

{
  if ( zmask == GL_TRUE )
  {
    GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

    if ( funcs )
    {
      funcs->glDepthMask ( zmask );
    }

    gtx.setColor    ( Color( 0.0F, 0.0F, 0.0F, 0.0F ) );
    gtx.setLighting ( false );
    drawNoColors3D_ ( gtx );
  }
}


//-----------------------------------------------------------------------
//   findTriangle2D_
//-----------------------------------------------------------------------


void TriangleSet::findTriangle2D_

  ( idx_t&         index,
    GLfloat&       dist,
    const GLfloat  pos[3] ) const

{
  Work2D_   w ( *this );

  GLdouble  minDist2;
  GLdouble  dist2;


  minDist2 = 0.0;

  for ( idx_t i = 0; i < w.trCount; i++ )
  {
    dist2      = w.distanceTo ( pos );
    w.triangs += 3;

    if ( dist2 < 0.0 )
    {
      continue;
    }

    if ( index < 0 || dist2 < minDist2 )
    {
      minDist2 = dist2;
      index    = i;
    }
  }

  if ( index >= 0 )
  {
    minDist2 += pos[2] * pos[2];
    dist      = (GLfloat) std::sqrt ( minDist2 );
  }
}


//-----------------------------------------------------------------------
//   findTriangle3D_
//-----------------------------------------------------------------------


void TriangleSet::findTriangle3D_

  ( idx_t&         index,
    GLfloat&       dist,
    const GLfloat  pos[3] ) const

{
  Work3D_   w ( *this );

  GLdouble  minDist2;
  GLdouble  dist2;


  minDist2 = 0.0;

  for ( idx_t i = 0; i < w.trCount; i++ )
  {
    dist2      = w.distanceTo ( pos );
    w.triangs += 3;

    if ( dist2 < 0.0 )
    {
      continue;
    }

    if ( index < 0 || dist2 < minDist2 )
    {
      minDist2 = dist2;
      index    = i;
    }
  }

  if ( index >= 0 )
  {
    dist = (GLfloat) std::sqrt ( minDist2 );
  }
}


//-----------------------------------------------------------------------
//   checkData_
//-----------------------------------------------------------------------


void TriangleSet::checkData_ ( const String& context )
{
  const idx_t  rank    = vertices_->rank ();
  const idx_t  vxCount = vertices_->size ();
  const idx_t  trCount = triangles_.size (1);


  if ( vxCount > 0 && (rank < 2 || rank > 3) )
  {
    shapeError ( context, "vertex array", vertices_->shape() );
  }

  if ( trCount > 0 )
  {
    idx_t  imin = jem::min ( triangles_ );
    idx_t  imax = jem::max ( triangles_ );

    if ( imin < 0 )
    {
      indexError ( context, "vertex", imin );
    }
    if ( imax >= vxCount )
    {
      indexError ( context, "vertex", imax, vxCount );
    }
  }

  if ( colorMode_       == VERTEX_COLORS  &&
       vxColors_.size() != vxCount )
  {
    sizeError ( context,
                "vertex color vector", vxColors_.size() );
  }

  if ( normalMode_      == VERTEX_NORMALS &&
       normals_.size(1) != vxCount )
  {
    shapeError ( context,
                 "vertex normal matrix", normals_.shape() );
  }

  checked_ = true;
}


//-----------------------------------------------------------------------
//   updateNormals_
//-----------------------------------------------------------------------


void TriangleSet::updateNormals_ ()
{
  using jem::io::Writer;

  if ( vertices_->rank() == 3 )
  {
    GLfloatMatrix   vertices = vertices_->getCoords ();

    const idx_t     trCount  = triangles_.size (1);
    const idx_t*    triangs  = triangles_.addr ();
    const GLfloat*  vcoords  = vertices  .addr ();

    GLbyte*         normals;
    Vertex3d        a, b, c;
    idx_t           errCount;


    normals_.resize ( 3, trCount );

    errCount = 0;
    normals  = normals_.addr ();

    for ( idx_t i = 0; i < trCount; i++ )
    {
      idx_t  k = 3 * triangs[0];

      c[0] = vcoords[k + 0];
      c[1] = vcoords[k + 1];
      c[2] = vcoords[k + 2];

      k    = 3 * triangs[1];

      a[0] = vcoords[k + 0] - c[0];
      a[1] = vcoords[k + 1] - c[1];
      a[2] = vcoords[k + 2] - c[2];

      k    = 3 * triangs[2];

      b[0] = vcoords[k + 0] - c[0];
      b[1] = vcoords[k + 1] - c[1];
      b[2] = vcoords[k + 2] - c[2];

      if ( isTiny( getNormal( c, a, b ) ) )
      {
        errCount++;

        c[0] = 0.0;
        c[1] = 0.0;
        c[2] = 1.0;
      }

      normals[0] = toByte ( c[0] );
      normals[1] = toByte ( c[1] );
      normals[2] = toByte ( c[2] );

      triangs   += 3;
      normals   += 3;
    }

    if ( errCount )
    {
      Writer&  warn = System::warn ();

      print ( warn, myName_ );

      if ( errCount == 1 )
      {
        print ( warn, " : singular triangle (ignored)\n\n" );
      }
      else
      {
        print ( warn, " : ", errCount,
                " singular triangles (ignored)\n\n" );
      }
    }
  }
  else
  {
    normals_.ref ( GLbyteMatrix() );
  }

  updated_ = true;
}


//-----------------------------------------------------------------------
//   connectHandlers_
//-----------------------------------------------------------------------


void TriangleSet::connectHandlers_ ()
{
  util::connect ( vertices_->newGeomEvent,
                  this, & Self::newGeomHandler_ );

  util::connect ( vertices_->newShapeEvent,
                  this, & Self::newShapeHandler_ );
}


//-----------------------------------------------------------------------
//   newGeomHandler_
//-----------------------------------------------------------------------


void TriangleSet::newGeomHandler_ ()
{
  buffer_->invalidate ();

  if ( normalMode_ == AUTO_NORMALS )
  {
    updated_ = false;
  }
}


//-----------------------------------------------------------------------
//   newShapeHandler_
//-----------------------------------------------------------------------


void TriangleSet::newShapeHandler_ ()
{
  checked_ = false;

  buffer_->invalidate ();

  if ( normalMode_ == AUTO_NORMALS )
  {
    updated_ = false;
  }
}


//-----------------------------------------------------------------------
//   getPointSize_
//-----------------------------------------------------------------------


GLfloat TriangleSet::getPointSize_

  ( GLContext&  gtx,
    double      lod ) const

{
  Rect       vport = gtx.getViewport  ();

  Transform  p     = gtx.getTransform ( GLContext::PROJECTION );
  Transform  t     = gtx.getTransform ();

  Box        box;
  Vertex3d   a, b;
  double     w, h;
  GLfloat    ps;


  vertices_->getBBox ( box );

  box.transform  ( t );
  box.project    ( p );
  box.getCorners ( a, b );

  // Forget it if this triangle set is not visible

  if ( a[0] > 1.0 || b[0] < -1.0 ||
       a[1] > 1.0 || b[1] < -1.0 ||
       a[2] > 1.0 || b[2] < -1.0 )
  {
    return -1.0;
  }

  toWindow ( a, vport );
  toWindow ( b, vport );

  w  = b[0] - a[0];
  h  = b[1] - a[1];

  ps = (GLfloat)
    (1.2 * std::sqrt ( (w * h) / (lod * (double) this->size()) ));

  return clamped ( ps, (GLfloat) 1.0, (GLfloat) 20.0 );
}


//-----------------------------------------------------------------------
//   copyNormal_
//-----------------------------------------------------------------------


void TriangleSet::copyNormal_

  ( GLfloat*       dest,
    const GLbyte*  src )

{
  const GLfloat  byteToFloat = 1.0F / 255.0F;

  dest[0] = byteToFloat * (GLfloat) (src[0]);
  dest[1] = byteToFloat * (GLfloat) (src[1]);
  dest[2] = byteToFloat * (GLfloat) (src[2]);
}


JEM_END_PACKAGE( gl )
