
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/gl/byte.h>
#include <jem/gl/error.h>
#include <jem/gl/render.h>
#include <jem/gl/Menu.h>
#include <jem/gl/Rect.h>
#include <jem/gl/ColorMap.h>
#include <jem/gl/Commands.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions3_3.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/ShaderProg.h>
#include <jem/gl/VertexSet.h>
#include <jem/gl/VertexBufferObject.h>
#include <jem/gl/LineSet.h>
#include "ShaderNames.h"
#include "ShaderSource.h"
#include "VertexArray.h"


JEM_DEFINE_CLASS( jem::gl::LineSet );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class LineSet::Work_
//=======================================================================


class LineSet::Work_
{
 public:

  explicit inline         Work_

    ( const LineSet&        ls );

  void                    getZOrder

    ( GLContext&            gtx,
      const IdxVector&      iperm );


 public:

  const idx_t             segCount;
  const idx_t*            segments;
  const idx_t*            segLines;
  const GLfloat*          vxCoords;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline LineSet::Work_::Work_ ( const LineSet& ls ) :

  segCount ( ls.segments_.size(1) )

{
  segments = ls.segments_.addr     ();
  segLines = ls.segLines_.addr     ();
  vxCoords = ls.vertices_->getData ();
}


//-----------------------------------------------------------------------
//   getZOrder
//-----------------------------------------------------------------------


void LineSet::Work_::getZOrder

  ( GLContext&        gtx,
    const IdxVector&  iperm )
{
  const GLfloat*  vx = vxCoords;

  GLfloatVector   z  ( segCount );

  GLdouble        mx[16];
  GLfloat         t0, t1, t2;

  gtx.getTransform ( mx );

  t0 = (GLfloat) mx[ 2];
  t1 = (GLfloat) mx[ 6];
  t2 = (GLfloat) mx[10];

  for ( idx_t i = 0; i < segCount; i++ )
  {
    idx_t  j = 3 * segments[2 * i];

    z    [i] = t0 * vx[j + 0] + t1 * vx[j + 1] + t2 * vx[j + 2];
    iperm[i] = i;
  }

  sort ( iperm, z );
}


//=======================================================================
//   class LineSet::Buffer_
//=======================================================================


class LineSet::Buffer_ : public Collectable
{
 public:

  static const int        POSITIONS  = 1 << 0;
  static const int        COLORS     = 1 << 1;
  static const int        TEX_COORDS = 1 << 2;


                          Buffer_       ();

  inline bool             isValid       () const noexcept;
  inline void             setValid      ();
  inline void             invalidate    ();

  inline void             drawArrays

    ( GLContext&            gtx,
      int                   what = 0 );

  void                    drawArrays

    ( GLContext&            gtx,
      const Ref<ColorMap>&  cmap,
      int                   what = 0 );

  inline GLfloat*         getPositions  () const;
  inline GLfloat*         getColors     () const;
  inline GLfloat*         getTexCoords  () const;

  inline GLfloat*         allocPositions

    ( idx_t                 size );

  inline GLfloat*         allocColors

    ( idx_t                 size );

  inline GLfloat*         allocTexCoords

    ( idx_t                 size );


 private:

  Ref<VBO>                positions_;
  Ref<VBO>                colors_;
  Ref<VBO>                texCoords_;

  bool                    valid_;
  bool                    synced_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


LineSet::Buffer_::Buffer_ ()
{
  positions_ = newInstance<VBO> ();
  colors_    = newInstance<VBO> ();
  texCoords_ = newInstance<VBO> ();
  valid_     = false;
  synced_    = false;
}


//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


inline bool LineSet::Buffer_::isValid () const noexcept
{
  return valid_;
}


//-----------------------------------------------------------------------
//   setValid
//-----------------------------------------------------------------------


inline void LineSet::Buffer_::setValid ()
{
  valid_ = true;
}


//-----------------------------------------------------------------------
//   invalidate
//-----------------------------------------------------------------------


inline void LineSet::Buffer_::invalidate ()
{
  valid_  = false;
  synced_ = false;
}


//-----------------------------------------------------------------------
//   drawArrays
//-----------------------------------------------------------------------


inline void LineSet::Buffer_::drawArrays

  ( GLContext&  gtx,
    int         what )

{
  drawArrays ( gtx, nullptr, what );
}


void LineSet::Buffer_::drawArrays

  ( GLContext&            gtx,
    const Ref<ColorMap>&  cmap,
    int                   what )

{
  GLFuncs3_3Core&  funcs  = gtx.getFunctions<GLFuncs3_3Core> ();
  ShaderProg*      shader = nullptr;
  GLfloat          alpha  = gtx.getAlphaTest ();
  GLfloat          width  = gtx.getLineWidth ();
  Rect             vport  = gtx.getViewport  ();

  Vertex2f         frame  ( (GLfloat) vport.width,
                            (GLfloat) vport.height );
  VertexArray      vao    ( gtx );

  String           name;
  GLsizei          count;
  GLfloat          mvp[16];


  if ( cmap )
  {
    what |= TEX_COORDS;
  }

  if      ( what & COLORS )
  {
    name = ShaderNames::LINES_MULTI_COLOR;
  }
  else if ( what & TEX_COORDS )
  {
    name = ShaderNames::LINES_TEX1D;
  }
  else
  {
    name = ShaderNames::LINES_CONST_COLOR;
  }

  shader = & ShaderSource::getProgram ( gtx, name );

  if ( ! shader->bind( gtx ) )
  {
    return;
  }

  gtx.getModelViewProj  ( mvp );
  shader->setUniform    ( gtx, ShaderUnis::WIDTH,     width );
  shader->setUniform    ( gtx, ShaderUnis::ALPHA_MIN, alpha );
  shader->setUniform2   ( gtx, ShaderUnis::FRAME_SIZE, frame );
  shader->setUniform4   ( gtx, ShaderUnis::COLOR, gtx.getColor() );
  shader->setUniMatrix4 ( gtx, ShaderUnis::MVP, mvp );

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

  funcs.glDrawArrays ( GL_LINES, 0, count );
  shader->unbind     ( gtx );
}


//-----------------------------------------------------------------------
//   get ...
//-----------------------------------------------------------------------


inline GLfloat* LineSet::Buffer_::getPositions () const
{
  return positions_->getData().addr ();
}


inline GLfloat* LineSet::Buffer_::getColors    () const
{
  return colors_   ->getData().addr ();
}


inline GLfloat* LineSet::Buffer_::getTexCoords () const
{
  return texCoords_->getData().addr ();
}


//-----------------------------------------------------------------------
//   alloc ...
//-----------------------------------------------------------------------


inline GLfloat* LineSet::Buffer_::allocPositions ( idx_t size )
{
  invalidate         ();
  positions_->resize ( size );

  return positions_->getData().addr ();
}


inline GLfloat* LineSet::Buffer_::allocColors ( idx_t size )
{
  invalidate      ();
  colors_->resize ( size );

  return colors_->getData().addr ();
}


inline GLfloat* LineSet::Buffer_::allocTexCoords ( idx_t size )
{
  invalidate         ();
  texCoords_->resize ( size );

  return texCoords_->getData().addr ();
}


//=======================================================================
//   class LineSet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  LineSet::SAMPLE_COUNT = 16 * 1024;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


LineSet::LineSet

  ( const String&          name,
    const Ref<VertexSet>&  vx ) :

    Super      ( name ),
    vertices_  (   vx ),
    lnOffsets_ (    1 )

{
  JEM_PRECHECK ( vx );

  if ( vx->size() > 0 && (vx->rank() < 2 || vx->rank() > 3) )
  {
    shapeError ( JEM_FUNC, "vertex array", vx->shape() );
  }

  lnOffsets_  = 0;
  colorMode_  = NO_COLORS;
  opaque_     = true;
  checked_    = true;
  buffer_     = newInstance<Buffer_> ();

  connectHandlers_ ();
}


LineSet::~LineSet ()
{}


//-----------------------------------------------------------------------
//   getBBox
//-----------------------------------------------------------------------


void LineSet::getBBox ( Box& box ) const
{
  // TO DO: get actual bbox of lines (not all vertices have to
  //        be used).

  vertices_->getBBox ( box );
}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


void LineSet::redraw ( RenderContext& rtx )
{
  if ( this->size() == 0 )
  {
    return;
  }

  GLContext&   gtx     = rtx.getGLContext ();
  const idx_t  rank    = vertices_->rank  ();

  AttribScope  attribs ( gtx, GLContext::COLOR |
                              GLContext::LIGHTING );

  if ( ! checked_ )
  {
    checkData_ ( getContext() );
  }

  switch ( colorMode_ )
  {
  case NO_COLORS:

    if ( isOpaque( gtx.getColor()[3] ) )
    {
      DisableScope  blendScope ( gtx, GL_BLEND );

      if ( rank == 2 )
      {
        drawNoColors2D_  ( gtx );
      }
      else
      {
        drawNoColors3D_  ( gtx );
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

  case LINE_COLORS:

    if ( opaque_ )
    {
      DisableScope  blendScope ( gtx, GL_BLEND );

      if ( rank == 2 )
      {
        drawLnColors2D_  ( gtx );
      }
      else
      {
        drawLnColors3D_  ( gtx );
      }
    }
    else
    {
      if ( rank == 2 )
      {
        drawLnColors2D_  ( gtx );
      }
      else
      {
        blendLnColors3D_ ( gtx );
      }
    }

    break;

  case VERTEX_COLORS:

    colorMap_->enable ( gtx );

    if ( rank == 2 )
    {
      drawVxColors2D_ ( gtx );
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


void LineSet::redrawFast

  ( RenderContext&  rtx,
    double          lod )

{
  if ( lod <= 0.0 )
  {
    return;
  }

  if ( this->size() < SAMPLE_COUNT || lod >= 1.0 )
  {
    Self::redraw ( rtx ); return;
  }

  GLContext&    gtx        = rtx.getGLContext ();
  const idx_t   rank       = vertices_->rank  ();

  DisableScope  blendScope ( gtx, GL_BLEND );
  AttribScope   attribs    ( gtx, GLContext::COLOR    |
                                  GLContext::LIGHTING |
                                  GLContext::POINT_SIZE );

  gtx.setLighting ( false );

  if ( ! checked_ )
  {
    checkData_ ( getContext() );
  }

  gtx.setPointSize ( 2.0F );

  switch ( colorMode_ )
  {
  case NO_COLORS:

    if ( rank == 2 )
    {
      plotNoColors2D_ ( gtx, lod );
    }
    else
    {
      plotNoColors3D_ ( gtx, lod );
    }

    break;

  case LINE_COLORS:

    if ( rank == 2 )
    {
      plotLnColors2D_ ( gtx, lod );
    }
    else
    {
      plotLnColors3D_ ( gtx, lod );
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


bool LineSet::exeCommand

  ( RenderContext&     rtx,
    const String&      cmd,
    const Properties&  params )

{
  if ( cmd == Commands::HIDE_LINES )
  {
    hidden = true;
    return   true;
  }

  if ( cmd == Commands::SHOW_LINES )
  {
    hidden = false;
    return   true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   listOptions
//-----------------------------------------------------------------------


void LineSet::listOptions ( Menu& menu ) const
{
  menu.addOption ( Commands::HIDE_LINES,
                   Commands::SHOW_LINES, hidden );
}


//-----------------------------------------------------------------------
//   setTopology
//-----------------------------------------------------------------------


void LineSet::setTopology

  ( const IdxVector&  offsets,
    const IdxVector&  indices )

{
  JEM_PRECHECK ( offsets.size() > 0 );

  const idx_t  lnCount  = offsets.size() - 1;
  idx_t        segCount = 0;


  buffer_->invalidate ();
  lnOffsets_.ref      ( makeContiguous( offsets ) );
  vxIndices_.ref      ( makeContiguous( indices ) );

  setNoColors ();

  checked_ = false;

  checkData_ ( JEM_FUNC );

  for ( idx_t i = 0; i < lnCount; i++ )
  {
    segCount += (offsets[i + 1] - offsets[i] - 1);
  }

  segLines_.resize ( segCount );

  if ( segCount == lnCount )
  {
    segments_.ref ( jem::reshape( vxIndices_, 2, segCount ) );

    segLines_ = iarray ( segCount );
  }
  else
  {
    segments_.resize ( 2, segCount );

    segCount = 0;

    for ( idx_t i = 0; i < lnCount; i++ )
    {
      const idx_t  n = offsets[i + 1] - 1;

      for ( idx_t j = offsets[i]; j < n; j++, segCount++ )
      {
        segments_(0,segCount) = indices[j + 0];
        segments_(1,segCount) = indices[j + 1];
        segLines_  [segCount] = i;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getLineOffsets
//-----------------------------------------------------------------------


IdxVector LineSet::getLineOffsets () const noexcept
{
  return lnOffsets_;
}


//-----------------------------------------------------------------------
//   getVertexIndices
//-----------------------------------------------------------------------


IdxVector LineSet::getVertexIndices () const noexcept
{
  return vxIndices_;
}


//-----------------------------------------------------------------------
//   setNoColors
//-----------------------------------------------------------------------


void LineSet::setNoColors ()
{
  if ( colorMode_ != NO_COLORS )
  {
    buffer_->invalidate ();
    lnColors_.ref       ( GLubyteMatrix() );
    vxColors_.ref       ( GLfloatVector() );

    opaque_    = true;
    colorMap_  = nullptr;
    colorMode_ = NO_COLORS;
  }
}


//-----------------------------------------------------------------------
//   setLineColors
//-----------------------------------------------------------------------


void LineSet::setLineColors ( const GLubyteMatrix& colors )
{
  JEM_PRECHECK ( colors.size(0) >= 3 &&
                 colors.size(0) <= 4 );

  const idx_t  n     = colors.size (1);

  GLubyte      alpha = toUbyte ( Color::OPAQUE_ALPHA );


  buffer_->invalidate ();

  if ( n != lnOffsets_.size() - 1 )
  {
    shapeError ( JEM_FUNC,
                 "line color matrix", colors.shape() );
  }

  lnColors_.ref ( makeContiguous( colors ) );
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
  colorMode_ = LINE_COLORS;
}


//-----------------------------------------------------------------------
//   setVertexColors
//-----------------------------------------------------------------------


void LineSet::setVertexColors

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
  vxColors_.ref       ( makeContiguous( colors ) );
  lnColors_.ref       ( GLubyteMatrix() );

  colorMap_  = colorMap;
  colorMode_ = VERTEX_COLORS;
}


//-----------------------------------------------------------------------
//   findLine
//-----------------------------------------------------------------------


void LineSet::findLine

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

  const idx_t     rank      = vertices_->rank    ();
  const idx_t     lnCount   = this->size         ();
  const GLfloat*  vxCoords  = vertices_->getData ();
  const idx_t*    lnOffsets = lnOffsets_.addr    ();
  const idx_t*    vxIndices = vxIndices_.addr    ();

  GLdouble        u[3];
  GLdouble        v[3];
  GLdouble        uu, uv, vv;
  GLdouble        d, dmin, t;


  index = -1;
  dist  =  0.0F;
  dmin  =  0.0;

  for ( idx_t i = 0; i < lnCount; i++ )
  {
    const idx_t  n = lnOffsets[i + 1] - 1;

    for ( idx_t j = lnOffsets[i]; j < n; j++ )
    {
      idx_t  jv0 = rank * vxIndices[j + 0];
      idx_t  jv1 = rank * vxIndices[j + 1];

      if ( rank == 2 )
      {
        u[0] = vxCoords[jv0 + 0] - x;
        u[1] = vxCoords[jv0 + 1] - y;

        v[0] = vxCoords[jv1 + 0] - vxCoords[jv0 + 0];
        v[1] = vxCoords[jv1 + 1] - vxCoords[jv0 + 1];

        uu   = u[0] * u[0] + u[1] * u[1];
        uv   = u[0] * v[0] + u[1] * v[1];
        vv   = v[0] * v[0] + v[1] * v[1];
      }
      else
      {
        u[0] = vxCoords[jv0 + 0] - x;
        u[1] = vxCoords[jv0 + 1] - y;
        u[2] = vxCoords[jv0 + 2] - z;

        v[0] = vxCoords[jv1 + 0] - vxCoords[jv0 + 0];
        v[1] = vxCoords[jv1 + 1] - vxCoords[jv0 + 1];
        v[2] = vxCoords[jv1 + 2] - vxCoords[jv0 + 2];

        uu   = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        uv   = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        vv   = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      }

      if ( isTiny( vv ) )
      {
        d = uu;
      }
      else
      {
        t = clamped ( -uv / vv );
        d = uu + 2.0 * t * uv + t * t * vv;
      }

      if ( d >= 0.0 )
      {
        if ( (index < 0) || (d < dmin) )
        {
          index = i;
          dmin  = d;
        }
      }
    }
  }

  if ( index >= 0 )
  {
    dist = (GLfloat) std::sqrt ( dmin );
  }
}


//-----------------------------------------------------------------------
//   drawNoColors2D_
//-----------------------------------------------------------------------


void LineSet::drawNoColors2D_ ( GLContext& gtx ) const
{
  Work_  w ( *this );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    if ( ! buffer_->isValid() )
    {
      const idx_t  vcount  = 6 * w.segCount;
      GLfloat*     vertPos = buffer_->allocPositions ( vcount );

      for ( idx_t i = 0; i < w.segCount; i++ )
      {
        const GLfloat*  coord0 = w.vxCoords + 2 * w.segments[0];
        const GLfloat*  coord1 = w.vxCoords + 2 * w.segments[1];

        vertPos[0]  = coord0[0];
        vertPos[1]  = coord0[1];
        vertPos[2]  = 0.0F;

        vertPos[3]  = coord1[0];
        vertPos[4]  = coord1[1];
        vertPos[5]  = 0.0F;

        vertPos    += 6;
        w.segments += 2;
      }

      buffer_->setValid ();
    }

    buffer_->drawArrays ( gtx );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( idx_t i = w.segCount; i > 0; i-- )
      {
        funcs->glVertex2 ( w.vxCoords + (2 * w.segments[0] ) );
        funcs->glVertex2 ( w.vxCoords + (2 * w.segments[1] ) );

        w.segments += 2;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawNoColors3D_
//-----------------------------------------------------------------------


void LineSet::drawNoColors3D_ ( GLContext& gtx ) const
{
  Work_  w ( *this );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    if ( ! buffer_->isValid() )
    {
      const idx_t  vcount  = 6 * w.segCount;
      GLfloat*     vertPos = buffer_->allocPositions ( vcount );

      for ( idx_t i = 0; i < w.segCount; i++ )
      {
        const GLfloat*  coord0 = w.vxCoords + 3 * w.segments[0];
        const GLfloat*  coord1 = w.vxCoords + 3 * w.segments[1];

        vertPos[0]  = coord0[0];
        vertPos[1]  = coord0[1];
        vertPos[2]  = coord0[2];
        vertPos[3]  = coord1[0];
        vertPos[4]  = coord1[1];
        vertPos[5]  = coord1[2];

        vertPos    += 6;
        w.segments += 2;
      }

      buffer_->setValid ();
    }

    buffer_->drawArrays ( gtx );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( idx_t i = w.segCount; i > 0; i-- )
      {
        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[0] ) );
        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[1] ) );

        w.segments += 2;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawLnColors2D_
//-----------------------------------------------------------------------


void LineSet::drawLnColors2D_ ( GLContext& gtx ) const
{
  const GLubyte*  colors = lnColors_.addr ();
  const idx_t     cst    = lnColors_.size (0);

  Work_           w      ( *this );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    if ( ! buffer_->isValid() )
    {
      const idx_t  vcount   = 6 * w.segCount;
      const idx_t  ccount   = 8 * w.segCount;

      GLfloat*     vertPos  = buffer_->allocPositions ( vcount );
      GLfloat*     vertCols = buffer_->allocColors    ( ccount );

      for ( idx_t i = 0; i < w.segCount; i++ )
      {
        const GLfloat*  coord0 = w.vxCoords + 2 * w.segments[0];
        const GLfloat*  coord1 = w.vxCoords + 2 * w.segments[1];
        const GLubyte*  segcol = colors   + cst * w.segLines[i];

        vertPos [0] = coord0[0];
        vertPos [1] = coord0[1];
        vertPos [2] = 0.0F;
        vertPos [3] = coord1[0];
        vertPos [4] = coord1[1];
        vertPos [5] = 0.0F;

        vertCols[0] = toFloat ( segcol[0] );
        vertCols[1] = toFloat ( segcol[1] );
        vertCols[2] = toFloat ( segcol[2] );
        vertCols[3] = (cst == 3) ? 1.0F :
                      toFloat ( segcol[3] );
        vertCols[4] = toFloat ( segcol[0] );
        vertCols[5] = toFloat ( segcol[1] );
        vertCols[6] = toFloat ( segcol[2] );
        vertCols[7] = (cst == 3) ? 1.0F :
                      toFloat ( segcol[3] );

        vertPos    += 6;
        vertCols   += 8;
        w.segments += 2;
      }

      buffer_->setValid ();
    }

    buffer_->drawArrays ( gtx, Buffer_::COLORS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( idx_t i = w.segCount; i > 0; i-- )
      {
        if ( cst == 3 )
        {
          funcs->glColor3 ( colors + cst * w.segLines[i] );
        }
        else
        {
          funcs->glColor4 ( colors + cst * w.segLines[i] );
        }

        funcs->glVertex2 ( w.vxCoords + (2 * w.segments[0] ) );
        funcs->glVertex2 ( w.vxCoords + (2 * w.segments[1] ) );

        w.segments += 2;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawLnColors3D_
//-----------------------------------------------------------------------


void LineSet::drawLnColors3D_ ( GLContext& gtx ) const
{
  const GLubyte*  colors = lnColors_.addr ();
  const idx_t     cst    = lnColors_.size (0);

  Work_           w      ( *this );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    if ( ! buffer_->isValid() )
    {
      const idx_t  vcount   = 6 * w.segCount;
      const idx_t  ccount   = 8 * w.segCount;

      GLfloat*     vertPos  = buffer_->allocPositions ( vcount );
      GLfloat*     vertCols = buffer_->allocColors    ( ccount );

      for ( idx_t i = 0; i < w.segCount; i++ )
      {
        const GLfloat*  coord0 = w.vxCoords + 3 * w.segments[0];
        const GLfloat*  coord1 = w.vxCoords + 3 * w.segments[1];
        const GLubyte*  segcol = colors   + cst * w.segLines[i];

        vertPos [0] = coord0[0];
        vertPos [1] = coord0[1];
        vertPos [2] = coord0[2];

        vertPos [3] = coord1[0];
        vertPos [4] = coord1[1];
        vertPos [5] = coord1[2];

        vertCols[0] = toFloat ( segcol[0] );
        vertCols[1] = toFloat ( segcol[1] );
        vertCols[2] = toFloat ( segcol[2] );
        vertCols[3] = (cst == 3) ? 1.0F :
                      toFloat ( segcol[3] );

        vertCols[4] = toFloat ( segcol[0] );
        vertCols[5] = toFloat ( segcol[1] );
        vertCols[6] = toFloat ( segcol[2] );
        vertCols[7] = (cst == 3) ? 1.0F :
                      toFloat ( segcol[3] );

        vertPos    += 6;
        vertCols   += 8;
        w.segments += 2;
      }

      buffer_->setValid ();
    }

    buffer_->drawArrays ( gtx, Buffer_::COLORS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( idx_t i = w.segCount; i > 0; i-- )
      {
        if ( cst == 3 )
        {
          funcs->glColor3 ( colors + cst * w.segLines[i] );
        }
        else
        {
          funcs->glColor4 ( colors + cst * w.segLines[i]);
        }

        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[0] ) );
        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[1] ) );

        w.segments += 2;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawVxColors2D_
//-----------------------------------------------------------------------


void LineSet::drawVxColors2D_ ( GLContext& gtx ) const
{
  const GLfloat*  colors = vxColors_.addr ();

  Work_           w ( *this );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    if ( ! buffer_->isValid() )
    {
      const idx_t  vcount  = 6 * w.segCount;
      const idx_t  tcount  = 2 * w.segCount;

      GLfloat*     vertPos = buffer_->allocPositions ( vcount );
      GLfloat*     texCrds = buffer_->allocTexCoords ( tcount );

      for ( idx_t i = 0; i < w.segCount; i++ )
      {
        const GLfloat*  coord0 = w.vxCoords + 2 * w.segments[0];
        const GLfloat*  coord1 = w.vxCoords + 2 * w.segments[1];

        vertPos[0]  = coord0[0];
        vertPos[1]  = coord0[1];
        vertPos[2]  = 0.0F;

        vertPos[3]  = coord1[0];
        vertPos[4]  = coord1[1];
        vertPos[5]  = 0.0F;

        texCrds[0]  = colors[w.segments[0]];
        texCrds[1]  = colors[w.segments[1]];

        vertPos     += 6;
        texCrds     += 2;
        w.segments  += 2;
      }

      buffer_->setValid ();
    }

    buffer_->drawArrays ( gtx, colorMap_ );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( idx_t i = w.segCount; i > 0; i-- )
      {
        idx_t  iv0 = w.segments[0];
        idx_t  iv1 = w.segments[1];

        funcs->glTexCoord ( colors[iv0] );
        funcs->glVertex2  ( w.vxCoords + (2 * iv0) );
        funcs->glTexCoord ( colors[iv1] );
        funcs->glVertex2  ( w.vxCoords + (2 * iv1) );

        w.segments += 2;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   drawVxColors3D_
//-----------------------------------------------------------------------


void LineSet::drawVxColors3D_ ( GLContext& gtx ) const
{
  const GLfloat*  colors = vxColors_.addr ();

  Work_           w ( *this );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    if ( ! buffer_->isValid() )
    {
      const idx_t  vcount  = 6 * w.segCount;
      const idx_t  tcount  = 2 * w.segCount;

      GLfloat*     vertPos = buffer_->allocPositions ( vcount );
      GLfloat*     texCrds = buffer_->allocTexCoords ( tcount );

      for ( idx_t i = 0; i < w.segCount; i++ )
      {
        const GLfloat*  coord0 = w.vxCoords + 3 * w.segments[0];
        const GLfloat*  coord1 = w.vxCoords + 3 * w.segments[1];

        vertPos[0]  = coord0[0];
        vertPos[1]  = coord0[1];
        vertPos[2]  = coord0[2];

        vertPos[3]  = coord1[0];
        vertPos[4]  = coord1[1];
        vertPos[5]  = coord1[2];

        texCrds[0]  = colors[w.segments[0]];
        texCrds[1]  = colors[w.segments[1]];

        vertPos    += 6;
        texCrds    += 2;
        w.segments += 2;
      }

      buffer_->setValid ();
    }

    buffer_->drawArrays ( gtx, colorMap_ );
  }
  else
  {
    GLFunctions1*   funcs  = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( idx_t i = w.segCount; i > 0; i-- )
      {
        idx_t  iv0 = w.segments[0];
        idx_t  iv1 = w.segments[1];

        funcs->glTexCoord ( colors[iv0] );
        funcs->glVertex3  ( w.vxCoords + (3 * iv0) );
        funcs->glTexCoord ( colors[iv1] );
        funcs->glVertex3  ( w.vxCoords + (3 * iv1) );

        w.segments += 2;
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   blendNoColors3D_
//-----------------------------------------------------------------------


void LineSet::blendNoColors3D_ ( GLContext& gtx ) const
{
  Work_      w     ( *this );
  IdxVector  iperm ( w.segCount );

  GLboolean  zmask;


  beginBlend_ ( gtx, zmask );
  w.getZOrder ( gtx, iperm );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    const idx_t  vcount  = 6 * w.segCount;
    GLfloat*     vertPos = buffer_->allocPositions ( vcount );

    for ( idx_t i = 0; i < w.segCount; i++ )
    {
      const idx_t     k      = 2 * iperm[i];
      const GLfloat*  coord0 = w.vxCoords + 3 * w.segments[k + 0];
      const GLfloat*  coord1 = w.vxCoords + 3 * w.segments[k + 1];

      vertPos[0] = coord0[0];
      vertPos[1] = coord0[1];
      vertPos[2] = coord0[2];

      vertPos[3] = coord1[0];
      vertPos[4] = coord1[1];
      vertPos[5] = coord1[2];

      vertPos   += 6;
    }

    buffer_->drawArrays ( gtx );
  }
  else
  {
    GLFunctions1*  funcs  = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( idx_t i = 0; i < w.segCount; i++ )
      {
        idx_t  k = 2 * iperm[i];

        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[k + 0] ) );
        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[k + 1] ) );
      }

      funcs->glEnd ();
    }
  }

  endBlend_ ( gtx, zmask );
}


//-----------------------------------------------------------------------
//   blendLnColors3D_
//-----------------------------------------------------------------------


void LineSet::blendLnColors3D_ ( GLContext& gtx ) const
{
  const GLubyte*  colors = lnColors_.addr ();
  const idx_t     cst    = lnColors_.size (0);

  Work_           w      ( *this );
  IdxVector       iperm  ( w.segCount );

  GLboolean       zmask;


  beginBlend_ ( gtx, zmask );
  w.getZOrder ( gtx, iperm );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    const idx_t  vcount   = 6 * w.segCount;
    const idx_t  ccount   = 8 * w.segCount;

    GLfloat*     vertPos  = buffer_->allocPositions ( vcount );
    GLfloat*     vertCols = buffer_->allocColors    ( ccount );

    for ( idx_t i = 0; i < w.segCount; i++ )
    {
      const idx_t     k     = iperm[i];
      const idx_t     iv0   = w.segments[2 * k + 0];
      const idx_t     iv1   = w.segments[2 * k + 1];

      const GLubyte*  color = colors     + cst * w.segLines[k];
      const GLfloat*  crd0  = w.vxCoords + 3   * iv0;
      const GLfloat*  crd1  = w.vxCoords + 3   * iv1;

      vertPos[0]  = crd0[0];
      vertPos[1]  = crd0[1];
      vertPos[2]  = crd0[2];

      vertPos[3]  = crd1[0];
      vertPos[4]  = crd1[1];
      vertPos[5]  = crd1[2];

      vertCols[0] = toFloat ( color[0] );
      vertCols[1] = toFloat ( color[1] );
      vertCols[2] = toFloat ( color[2] );
      vertCols[3] = cst == 3 ? 1.0F :
                    toFloat ( color[3] );

      vertCols[4] = toFloat ( color[0] );
      vertCols[5] = toFloat ( color[1] );
      vertCols[6] = toFloat ( color[2] );
      vertCols[7] = cst == 3 ? 1.0F :
                    toFloat ( color[3] );

      vertPos    += 6;
      vertCols   += 8;
    }

    buffer_->drawArrays ( gtx, Buffer_::COLORS );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( idx_t i = 0; i < w.segCount; i++ )
      {
        idx_t  k = iperm[i];

        if ( cst == 3 )
        {
          funcs->glColor3 ( colors + cst * w.segLines[k] );
        }
        else
        {
          funcs->glColor4 ( colors + cst * w.segLines[k] );
        }

        k *= 2;

        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[k + 0] ) );
        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[k + 1] ) );
      }

      funcs->glEnd ();
    }
  }

  endBlend_ ( gtx, zmask );
}


//-----------------------------------------------------------------------
//   blendVxColors3D_
//-----------------------------------------------------------------------


void LineSet::blendVxColors3D_ ( GLContext& gtx ) const
{
  const GLfloat*  colors = vxColors_.addr ();

  Work_           w      ( *this );
  IdxVector       iperm  ( w.segCount );

  GLboolean       zmask;


  beginBlend_ ( gtx, zmask );
  w.getZOrder ( gtx, iperm );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    const idx_t  vcount  = 6 * w.segCount;
    const idx_t  tcount  = 2 * w.segCount;

    GLfloat*     vertPos = buffer_->allocPositions ( vcount );
    GLfloat*     texCrds = buffer_->allocTexCoords ( tcount );

    for ( idx_t i = 0; i < w.segCount; i++ )
    {
      const idx_t     k      = 2 * iperm[i];
      const GLfloat*  coord0 = w.vxCoords + 3 * w.segments[k + 0];
      const GLfloat*  coord1 = w.vxCoords + 3 * w.segments[k + 1];

      vertPos[0] = coord0[0];
      vertPos[1] = coord0[1];
      vertPos[2] = coord0[2];

      vertPos[3] = coord1[0];
      vertPos[4] = coord1[1];
      vertPos[5] = coord1[2];

      texCrds[0] = colors[w.segments[k + 0]];
      texCrds[1] = colors[w.segments[k + 1]];

      vertPos   += 6;
      texCrds   += 2;
    }

    buffer_->drawArrays ( gtx, colorMap_ );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( idx_t i = 0; i < w.segCount; i++ )
      {
        idx_t  k   = 2 * iperm[i];
        idx_t  iv0 = w.segments[k + 0];
        idx_t  iv1 = w.segments[k + 1];

        funcs->glTexCoord ( colors[iv0] );
        funcs->glVertex3  ( w.vxCoords + (3 * iv0) );
        funcs->glTexCoord ( colors[iv1] );
        funcs->glVertex3  ( w.vxCoords + (3 * iv1) );
      }

      funcs->glEnd ();
    }
  }

  endBlend_ ( gtx, zmask );
}


//-----------------------------------------------------------------------
//   plotNoColors2D_
//-----------------------------------------------------------------------


void LineSet::plotNoColors2D_

  ( GLContext&  gtx,
    double      lod ) const

{
  const double  st = 1.0 / lod;

  Work_         w  ( *this );


  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    const idx_t  vcount  = 6 * w.segCount;

    GLfloat*     vertPos = buffer_->allocPositions ( vcount );

    for ( idx_t i = 0, j = 0; j < w.segCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      const GLfloat*  coord0 = w.vxCoords + 2 * w.segments[2 * j + 0];
      const GLfloat*  coord1 = w.vxCoords + 2 * w.segments[2 * j + 1];

      vertPos[0] = coord0[0];
      vertPos[1] = coord0[1];
      vertPos[2] = 0.0F;

      vertPos[3] = coord1[0];
      vertPos[4] = coord1[1];
      vertPos[5] = 0.0F;

      vertPos   += 6;
    }

    ShaderDraw::draw ( gtx, vcount,
                       buffer_->getPositions (),
                       gtx     .getColor     (),
                       ShaderDraw::POINTS );
  }
  else
  {
    GLFunctions1*  funcs  = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_POINTS );

      for ( idx_t i = 0, j = 0; j < w.segCount;
            i++, j = (idx_t) (st * (double) i) )
      {
        funcs->glVertex2 ( w.vxCoords + (2 * w.segments[2 * j + 0] ) );
        funcs->glVertex2 ( w.vxCoords + (2 * w.segments[2 * j + 1] ) );
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   plotNoColors3D_
//-----------------------------------------------------------------------


void LineSet::plotNoColors3D_

  ( GLContext&  gtx,
    double      lod ) const

{
  const double  st = 1.0 / lod;

  Work_         w  ( *this );


  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    const idx_t  vcount  = 6 * w.segCount;

    GLfloat*     vertPos = buffer_->allocPositions ( vcount );

    for ( idx_t i = 0, j = 0; j < w.segCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      const GLfloat*  coord0 = w.vxCoords + 3 * w.segments[2 * j + 0];
      const GLfloat*  coord1 = w.vxCoords + 3 * w.segments[2 * j + 1];

      vertPos[0] = coord0[0];
      vertPos[1] = coord0[1];
      vertPos[2] = coord0[2];

      vertPos[3] = coord1[0];
      vertPos[4] = coord1[1];
      vertPos[5] = coord1[2];

      vertPos   += 6;
    }

    ShaderDraw::draw ( gtx, vcount,
                       buffer_->getPositions (),
                       gtx     .getColor     (),
                       ShaderDraw::POINTS );
  }
  else
  {
    GLFunctions1*  funcs  = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_POINTS );

      for ( idx_t i = 0, j = 0; j < w.segCount;
            i++, j = (idx_t) (st * (double) i) )
      {
        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[2 * j + 0] ) );
        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[2 * j + 1] ) );
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   plotLnColors2D_
//-----------------------------------------------------------------------


void LineSet::plotLnColors2D_

  ( GLContext&  gtx,
    double      lod ) const

{
  const GLubyte*  colors = lnColors_.addr ();
  const idx_t     cst    = lnColors_.size (0);

  const double    st     = 1.0 / lod;

  Work_           w      ( *this );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    const idx_t  vcount   = 6 * w.segCount;
    const idx_t  ccount   = 8 * w.segCount;

    GLfloat*     vertPos  = buffer_->allocPositions ( vcount );
    GLfloat*     vertCols = buffer_->allocColors    ( ccount );

    for ( idx_t i = 0, j = 0; j < w.segCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      const GLfloat*  coord0 = w.vxCoords + 2 * w.segments[2 * j + 0];
      const GLfloat*  coord1 = w.vxCoords + 2 * w.segments[2 * j + 1];
      const GLubyte*  color  = colors   + cst * w.segLines[j];

      vertPos[0]  = coord0[0];
      vertPos[1]  = coord0[1];
      vertPos[2]  = 0.0F;

      vertPos[3]  = coord1[0];
      vertPos[4]  = coord1[1];
      vertPos[5]  = 0.0F;

      vertCols[0] = toFloat ( color[0] );
      vertCols[1] = toFloat ( color[1] );
      vertCols[2] = toFloat ( color[2] );
      vertCols[3] = cst == 3 ? 1.0F :
                    toFloat ( color[3] );

      vertCols[4] = toFloat ( color[0] );
      vertCols[5] = toFloat ( color[1] );
      vertCols[6] = toFloat ( color[2] );
      vertCols[7] = cst == 3 ? 1.0F :
                    toFloat ( color[3] );

      vertPos    += 6;
      vertCols   += 8;
    }

    ShaderDraw::draw ( gtx, vcount,
                       buffer_->getPositions(),
                       buffer_->getColors   (),
                       ShaderDraw::POINTS );
  }
  else
  {
    GLFunctions1*   funcs  = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_POINTS );

      for ( idx_t i = 0, j = 0; j < w.segCount;
            i++, j = (idx_t) (st * (double) i) )
      {
        funcs->glColor3  ( colors    + cst * w.segLines[j] );
        funcs->glVertex2 ( w.vxCoords + (2 * w.segments[2 * j + 0] ) );
        funcs->glVertex2 ( w.vxCoords + (2 * w.segments[2 * j + 1] ) );
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   plotLnColors3D_
//-----------------------------------------------------------------------


void LineSet::plotLnColors3D_

  ( GLContext&  gtx,
    double      lod ) const

{
  const GLubyte*  colors = lnColors_.addr ();
  const idx_t     cst    = lnColors_.size (0);

  const double    st     = 1.0 / lod;

  Work_           w ( *this );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    const idx_t  vcount   = 6 * w.segCount;
    const idx_t  ccount   = 8 * w.segCount;

    GLfloat*     vertPos  = buffer_->allocPositions ( vcount );
    GLfloat*     vertCols = buffer_->allocColors    ( ccount );

    for ( idx_t i = 0, j = 0; j < w.segCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      const GLfloat*  coord0 = w.vxCoords + 3 * w.segments[2 * j + 0];
      const GLfloat*  coord1 = w.vxCoords + 3 * w.segments[2 * j + 1];
      const GLubyte*  color  = colors   + cst * w.segLines[j];

      vertPos[0]  = coord0[0];
      vertPos[1]  = coord0[1];
      vertPos[2]  = coord0[2];

      vertPos[3]  = coord1[0];
      vertPos[4]  = coord1[1];
      vertPos[5]  = coord1[2];

      vertCols[0] = toFloat ( color[0] );
      vertCols[1] = toFloat ( color[1] );
      vertCols[2] = toFloat ( color[2] );
      vertCols[3] = cst == 3 ? 1.0F :
                    toFloat ( color[3] );

      vertCols[4] = toFloat ( color[0] );
      vertCols[5] = toFloat ( color[1] );
      vertCols[6] = toFloat ( color[2] );
      vertCols[7] = cst == 3 ? 1.0F :
                    toFloat ( color[3] );

      vertPos    += 6;
      vertCols   += 8;
    }

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
      funcs->glBegin ( GL_POINTS );

      for ( idx_t i = 0, j = 0; j < w.segCount;
            i++, j = (idx_t) (st * (double) i) )
      {
        funcs->glColor3  ( colors    + cst * w.segLines[j] );
        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[2 * j + 0] ) );
        funcs->glVertex3 ( w.vxCoords + (3 * w.segments[2 * j + 1] ) );
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   plotVxColors2D_
//-----------------------------------------------------------------------


void LineSet::plotVxColors2D_

  ( GLContext&  gtx,
    double      lod ) const

{
  const GLfloat*  colors = vxColors_.addr ();

  const double    st     = 1.0 / lod;

  Work_           w      ( *this );


  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    const idx_t  vcount  = 6 * w.segCount;
    const idx_t  tcount  = 2 * w.segCount;

    GLfloat*     vertPos = buffer_->allocPositions ( vcount );
    GLfloat*     texCrds = buffer_->allocTexCoords ( tcount );

    for ( idx_t i = 0, j = 0; j < w.segCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      const GLfloat*  coord0 = w.vxCoords + 2 * w.segments[2 * j + 0];
      const GLfloat*  coord1 = w.vxCoords + 2 * w.segments[2 * j + 1];

      vertPos[0] = coord0[0];
      vertPos[1] = coord0[1];
      vertPos[2] = 0.0F;

      vertPos[3] = coord1[0];
      vertPos[4] = coord1[1];
      vertPos[5] = 0.0F;

      texCrds[0] = colors[w.segments[2 * j + 0]];
      texCrds[1] = colors[w.segments[2 * j + 1]];

      vertPos   += 6;
      texCrds   += 2;
    }

    ShaderDraw::drawTex1D ( gtx, vcount,
                            buffer_->getPositions(),
                            buffer_->getTexCoords(),
                            colorMap_->getTexID  (),
                            ShaderDraw::POINTS );
  }
  else
  {
    GLFunctions1*   funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_POINTS );

      for ( idx_t i = 0, j = 0; j < w.segCount;
            i++, j = (idx_t) (st * (double) i) )
      {
        idx_t  iv0 = w.segments[2 * j + 0];
        idx_t  iv1 = w.segments[2 * j + 1];

        funcs->glTexCoord ( colors[iv0] );
        funcs->glVertex2  ( w.vxCoords + (2 * iv0) );
        funcs->glTexCoord ( colors[iv1] );
        funcs->glVertex2  ( w.vxCoords + (2 * iv1) );
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   plotVxColors3D_
//-----------------------------------------------------------------------


void LineSet::plotVxColors3D_

  ( GLContext&  gtx,
    double      lod ) const

{
  const GLfloat*  colors = vxColors_.addr ();

  const double    st     = 1.0 / lod;

  Work_           w      ( *this );


  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    const idx_t  vcount  = 6 * w.segCount;
    const idx_t  tcount  = 2 * w.segCount;

    GLfloat*     vertPos = buffer_->allocPositions ( vcount );
    GLfloat*     texCrds = buffer_->allocTexCoords ( tcount );

    for ( idx_t i = 0, j = 0; j < w.segCount;
          i++, j = (idx_t) (st * (double) i) )
    {
      const GLfloat*  coord0 = w.vxCoords + 3 * w.segments[2 * j + 0];
      const GLfloat*  coord1 = w.vxCoords + 3 * w.segments[2 * j + 1];

      vertPos[0] = coord0[0];
      vertPos[1] = coord0[1];
      vertPos[2] = coord0[2];

      vertPos[3] = coord1[0];
      vertPos[4] = coord1[1];
      vertPos[5] = coord1[2];

      texCrds[0] = colors[w.segments[2*j + 0]];
      texCrds[1] = colors[w.segments[2*j + 1]];

      vertPos   += 6;
      texCrds   += 2;
    }

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
      funcs->glBegin ( GL_POINTS );

      for ( idx_t i = 0, j = 0; j < w.segCount;
            i++, j = (idx_t) (st * (double) i) )
      {
        idx_t  iv0 = w.segments[2 * j + 0];
        idx_t  iv1 = w.segments[2 * j + 1];

        funcs->glTexCoord ( colors[iv0] );
        funcs->glVertex2  ( w.vxCoords + (3 * iv0) );
        funcs->glTexCoord ( colors[iv1] );
        funcs->glVertex2  ( w.vxCoords + (3 * iv1) );
      }

      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   beginBlend_
//-----------------------------------------------------------------------


inline void LineSet::beginBlend_

  ( GLContext&  gtx,
    GLboolean&  zmask ) const

{
  GLFunctions1*   funcs  = gtx.findFunctions<GLFunctions1> ();

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


void LineSet::endBlend_

  ( GLContext&  gtx,
    GLboolean   zmask ) const

{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

  if ( zmask == GL_TRUE )
  {
    if ( funcs )
    {
      funcs->glDepthMask ( zmask );
      gtx.setColor       ( Color( 0.0F, 0.0F, 0.0F, 0.0F ) );
    }

    gtx.setLighting ( false );
    drawNoColors3D_ ( gtx );
  }
}


//-----------------------------------------------------------------------
//   checkData_
//-----------------------------------------------------------------------


void LineSet::checkData_ ( const String& context )
{
  const idx_t  rank    = vertices_->rank ();
  const idx_t  vxCount = vertices_->size ();
  const idx_t  lnCount = lnOffsets_.size () - 1;
  const idx_t  ixCount = vxIndices_.size ();


  if ( vxCount > 0 && (rank < 2 || rank > 3) )
  {
    shapeError ( context, "vertex array", vertices_->shape() );
  }

  for ( idx_t i = 0; i < lnCount; i++ )
  {
    idx_t  j = lnOffsets_[i];
    idx_t  k = lnOffsets_[i + 1];

    if ( k > j && (j < 0 || j >= ixCount || k > ixCount) )
    {
      throw IllegalArgumentException (
        context,
        "invalid line offset array"
      );
    }
  }

  if ( ixCount > 0 )
  {
    idx_t  imin = min ( vxIndices_ );
    idx_t  imax = max ( vxIndices_ );

    if ( imin < 0 )
    {
      indexError ( context, "vertex", imin );
    }
    if ( imax >= vxCount )
    {
      indexError ( context, "vertex", imax, vxCount );
    }
  }

  if ( colorMode_       == VERTEX_COLORS &&
       vxColors_.size() != vxCount )
  {
    sizeError ( context,
                "vertex color vector", vxColors_.size() );
  }

  checked_ = true;
}


//-----------------------------------------------------------------------
//   connectHandlers_
//-----------------------------------------------------------------------


void LineSet::connectHandlers_ ()
{
  util::connect ( vertices_->newGeomEvent,
                  this, & Self::newGeomHandler_ );

  util::connect ( vertices_->newShapeEvent,
                  this, & Self::newShapeHandler_ );
}


//-----------------------------------------------------------------------
//   newGeomHandler_
//-----------------------------------------------------------------------


void LineSet::newGeomHandler_ ()
{
  buffer_->invalidate ();
}


//-----------------------------------------------------------------------
//   newShapeHandler_
//-----------------------------------------------------------------------


void LineSet::newShapeHandler_ ()
{
  checked_ = false;

  buffer_->invalidate ();
}


JEM_END_PACKAGE( gl )
