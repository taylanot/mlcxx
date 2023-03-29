
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


#include <jem/base/array/utilities.h>
#include <jem/gl/Color.h>
#include <jem/gl/Lights.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLException.h>
#include <jem/gl/GLFunctions3_3.h>
#include <jem/gl/Rect.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/ShaderProg.h>
#include "ShaderNames.h"
#include "ShaderSource.h"
#include "VertexArray.h"
#include "VertexBuffer.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class ShaderDraw
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int    ShaderDraw::COLOR_SIZE_     = 4;
const int    ShaderDraw::NORMAL_SIZE_    = 3;
const int    ShaderDraw::RADIUS_SIZE_    = 1;
const int    ShaderDraw::TEX1CRD_SIZE_   = 1;
const int    ShaderDraw::TEX2CRD_SIZE_   = 2;
const int    ShaderDraw::VERTEX_SIZE_    = 3;
const idx_t  ShaderDraw::MAX_BLOCK_SIZE_ = 100000_idx;


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


ShaderDraw::ShaderDraw

  ( ShaderType_  shaderType,
    idx_t        geomCount,
    Type         geomType ) :

    shaderType_ ( shaderType ),
    geomType_   ( geomType ),
    geomCount_  ( geomCount )

{
  JEM_PRECHECK ( geomCount >= 0 );

  positions_  = nullptr;
  normals_    = nullptr;
  colors_     = nullptr;
  uCoords_    = nullptr;
  uvCoords_   = nullptr;
  radii_      = nullptr;

  tex1ID_     = 0;
  tex2ID_     = 0;

  stipFactor_ = 0;
  stipPttrn_  = 0;
  options_    = 0;

  initType_ ();
}


//-----------------------------------------------------------------------
//   draw -- constant color and no shading
//-----------------------------------------------------------------------


void ShaderDraw::draw

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const Color&          color,
    Type                  type )

{
  draw ( gtx, positions.size(),
         makeContiguous( positions ).addr(),
         color, type );
}


void ShaderDraw::draw

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const Color&    color,
    Type            type )

{
  ShaderDraw  drawer   ( FLAT_COLOR_, count, type );

  drawer.setPositions_ ( positions );
  drawer.setColor_     ( color );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   draw -- constant color with shading
//-----------------------------------------------------------------------


void ShaderDraw::draw

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  normals,
    const Color&          color,
    Type                  type )

{
  JEM_PRECHECK ( positions.size() == normals.size() );

  draw ( gtx, positions.size(),
         makeContiguous( positions ).addr(),
         makeContiguous( normals   ).addr(),
         color, type );
}


void ShaderDraw::draw

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const GLfloat*  normals,
    const Color&    color,
    Type            type )

{
  ShaderDraw  drawer   ( SHADE_COLOR_, count, type );

  drawer.setPositions_ ( positions );
  drawer.setNormals_   ( normals );
  drawer.setColor_     ( color );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   draw -- multiple colors and no shading
//-----------------------------------------------------------------------


void ShaderDraw::draw

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  colors,
    Type                  type )

{
  JEM_PRECHECK ( positions.size() / VERTEX_SIZE_ ==
                 colors   .size() / COLOR_SIZE_ );

  draw ( gtx, positions.size(),
         makeContiguous( positions ).addr(),
         makeContiguous( colors    ).addr(),
         type );
}


void ShaderDraw::draw

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const GLfloat*  colors,
    Type            type )

{
  ShaderDraw  drawer   ( FLAT_COLORS_, count, type );

  drawer.setPositions_ ( positions );
  drawer.setColors_    ( colors );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   draw -- multiple colors and shading
//-----------------------------------------------------------------------


void ShaderDraw::draw

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  normals,
    const GLfloatVector&  colors,
    Type                  type )

{
  JEM_PRECHECK ( (positions.size() == normals.size()) &&
                 (positions.size() / VERTEX_SIZE_ ==
                  colors   .size() / COLOR_SIZE_) );

  draw ( gtx, positions.size(),
         makeContiguous( positions ).addr(),
         makeContiguous( normals   ).addr(),
         makeContiguous( colors    ).addr(),
         type );
}


void ShaderDraw::draw

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const GLfloat*  normals,
    const GLfloat*  colors,
    Type            type )

{
  ShaderDraw  drawer ( SHADE_COLORS_, count, type );

  drawer.setPositions_ ( positions );
  drawer.setNormals_   ( normals );
  drawer.setColors_    ( colors );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawTex1D -- texture 1D and no shading
//-----------------------------------------------------------------------


void ShaderDraw::drawTex1D

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  uCoords,
    GLuint                tex1ID,
    Type                  type )

{
  JEM_PRECHECK ( positions.size() / VERTEX_SIZE_ ==
                 uCoords  .size() / TEX1CRD_SIZE_ );

  drawTex1D ( gtx, positions.size(),
              makeContiguous( positions ).addr(),
              makeContiguous( uCoords   ).addr(),
              tex1ID, type );
}


void ShaderDraw::drawTex1D

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const GLfloat*  uCoords,
    GLuint          tex1ID,
    Type            type )

{
  ShaderDraw  drawer   ( FLAT_TEX1_, count, type );

  drawer.setPositions_ ( positions );
  drawer.set1DTexture_ ( uCoords, tex1ID );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawTex1D -- texture 1D and shading
//-----------------------------------------------------------------------


void ShaderDraw::drawTex1D

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  normals,
    const GLfloatVector&  uCoords,
    GLuint                tex1ID,
    Type                  type )

{
  JEM_PRECHECK ( (positions.size() == normals.size()) &&
                 (positions.size() / VERTEX_SIZE_ ==
                  uCoords  .size() / TEX1CRD_SIZE_) );

  drawTex1D ( gtx, positions.size(),
              makeContiguous( positions ).addr(),
              makeContiguous( normals   ).addr(),
              makeContiguous( uCoords   ).addr(),
              tex1ID, type );
}


void ShaderDraw::drawTex1D

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const GLfloat*  normals,
    const GLfloat*  uCoords,
    GLuint          tex1ID,
    Type            type )

{
  ShaderDraw  drawer   ( SHADE_TEX1_, count, type );

  drawer.setPositions_ ( positions );
  drawer.setNormals_   ( normals );
  drawer.set1DTexture_ ( uCoords, tex1ID );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawTex1D -- texture 1D (alpha only) and no shading
//-----------------------------------------------------------------------


void ShaderDraw::drawTex1D

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  uCoords,
    GLuint                tex1ID,
    const Color&          color,
    Type                  type )

{
  JEM_PRECHECK ( positions.size() / VERTEX_SIZE_ ==
                 uCoords  .size() / TEX1CRD_SIZE_ );

  drawTex1D ( gtx, positions.size(),
              makeContiguous( positions ).addr(),
              makeContiguous( uCoords   ).addr(),
              tex1ID, color, type );
}


void ShaderDraw::drawTex1D

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const GLfloat*  uCoords,
    GLuint          tex1ID,
    const Color&    color,
    Type            type )

{
  ShaderDraw  drawer   ( FLAT_TEX1_COLOR_, count, type );

  drawer.setPositions_ ( positions );
  drawer.set1DTexture_ ( uCoords, tex1ID );
  drawer.setColor_     ( color );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawTex1D -- texture 1D (alpha only) and shading
//-----------------------------------------------------------------------


void ShaderDraw::drawTex1D

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  normals,
    const GLfloatVector&  uCoords,
    GLuint                tex1ID,
    const Color&          color,
    Type                  type )

{
  JEM_PRECHECK ( (positions.size() == normals.size()) &&
                 (positions.size() / VERTEX_SIZE_ ==
                  uCoords  .size() / TEX1CRD_SIZE_) );

  drawTex1D ( gtx, positions.size(),
              makeContiguous( positions ).addr(),
              makeContiguous( normals   ).addr(),
              makeContiguous( uCoords   ).addr(),
              tex1ID, color, type );
}


void ShaderDraw::drawTex1D

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const GLfloat*  normals,
    const GLfloat*  uCoords,
    GLuint          tex1ID,
    const Color&    color,
    Type            type )

{
  ShaderDraw  drawer   ( SHADE_TEX1_COLOR_, count, type );

  drawer.setPositions_ ( positions );
  drawer.setNormals_   ( normals );
  drawer.set1DTexture_ ( uCoords, tex1ID );
  drawer.setColor_     ( color );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawTex2D -- texture 2D and no shading
//-----------------------------------------------------------------------


void ShaderDraw::drawTex2D

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  uvCoords,
    GLuint                tex2ID,
    Type                  type )

{
  JEM_PRECHECK ( positions.size() / VERTEX_SIZE_ ==
                 uvCoords .size() / TEX2CRD_SIZE_ );

  drawTex2D ( gtx, positions.size(),
              makeContiguous( positions ).addr(),
              makeContiguous( uvCoords  ).addr(),
              tex2ID, type );
}


void ShaderDraw::drawTex2D

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const GLfloat*  uvCoords,
    GLuint          tex2ID,
    Type            type )

{
  ShaderDraw  drawer   ( FLAT_TEX2_, count, type );

  drawer.setPositions_ ( positions );
  drawer.set2DTexture_ ( uvCoords, tex2ID );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawTex2D -- texture 2D and shading
//-----------------------------------------------------------------------


void ShaderDraw::drawTex2D

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  normals,
    const GLfloatVector&  uvCoords,
    GLuint                tex2ID,
    Type                  type )

{
  JEM_PRECHECK ( (positions.size() == normals.size()) &&
                 (positions.size() / VERTEX_SIZE_ ==
                  uvCoords .size() / TEX2CRD_SIZE_) );

  drawTex2D ( gtx, positions.size(),
              makeContiguous( positions ).addr(),
              makeContiguous( normals   ).addr(),
              makeContiguous( uvCoords  ).addr(),
              tex2ID, type );
}


void ShaderDraw::drawTex2D

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const GLfloat*  normals,
    const GLfloat*  uvCoords,
    GLuint          tex2ID,
    Type            type )

{
  ShaderDraw  drawer   ( SHADE_TEX2_, count, type );

  drawer.setPositions_ ( positions );
  drawer.setNormals_   ( normals );
  drawer.set2DTexture_ ( uvCoords, tex2ID );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawTex2D -- texture 2D (alpha only) and no shading
//-----------------------------------------------------------------------


void ShaderDraw::drawTex2D

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  uvCoords,
    GLuint                tex2ID,
    const Color&          color,
    Type                  type )

{
  JEM_PRECHECK ( positions.size() / VERTEX_SIZE_ ==
                 uvCoords .size() / TEX2CRD_SIZE_ );

  drawTex2D ( gtx, positions.size(),
              makeContiguous( positions ).addr(),
              makeContiguous( uvCoords  ).addr(),
              tex2ID, color, type );
}


void ShaderDraw::drawTex2D

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const GLfloat*  uvCoords,
    GLuint          tex2ID,
    const Color&    color,
    Type            type )

{
  ShaderDraw  drawer   ( FLAT_TEX2_COLOR_, count, type );

  drawer.setPositions_ ( positions );
  drawer.set2DTexture_ ( uvCoords, tex2ID );
  drawer.setColor_     ( color );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawTex2D -- texture 2D (alpha only) and shading
//-----------------------------------------------------------------------


void ShaderDraw::drawTex2D

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  normals,
    const GLfloatVector&  uvCoords,
    GLuint                tex2ID,
    const Color&          color,
    Type                  type )

{
  JEM_PRECHECK ( (positions.size() == normals.size()) &&
                 (positions.size() / VERTEX_SIZE_ ==
                  uvCoords .size() / TEX2CRD_SIZE_) );


  drawTex2D ( gtx, positions.size(),
              makeContiguous( positions ).addr(),
              makeContiguous( normals   ).addr(),
              makeContiguous( uvCoords  ).addr(),
              tex2ID, color, type );
}


void ShaderDraw::drawTex2D

  ( GLContext&      gtx,
    idx_t           count,
    const GLfloat*  positions,
    const GLfloat*  normals,
    const GLfloat*  uvCoords,
    GLuint          tex2ID,
    const Color&    color,
    Type            type )

{
  ShaderDraw  drawer   ( SHADE_TEX2_COLOR_, count, type );

  drawer.setPositions_ ( positions );
  drawer.setNormals_   ( normals );
  drawer.set2DTexture_ ( uvCoords, tex2ID );
  drawer.setColor_     ( color );
  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawStippleLines
//-----------------------------------------------------------------------


void ShaderDraw::drawStippleLines

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const Color&          color,
    GLuint                factor,
    GLuint                pattern )

{
  drawStippleLines ( gtx, positions.size(),
                     makeContiguous( positions ).addr(),
                     color, factor, pattern );
}


void ShaderDraw::drawStippleLines

  ( GLContext&       gtx,
    idx_t            count,
    const GLfloat*   positions,
    const Color&     color,
    GLuint           factor,
    GLuint           pattern)

{
  ShaderDraw  drawer   ( LINES_STIPPLE_, count, LINES );

  drawer.setPositions_ ( positions );
  drawer.setStipple_   ( factor, pattern );
  drawer.setColor_     ( color );

  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawOrbs -- uniform appearance
//-----------------------------------------------------------------------


void ShaderDraw::drawOrbs

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    float                 radius,
    const Color&          color )

{
  drawOrbs ( gtx, positions.size(),
             makeContiguous( positions ).addr(),
             radius,
             color );
}


void ShaderDraw::drawOrbs

  ( GLContext&      gtx,
    idx_t           size,
    const GLfloat*  positions,
    float           radius,
    const Color&    color )

{
  JEM_PRECHECK ( radius > 0.0F );

  ShaderDraw  drawer   ( ORBS_UNIFORM_, size, POINTS );

  drawer.setPositions_ ( positions );
  drawer.setRadius_    ( radius );
  drawer.setColor_     ( color );

  drawer.options_      |= HAS_SHADING_;

  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawOrbs -- varying appearance
//-----------------------------------------------------------------------


void ShaderDraw::drawOrbs

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    const GLfloatVector&  radii,
    const GLfloatVector&  colors )

{
  JEM_PRECHECK ( positions.size() / VERTEX_SIZE_ ==
                 radii    .size() / RADIUS_SIZE_ );

  drawOrbs ( gtx, positions.size(),
             makeContiguous( positions ).addr(),
             makeContiguous( radii     ).addr(),
             makeContiguous( colors    ).addr() );
}


void ShaderDraw::drawOrbs

  ( GLContext&      gtx,
    idx_t           size,
    const GLfloat*  positions,
    const GLfloat*  radii,
    const GLfloat*  colors )

{
  ShaderDraw  drawer   ( ORBS_VARYING_, size, POINTS );

  drawer.setPositions_ ( positions );
  drawer.setRadii_     ( radii );
  drawer.setColors_    ( colors );

  drawer.options_      |= HAS_SHADING_;

  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawOrbs -- checker appearance
//-----------------------------------------------------------------------


void ShaderDraw::drawOrbs

  ( GLContext&            gtx,
    const GLfloatVector&  positions,
    float                 radius,
    const Color&          color1,
    const Color&          color2 )

{
  drawOrbs ( gtx, positions.size(),
             makeContiguous( positions ).addr(),
             radius, color1, color2 );
}

void ShaderDraw::drawOrbs

  ( GLContext&      gtx,
    idx_t           size,
    const GLfloat*  positions,
    float           radius,
    const Color&    color1,
    const Color&    color2 )

{
  JEM_PRECHECK ( radius > 0.0F );

  ShaderDraw  drawer   ( ORBS_CHECKER_, size, POINTS );

  drawer.setPositions_ ( positions );
  drawer.setRadius_    ( radius );
  drawer.setColor_     ( color1 );
  drawer.setColor2_    ( color2 );

  drawer.options_      |= HAS_SHADING_;

  drawer.draw_         ( gtx );
}


//-----------------------------------------------------------------------
//   drawOrb
//-----------------------------------------------------------------------


void ShaderDraw::drawOrb

  ( GLContext&       gtx,
    const Vertex3f&  position,
    float            radius,
    const Color&     color )

{
  drawOrbs ( gtx, 3, position.addr(), radius, color );
}


void ShaderDraw::drawOrb

  ( GLContext&       gtx,
    const Vertex3f&  position,
    float            radius,
    const Color&     color1,
    const Color&     color2 )

{
  drawOrbs ( gtx, 3, position.addr(), radius, color1, color2 );
}


//-----------------------------------------------------------------------
//   uploadLights
//-----------------------------------------------------------------------


void ShaderDraw::uploadLights

  ( GLContext&  gtx,
    ShaderProg&  shader )

{
  const Lights&  lights    = *gtx   .getLights     ();
  Color          ambient   =  lights.getAmbient    ();
  Color          specular  =  gtx   .getSpecular   ();
  GLfloat        shininess =  gtx   .getShininess  ();
  int            mask      =  lights.getActiveMask ();
  int            light     =  Lights::LIGHT0;
  int            count     =  0;


  if ( ! gtx.getLighting() )
  {
    mask     = 0;
    specular = Color::black ();
    ambient  = Color::white ();
  }

  while ( mask > 0 )
  {
    JEM_ASSERT ( count < 3 );

    if ( mask & 1 )
    {
      shader.setUniform3 ( gtx, ShaderUnis::LIGHT_COLOR[count],
                           lights.getColor     ( light ) );
      shader.setUniform3 ( gtx, ShaderUnis::LIGHT_DIR[count],
                           lights.getDirection ( light ) );
    }

    mask  >>= 1;
    light <<= 1;
    count++;
  }

  shader.setUniform3 ( gtx, ShaderUnis::AMBIENT,     ambient   );
  shader.setUniform3 ( gtx, ShaderUnis::SPECULAR,    specular  );
  shader.setUniform  ( gtx, ShaderUnis::SHININESS,   shininess );
  shader.setUniform  ( gtx, ShaderUnis::LIGHT_COUNT, count     );
}


//-----------------------------------------------------------------------
//   initType_
//-----------------------------------------------------------------------


void ShaderDraw::initType_ ()
{
  switch ( geomType_ )
  {
  case LINES:
  case LINE_STRIP:

    initLines_ ();

    return;

  default:

    return;
  }
}


//-----------------------------------------------------------------------
//   initType_
//-----------------------------------------------------------------------


void ShaderDraw::initLines_ ()
{
  options_ |= HAS_WIDTH_;

  switch ( shaderType_ )
  {
  case FLAT_COLOR_:

    shaderType_ = LINES_CONST_COLOR_;

    return;

  case FLAT_COLORS_:

    shaderType_ = LINES_MULTI_COLOR_;

    return;

  case FLAT_TEX1_:

    shaderType_ = LINES_TEX1D_;

    return;

  case LINES_STIPPLE_:

    return;

  default:

    throw GLException ( JEM_FUNC,
                        "combination of shader type and lines "
                        "has not been implemented.");
  }
}


//-----------------------------------------------------------------------
//   setPositions_
//-----------------------------------------------------------------------


void ShaderDraw::setPositions_ ( const GLfloat* positions )
{
  positions_ = positions;
}


//-----------------------------------------------------------------------
//   setNormals_
//-----------------------------------------------------------------------


void ShaderDraw::setNormals_ ( const GLfloat* normals )
{
  normals_  = normals;
  options_ |= HAS_SHADING_;
}


//-----------------------------------------------------------------------
//   setColors_
//-----------------------------------------------------------------------


void ShaderDraw::setColors_ ( const GLfloat* colors )
{
  colors_ = colors;
}


//-----------------------------------------------------------------------
//   setRadii_
//-----------------------------------------------------------------------


void ShaderDraw::setRadii_ ( const GLfloat* radii )
{
  radii_ = radii;
}


//-----------------------------------------------------------------------
//   setColor_
//-----------------------------------------------------------------------


void ShaderDraw::setColor_ ( const Color& color )
{
  color_    = color;
  options_ |= HAS_COLOR_;
}


//-----------------------------------------------------------------------
//   setColor2_
//-----------------------------------------------------------------------


void ShaderDraw::setColor2_ ( const Color& color2 )
{
  color2_   = color2;
  options_ |= HAS_COLOR2_;
}


//-----------------------------------------------------------------------
//   set1DTexture_
//-----------------------------------------------------------------------


void ShaderDraw::set1DTexture_

  ( const GLfloat*  uCoords,
    GLuint          tex1ID )

{
  uCoords_  = uCoords;
  tex1ID_   = tex1ID;

  options_ |= HAS_TEX1_;
}


//-----------------------------------------------------------------------
//   set2DTexture_
//-----------------------------------------------------------------------


void ShaderDraw::set2DTexture_

  ( const GLfloat*  uvCoords,
    GLuint          tex2ID )

{
  uvCoords_ = uvCoords;
  tex2ID_   = tex2ID;
  options_ |= HAS_TEX2_;
}


//-----------------------------------------------------------------------
//   setStipple_
//-----------------------------------------------------------------------


void ShaderDraw::setStipple_

  ( GLuint  factor,
    GLuint  pattern )

{
  stipFactor_ = factor;
  stipPttrn_  = pattern;

  options_   |= HAS_STIPPLE_;
  options_   |= HAS_WIDTH_;
}


//-----------------------------------------------------------------------
//   setRadius_
//-----------------------------------------------------------------------


void ShaderDraw::setRadius_ ( float radius )
{
  radius_   = radius;
  options_ |= HAS_RADIUS_;
}


//-----------------------------------------------------------------------
//   draw_
//-----------------------------------------------------------------------


void ShaderDraw::draw_ ( GLContext&  gtx ) const
{
  JEM_PRECHECK ( geomCount_ % VERTEX_SIZE_ == 0 );
  JEM_PRECHECK ( positions_ );

  GLFuncs3_3Core&  funcs      = gtx.getFunctions<GLFuncs3_3Core> ();
  idx_t            vertCount  = geomCount_ / VERTEX_SIZE_;
  idx_t            skipCount  = 0;
  idx_t            offset     = 0;

  ShaderProg&      shader     = getProgram_ ( gtx, shaderType_ );
  GLfloat          alphaMin   = gtx.getAlphaTest ();

  VertexBuffer     vertexBuf  ( ShaderBuffers::POSITIONS   );
  VertexBuffer     normalBuf  ( ShaderBuffers::NORMALS     );
  VertexBuffer     colorsBuf  ( ShaderBuffers::COLORS      );
  VertexBuffer     radiiBuf   ( ShaderBuffers::RADII       );
  VertexBuffer     tex1CrdBuf ( ShaderBuffers::TEX1_COORDS );
  VertexBuffer     tex2CrdBuf ( ShaderBuffers::TEX2_COORDS );

  VertexArray      vao        ( gtx );

  GLfloat          mvp[16];


  if ( ! shader.bind( gtx ) )
  {
    return;
  }

  gtx.getModelViewProj ( mvp );
  shader.setUniform    ( gtx, ShaderUnis::ALPHA_MIN, alphaMin );
  shader.setUniMatrix4 ( gtx, ShaderUnis::MVP,       mvp );

  if ( options_ & HAS_SHADING_ )
  {
    GLfloat  mv  [16];
    GLfloat  proj[16];
    GLfloat  norm[9];

    uploadLights         ( gtx,  shader );

    gtx.getTransform     ( mv,   GLContext::MODEL_VIEW );
    gtx.getTransform     ( proj, GLContext::PROJECTION );
    gtx.getNormalMatrix  ( norm );

    shader.setUniMatrix4 ( gtx, ShaderUnis::MV,        mv   );
    shader.setUniMatrix4 ( gtx, ShaderUnis::PROJ_MAT,  proj );
    shader.setUniMatrix3 ( gtx, ShaderUnis::NORM_MAT,  norm );
  }

  if ( options_ & HAS_COLOR_ )
  {
    shader.setUniform4 ( gtx, ShaderUnis::COLOR, color_ );
  }

  if ( options_ & HAS_COLOR2_ )
  {
    shader.setUniform4 ( gtx, ShaderUnis::COLOR2, color2_ );
  }

  if ( options_ & HAS_TEX1_ )
  {
    funcs .glActiveTexture ( GL_TEXTURE0 );
    funcs .glBindTexture   ( GL_TEXTURE_1D, tex1ID_ );
    shader.setUniform      ( gtx, ShaderUnis::TEXTURE_1D, 0 );
  }

  if ( options_ & HAS_TEX2_ )
  {
    if ( options_ & HAS_TEX1_ )
    {
      funcs.glActiveTexture ( GL_TEXTURE1 );
    }
    else
    {
      funcs.glActiveTexture ( GL_TEXTURE0 );
    }

    funcs .glBindTexture ( GL_TEXTURE_2D, tex2ID_ );
    shader.setUniform    ( gtx, ShaderUnis::TEXTURE_2D, 0 );
  }

  if ( options_ & HAS_STIPPLE_ )
  {
    shader.setUniform ( gtx, ShaderUnis::FACTOR,  stipFactor_ );
    shader.setUniform ( gtx, ShaderUnis::PATTERN, stipPttrn_ );
  }

  if ( options_ & HAS_WIDTH_ )
  {
    GLfloat  width = gtx.getLineWidth ();

    JEM_PRECHECK ( width >= 0.0F );

    shader.setUniform ( gtx, ShaderUnis::WIDTH, width );
  }

  if ( options_ & (HAS_WIDTH_ | HAS_STIPPLE_) )
  {
    Rect      vport     = gtx.getViewport ();
    Vertex2f  frameSize ( (GLfloat) vport.width,
                          (GLfloat) vport.height );

    shader.setUniform2 ( gtx, ShaderUnis::FRAME_SIZE, frameSize );
  }

  if ( options_ & HAS_RADIUS_ )
  {
    shader.setUniform ( gtx, ShaderUnis::RADIUS, radius_ );
  }

  vao.addAttrib ( vertexBuf.getBufferID ( gtx ),
                  VERTEX_SIZE_, GL_FLOAT );

  if ( normals_ )
  {
    vao.addAttrib ( normalBuf.getBufferID ( gtx ),
                    NORMAL_SIZE_, GL_FLOAT );
  }

  if ( colors_ )
  {
    vao.addAttrib ( colorsBuf.getBufferID ( gtx ),
                    COLOR_SIZE_, GL_FLOAT );
  }

  if ( uCoords_ )
  {
    vao.addAttrib ( tex1CrdBuf.getBufferID ( gtx ),
                    TEX1CRD_SIZE_, GL_FLOAT );
  }

  if ( uvCoords_ )
  {
    vao.addAttrib ( tex2CrdBuf.getBufferID ( gtx ),
                    TEX2CRD_SIZE_, GL_FLOAT );
  }

  if ( radii_ )
  {
    vao.addAttrib ( radiiBuf.getBufferID ( gtx ),
                    RADIUS_SIZE_, GL_FLOAT );
  }

  // Write the draw data in blocks to the GPU.

  while ( checkSize_( vertCount, geomType_ ) )
  {
    idx_t   blockSize = getBlockSize_ ( vertCount, geomType_ );
    idx_t   dataStep  = getDataStep_  ( blockSize, geomType_ );

    vertexBuf.setData ( gtx,
                        positions_ + VERTEX_SIZE_ * skipCount,
                        VERTEX_SIZE_ * blockSize, offset );

    if ( normals_ )
    {
      normalBuf.setData ( gtx,
                          normals_ + NORMAL_SIZE_ * skipCount,
                          NORMAL_SIZE_ * blockSize, offset );
    }

    if ( colors_ )
    {
      colorsBuf.setData ( gtx,
                          colors_ + COLOR_SIZE_ * skipCount,
                          COLOR_SIZE_ * blockSize, offset );
    }

    if ( uCoords_ )
    {
      tex1CrdBuf.setData ( gtx,
                           uCoords_ + TEX1CRD_SIZE_ * skipCount,
                           TEX1CRD_SIZE_ * blockSize, offset );
    }

    if ( uvCoords_ )
    {
      tex2CrdBuf.setData ( gtx,
                           uvCoords_ + TEX2CRD_SIZE_ * skipCount,
                           TEX2CRD_SIZE_ * blockSize, offset );
    }

    if ( radii_ )
    {
      radiiBuf.setData ( gtx,
                         radii_ + RADIUS_SIZE_ * skipCount,
                         RADIUS_SIZE_ * blockSize, offset );
    }

    drawArrays_ ( funcs, blockSize + offset, geomType_ );

    skipCount += dataStep;
    vertCount -= dataStep;
    offset     = getOffset_ ( geomType_ );
  }

  shader.unbind ( gtx );
  checkGLError  ( gtx, JEM_FUNC );
}


//-----------------------------------------------------------------------
//   getProgram_
//-----------------------------------------------------------------------


ShaderProg& ShaderDraw::getProgram_

  ( GLContext&   gtx,
    ShaderType_  type )

{
  switch ( type )
  {
  case FLAT_COLOR_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::FLAT_CONST_COLOR );

  case FLAT_COLORS_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::FLAT_MULTI_COLOR );

  case SHADE_COLOR_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::SHADE_CONST_COLOR );

  case SHADE_COLORS_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::SHADE_MULTI_COLOR );

  case FLAT_TEX1_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::FLAT_TEX1D );

  case FLAT_TEX1_COLOR_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::FLAT_TEX1D_ALPHA );

  case SHADE_TEX1_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::SHADE_TEX1D );

  case SHADE_TEX1_COLOR_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::SHADE_TEX1D_ALPHA );

  case FLAT_TEX2_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::FLAT_TEX2D );

  case FLAT_TEX2_COLOR_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::FLAT_TEX2D_ALPHA );

  case SHADE_TEX2_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::SHADE_TEX2D );

  case SHADE_TEX2_COLOR_:

    return ShaderSource::
           getProgram ( gtx, ShaderNames::SHADE_TEX2D_ALPHA );

  case ORBS_UNIFORM_:

    return ShaderSource::
            getProgram ( gtx, ShaderNames::ORBS_UNIFORM );

  case ORBS_VARYING_:

    return ShaderSource::
            getProgram ( gtx, ShaderNames::ORBS_VARYING );

  case ORBS_CHECKER_:

    return ShaderSource::
            getProgram ( gtx, ShaderNames::ORBS_CHECKER );

  case LINES_CONST_COLOR_:

    return ShaderSource::
            getProgram ( gtx, ShaderNames::LINES_CONST_COLOR );

  case LINES_MULTI_COLOR_:

    return ShaderSource::
            getProgram ( gtx, ShaderNames::LINES_MULTI_COLOR );

  case LINES_TEX1D_:

    return ShaderSource::
            getProgram ( gtx, ShaderNames::LINES_TEX1D );

  case LINES_STIPPLE_:

    return ShaderSource::
            getProgram ( gtx, ShaderNames::LINES_STIPPLE );

  default:

    throw GLException (
      JEM_FUNC,
      "invalid shader type: " + String( (int) type )
    );
  }
}


//-----------------------------------------------------------------------
//   drawArrays_
//-----------------------------------------------------------------------


void ShaderDraw::drawArrays_

  ( GLFunctions3_3Core&  funcs,
    idx_t                vertCount,
    Type                 type )

{
  GLsizei  count = (GLsizei) vertCount;

  JEM_PRECHECK ( count >= 0 );

  switch ( type )
  {
  case POINTS:

    funcs.glDrawArrays ( GL_POINTS,         0, count );

    break;

  case LINES:

    funcs.glDrawArrays ( GL_LINES,          0, count );

    break;

  case LINE_STRIP:

    funcs.glDrawArrays ( GL_LINE_STRIP,     0, count );

    break;

  case TRIANGLES:

    funcs.glDrawArrays ( GL_TRIANGLES,      0, count );

    break;

  case TRIANGLE_STRIP:

    funcs.glDrawArrays ( GL_TRIANGLE_STRIP, 0, count );

    break;

  case TRIANGLE_FAN:

    funcs.glDrawArrays ( GL_TRIANGLE_FAN,   0, count );

    break;

  default:

    throw GLException (
      JEM_FUNC,
      "invalid array draw type: " + String ( (int) type )
    );
  }
}


//-----------------------------------------------------------------------
//   getBlockSize_
//-----------------------------------------------------------------------


idx_t ShaderDraw::getBlockSize_

  ( idx_t  count,
    Type   type )

{
  using jem::min;

  idx_t  bsz = 0;

  switch ( type )
  {
  case TRIANGLE_FAN:
  case TRIANGLE_STRIP:
  case LINE_STRIP:
  case POINTS:

    bsz = min ( count, MAX_BLOCK_SIZE_ );

    break;

  case LINES:

    bsz = 2 * min ( count / 2, MAX_BLOCK_SIZE_ / 2 );

    break;

  case TRIANGLES:

    bsz = 3 * min ( count / 3, MAX_BLOCK_SIZE_ / 3 );

    break;

  default:

    throw GLException ( JEM_FUNC, "unknown shader draw type" );
  }

  return bsz;
}


//-----------------------------------------------------------------------
//   getDataStep_
//-----------------------------------------------------------------------


idx_t ShaderDraw::getDataStep_

  ( idx_t  blockSize,
    Type   type )

{
  idx_t  stride = 0;

  switch ( type )
  {
  case LINES:
  case POINTS:
  case TRIANGLES:

    stride = blockSize;

    break;

  case TRIANGLE_FAN:
  case LINE_STRIP:

    stride = blockSize - 1;

    break;

  case TRIANGLE_STRIP:

    stride = blockSize - 2;

    break;

  default:

    throw GLException ( JEM_FUNC, "unknown shader draw type" );
  }

  return stride;
}


//-----------------------------------------------------------------------
//   getOffset_
//-----------------------------------------------------------------------


idx_t ShaderDraw::getOffset_ ( Type type )
{
  if ( type == TRIANGLE_FAN )
  {
    return 1;
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   checkSize_
//-----------------------------------------------------------------------


bool ShaderDraw::checkSize_

  ( idx_t  count,
    Type   type )

{
  switch ( type )
  {
  case TRIANGLE_FAN:
  case TRIANGLES:
  case TRIANGLE_STRIP:

    return (count >= 3);

  case LINES:
  case LINE_STRIP:

    return (count >= 2);

  case POINTS:

    return (count >= 1);

  default:

    throw GLException ( JEM_FUNC, "unknown shader draw type" );
  }

  return false;
}


JEM_END_PACKAGE( gl )
