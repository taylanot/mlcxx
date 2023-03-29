
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

#ifndef JEM_GL_SHADERDRAW_H
#define JEM_GL_SHADERDRAW_H

#include <jem/gl/Array.h>
#include <jem/gl/Color.h>
#include <jem/gl/Vertex.h>


JEM_BEGIN_PACKAGE( gl )


class Color;
class GLContext;
class ShaderProg;
class GLFunctions3_3Core;


//-----------------------------------------------------------------------
//   class ShaderDraw
//-----------------------------------------------------------------------


class ShaderDraw
{
 public:

  enum                    Type
  {
                            POINTS,
                            LINES,
                            LINE_STRIP,
                            TRIANGLES,
                            TRIANGLE_STRIP,
                            TRIANGLE_FAN
  };

  static void             draw

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const Color&          color,
      Type                  type );

  static void             draw

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const Color&          color,
      Type                  type );

  static void             draw

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const GLfloatVector&  normals,
      const Color&          color,
      Type                  type );

  static void             draw

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const GLfloat*        normals,
      const Color&          color,
      Type                  type );

  static void             draw

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const GLfloatVector&  colors,
      Type                  type );

  static void             draw

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const GLfloat*        colors,
      Type                  type );

  static void             draw

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const GLfloatVector&  normals,
      const GLfloatVector&  colors,
      Type                  type );

  static void             draw

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const GLfloat*        normals,
      const GLfloat*        colors,
      Type                  type );

  static void             drawTex1D

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const GLfloatVector&  vertU,
      GLuint                texID,
      Type                  type );

  static void             drawTex1D

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const GLfloat*        vertU,
      GLuint                texID,
      Type                  type );

  static void             drawTex1D

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const GLfloatVector&  normals,
      const GLfloatVector&  vertU,
      GLuint                texID,
      Type                  type );

  static void             drawTex1D

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const GLfloat*        normals,
      const GLfloat*        vertU,
      GLuint                texID,
      Type                  type );

  static void             drawTex1D

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const GLfloatVector&  vertU,
      GLuint                texID,
      const Color&          color,
      Type                  type );

  static void             drawTex1D

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const GLfloat*        vertU,
      GLuint                texID,
      const Color&          color,
      Type                  type );

  static void             drawTex1D

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const GLfloatVector&  normals,
      const GLfloatVector&  vertU,
      GLuint                texID,
      const Color&          color,
      Type                  type );

  static void             drawTex1D

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const GLfloat*        normals,
      const GLfloat*        vertU,
      GLuint                texID,
      const Color&          color,
      Type                  type );

  static void             drawTex2D

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const GLfloatVector&  vertUV,
      GLuint                texID,
      Type                  type );

  static void             drawTex2D

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const GLfloat*        vertUV,
      GLuint                texID,
      Type                  type );

  static void             drawTex2D

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const GLfloatVector&  normals,
      const GLfloatVector&  vertUV,
      GLuint                texID,
      Type                  type );

  static void             drawTex2D

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const GLfloat*        normals,
      const GLfloat*        vertUV,
      GLuint                texID,
      Type                  type );

  static void             drawTex2D

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const GLfloatVector&  vertUV,
      GLuint                texID,
      const Color&          color,
      Type                  type );

  static void             drawTex2D

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const GLfloat*        vertUV,
      GLuint                texID,
      const Color&          color,
      Type                  type );

  static void             drawTex2D

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const GLfloatVector&  normals,
      const GLfloatVector&  vertUV,
      GLuint                texID,
      const Color&          color,
      Type                  type );

  static void             drawTex2D

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const GLfloat*        normals,
      const GLfloat*        vertUV,
      GLuint                texID,
      const Color&          color,
      Type                  type );

  static void             drawStippleLines

    ( GLContext&            gtx,
      const GLfloatVector&  vertPos,
      const Color&          color,
      GLuint                factor,
      GLuint                pattern );

  static void             drawStippleLines

    ( GLContext&            gtx,
      idx_t                 count,
      const GLfloat*        vertPos,
      const Color&          color,
      GLuint                factor,
      GLuint                pattern );

  static void             drawOrbs

    ( GLContext&            gtx,
      const GLfloatVector&  positions,
      float                 radius,
      const Color&          color );

  static void             drawOrbs

    ( GLContext&            gtx,
      idx_t                 size,
      const GLfloat*        positions,
      float                 radius,
      const Color&          color );

  static void             drawOrbs

    ( GLContext&            gtx,
      const GLfloatVector&  positions,
      const GLfloatVector&  radii,
      const GLfloatVector&  colors );

  static void             drawOrbs

    ( GLContext&            gtx,
      idx_t                 size,
      const GLfloat*        positions,
      const GLfloat*        radii,
      const GLfloat*        colors );

  static void             drawOrbs

    ( GLContext&            gtx,
      const GLfloatVector&  positions,
      float                 radius,
      const Color&          color1,
      const Color&          color2 );

  static void             drawOrbs

    ( GLContext&            gtx,
      idx_t                 size,
      const GLfloat*        positions,
      float                 radius,
      const Color&          color1,
      const Color&          color2 );

  static void             drawOrb

    ( GLContext&            gtx,
      const Vertex3f&       position,
      float                 radius,
      const Color&          color );

  static void             drawOrb

    ( GLContext&            gtx,
      const Vertex3f&       position,
      float                 radius,
      const Color&          color1,
      const Color&          color2 );

  static void             uploadLights

    ( GLContext&            gtx,
      ShaderProg&           shader );


 private:

  enum                    ShaderType_
  {
                            FLAT_COLOR_,
                            FLAT_COLORS_,
                            SHADE_COLOR_,
                            SHADE_COLORS_,
                            FLAT_TEX1_,
                            FLAT_TEX1_COLOR_,
                            SHADE_TEX1_,
                            SHADE_TEX1_COLOR_,
                            FLAT_TEX2_,
                            FLAT_TEX2_COLOR_,
                            SHADE_TEX2_,
                            SHADE_TEX2_COLOR_,
                            ORBS_CHECKER_,
                            ORBS_UNIFORM_,
                            ORBS_VARYING_,
                            LINES_CONST_COLOR_,
                            LINES_MULTI_COLOR_,
                            LINES_TEX1D_,
                            LINES_STIPPLE_
  };

                          ShaderDraw

    ( ShaderType_           shaderType,
      idx_t                 geomCount,
      Type                  geomType );

  void                    initType_     ();
  void                    initLines_    ();

  void                    setPositions_

    ( const GLfloat*        positions );

  void                    setNormals_

    ( const GLfloat*        normals );

  void                    setColors_

    ( const GLfloat*        colors );

  void                    setRadii_

    ( const GLfloat*        radii );

  void                    setColor_

    ( const Color&          color );

  void                    setColor2_

    ( const Color&          color2 );

  void                    set1DTexture_

    ( const GLfloat*        uCoords,
      GLuint                tex1ID );

  void                    set2DTexture_

    ( const GLfloat*        uvCoords,
      GLuint                tex2ID );

  void                    setStipple_

    ( GLuint                factor,
      GLuint                pattern );

  void                    setRadius_

    ( float                 radius );

  void                    draw_

    ( GLContext&            gtx ) const;

  static ShaderProg&      getProgram_

    ( GLContext&            gtx,
      ShaderType_           type );

  static void             drawArrays_

    ( GLFunctions3_3Core&   funcs,
      idx_t                 count,
      Type                  type );

  static idx_t            getBlockSize_

    ( idx_t                 count,
      Type                  type );

  static idx_t            getDataStep_

    ( idx_t                 blockSize,
      Type                  type );

  static idx_t            getOffset_

    ( Type                  type );

  static bool             checkSize_

    ( idx_t                 count,
      Type                  type );


 private:

  static const int        COLOR_SIZE_;
  static const int        NORMAL_SIZE_;
  static const int        RADIUS_SIZE_;
  static const int        TEX1CRD_SIZE_;
  static const int        TEX2CRD_SIZE_;
  static const int        VERTEX_SIZE_;
  static const idx_t      MAX_BLOCK_SIZE_;

  enum                    Option_
  {
                            HAS_COLOR_   = 1 << 0,
                            HAS_COLOR2_  = 1 << 1,
                            HAS_RADIUS_  = 1 << 2,
                            HAS_SHADING_ = 1 << 3,
                            HAS_STIPPLE_ = 1 << 4,
                            HAS_TEX1_    = 1 << 5,
                            HAS_TEX2_    = 1 << 6,
                            HAS_WIDTH_   = 1 << 7
  };

  const GLfloat*          positions_;
  const GLfloat*          normals_;
  const GLfloat*          colors_;
  const GLfloat*          uCoords_;
  const GLfloat*          uvCoords_;
  const GLfloat*          radii_;

  Color                   color_;
  Color                   color2_;
  GLuint                  tex1ID_;
  GLuint                  tex2ID_;
  float                   radius_;

  ShaderType_             shaderType_;
  Type                    geomType_;
  GLuint                  stipFactor_;
  GLuint                  stipPttrn_;
  idx_t                   geomCount_;
  int                     options_;

};


JEM_END_PACKAGE( gl )

#endif
