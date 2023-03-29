
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


#include <jem/base/assert.h>
#include <jem/gl/render.h>
#include <jem/gl/Color.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/ShaderDraw.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   forward declaration of functions
//-----------------------------------------------------------------------


static void     renderArrow3D_v1

  ( GLContext&    gtx,
    const Color&  color,
    GLfloat       size );

static void     renderArrow3D_v3

  ( GLContext&    gtx,
    const Color&  color,
    GLfloat       size );


//-----------------------------------------------------------------------
//   renderArrow3D
//-----------------------------------------------------------------------


void            renderArrow3D

  ( GLContext&    gtx,
    const Color&  color,
    GLfloat       size )

{
  JEM_ASSERT ( size > 0.0F );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    renderArrow3D_v3 ( gtx, color, size );
  }
  else
  {
    renderArrow3D_v1 ( gtx, color, size );
  }
}


//-----------------------------------------------------------------------
//   renderArrow3D_v1
//-----------------------------------------------------------------------


void            renderArrow3D_v1

  ( GLContext&    gtx,
    const Color&  color,
    GLfloat       size )

{
  JEM_ASSERT ( size > 0.0F );

  static const GLfloat  POINTS[9][3]  =

    { {  0.577350F,  0.000000F, -1.0F },
      {  0.408248F,  0.408248F, -1.0F },
      {  0.000000F,  0.577350F, -1.0F },
      { -0.408248F,  0.408248F, -1.0F },
      { -0.577350F,  0.000000F, -1.0F },
      { -0.408248F, -0.408248F, -1.0F },
      {  0.000000F, -0.577350F, -1.0F },
      {  0.408248F, -0.408248F, -1.0F },
      {  0.577350F,  0.000000F, -1.0F } };

  static const GLfloat  NORMALS[9][3] =

    { {  0.866025F,  0.000000F,  0.5F },
      {  0.612372F,  0.612372F,  0.5F },
      {  0.000000F,  0.866025F,  0.5F },
      { -0.612372F,  0.612372F,  0.5F },
      { -0.866025F,  0.000000F,  0.5F },
      { -0.612372F, -0.612372F,  0.5F },
      {  0.000000F, -0.866025F,  0.5F },
      {  0.612372F, -0.612372F,  0.5F },
      {  0.866025F,  0.000000F,  0.5F } };

  const GLfloat  COLOR_SCALE = 0.4F;

  const GLfloat  z           = 1.0F - size;
  GLFunctions1*  funcs       = gtx.findFunctions<GLFunctions1> ();

  AttribScope    attribs     ( gtx, GLContext::COLOR |
                                    GLContext::LIGHTING );

  GLfloat        c[4];


  if ( ! funcs )
  {
    return;
  }

  funcs->glColor4 ( color.addr() );

  // The arrow head.

  {
    funcs->glBegin ( GL_TRIANGLES );

    for ( int i = 0; i < 8; i++ )
    {
      funcs->glNormal3 ( NORMALS[i] );

      funcs->glVertex  ( 0.0F, 0.0F, 1.0F );

      funcs->glVertex  ( size * POINTS[i][0],
                         size * POINTS[i][1], z );

      funcs->glNormal3 ( NORMALS[i + 1] );

      funcs->glVertex  ( size * POINTS[i + 1][0],
                         size * POINTS[i + 1][1], z );
    }

    funcs->glEnd ();

    c[0] = COLOR_SCALE * color[0];
    c[1] = COLOR_SCALE * color[1];
    c[2] = COLOR_SCALE * color[2];
    c[3] =               color[3];

    funcs->glBegin  ( GL_POLYGON );

    funcs->glColor4 ( c );
    funcs->glNormal ( 0.0F, 0.0F, -1.0F );

    for ( int i = 7; i >= 0; i-- )
    {
      funcs->glVertex ( size * POINTS[i][0],
                        size * POINTS[i][1], z );
    }

    funcs->glEnd ();

    funcs->glColor4 ( color.addr() );
  }

  // The arrow line.

  gtx.setLighting ( false );

  funcs->glBegin  ( GL_LINES );
  funcs->glVertex ( 0.0F, 0.0F, 0.0F );
  funcs->glVertex ( 0.0F, 0.0F,    z );
  funcs->glEnd    ();
}


//-----------------------------------------------------------------------
//   renderArrow3D_v3
//-----------------------------------------------------------------------


void            renderArrow3D_v3

  ( GLContext&    gtx,
    const Color&  color,
    GLfloat       size )

{
  JEM_ASSERT ( size > 0.0F );

  static const GLfloat  POINTS[9][3]  =

    { {  0.577350F,  0.000000F, -1.0F },
      {  0.408248F,  0.408248F, -1.0F },
      {  0.000000F,  0.577350F, -1.0F },
      { -0.408248F,  0.408248F, -1.0F },
      { -0.577350F,  0.000000F, -1.0F },
      { -0.408248F, -0.408248F, -1.0F },
      {  0.000000F, -0.577350F, -1.0F },
      {  0.408248F, -0.408248F, -1.0F },
      {  0.577350F,  0.000000F, -1.0F } };

  static const GLfloat  NORMALS[9][3] =

    { {  0.866025F,  0.000000F,  0.5F },
      {  0.612372F,  0.612372F,  0.5F },
      {  0.000000F,  0.866025F,  0.5F },
      { -0.612372F,  0.612372F,  0.5F },
      { -0.866025F,  0.000000F,  0.5F },
      { -0.612372F, -0.612372F,  0.5F },
      {  0.000000F, -0.866025F,  0.5F },
      {  0.612372F, -0.612372F,  0.5F },
      {  0.866025F,  0.000000F,  0.5F } };

  const GLfloat  COLOR_SCALE = 0.4F;
  const GLfloat  z           = 1.0F - size;

  const GLfloat  vertLine[6] = { 0.0F, 0.0F, 0.0F,
                                 0.0F, 0.0F,    z };

  GLfloat        vertPos  [72];
  GLfloat        vertNorms[72];
  Color          colorBottom;

  // Make arrow head (top).

  for ( int i = 0; i < 8; i++ )
  {
    idx_t  i9 = 9 * i;

    vertPos  [i9 + 0] = 0.0F;
    vertPos  [i9 + 1] = 0.0F;
    vertPos  [i9 + 2] = 1.0F;

    vertNorms[i9 + 0] = NORMALS[i][0];
    vertNorms[i9 + 1] = NORMALS[i][1];
    vertNorms[i9 + 2] = NORMALS[i][2];

    vertPos  [i9 + 3] = size * POINTS[i][0];
    vertPos  [i9 + 4] = size * POINTS[i][1];
    vertPos  [i9 + 5] = z;

    vertNorms[i9 + 3] = NORMALS[i][0];
    vertNorms[i9 + 4] = NORMALS[i][1];
    vertNorms[i9 + 5] = NORMALS[i][2];

    vertPos  [i9 + 6] = size * POINTS[i + 1][0];
    vertPos  [i9 + 7] = size * POINTS[i + 1][1];
    vertPos  [i9 + 8] = z;

    vertNorms[i9 + 6] = NORMALS[i + 1][0];
    vertNorms[i9 + 7] = NORMALS[i + 1][1];
    vertNorms[i9 + 8] = NORMALS[i + 1][2];
  }

  ShaderDraw::draw ( gtx, 72, vertPos, vertNorms, color,
                     ShaderDraw::TRIANGLES );

  // Make arrow head (bottom).

  vertPos  [0] =  0.0F;
  vertPos  [1] =  0.0F;
  vertPos  [2] =     z;

  vertNorms[0] =  0.0F;
  vertNorms[1] =  0.0F;
  vertNorms[2] = -1.0F;

  for ( int i = 8; i >= 0; i-- )
  {
    idx_t  i3 = 3 * (8 - i) + 3;

    vertPos  [i3 + 0] = size * POINTS[i][0];
    vertPos  [i3 + 1] = size * POINTS[i][1];
    vertPos  [i3 + 2] = z;

    vertNorms[i3 + 0] =  0.0F;
    vertNorms[i3 + 1] =  0.0F;
    vertNorms[i3 + 2] = -1.0F;
  }

  colorBottom[0] = COLOR_SCALE * color[0];
  colorBottom[1] = COLOR_SCALE * color[1];
  colorBottom[2] = COLOR_SCALE * color[2];
  colorBottom[3] =               color[3];

  ShaderDraw::draw ( gtx, 30, vertPos, vertNorms, colorBottom,
                     ShaderDraw::TRIANGLE_FAN );

  // Draw arrow line.

  ShaderDraw::draw ( gtx, 6, vertLine, color, ShaderDraw::LINES );
}


JEM_END_PACKAGE( gl )
