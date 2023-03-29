
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
#include <jem/gl/Array.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/ShaderDraw.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   forward declaration of functions
//-----------------------------------------------------------------------


static void   renderShadow_v1

  ( GLContext&  gtx,
    GLfloat     x0,
    GLfloat     y0,
    GLfloat     x1,
    GLfloat     y1,
    GLfloat     w );

static void   renderShadow_v3

  ( GLContext&  gtx,
    GLfloat     x0,
    GLfloat     y0,
    GLfloat     x1,
    GLfloat     y1,
    GLfloat     w );


//-----------------------------------------------------------------------
//   renderShadow
//-----------------------------------------------------------------------


void          renderShadow

  ( GLContext&  gtx,
    GLfloat     x0,
    GLfloat     y0,
    GLfloat     x1,
    GLfloat     y1,
    GLfloat     w )

{
  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    renderShadow_v3 ( gtx, x0, y0, x1, y1, w );
  }
  else
  {
    renderShadow_v1 ( gtx, x0, y0, x1, y1, w );
  }
}


//-----------------------------------------------------------------------
//   renderShadow_v1
//-----------------------------------------------------------------------


void          renderShadow_v1

  ( GLContext&  gtx,
    GLfloat     x0,
    GLfloat     y0,
    GLfloat     x1,
    GLfloat     y1,
    GLfloat     w )

{
  const GLfloat  LIGHT[4] = { 0.0F, 0.0F, 0.0F, 0.0F };
  const GLfloat  DARK [4] = { 0.0F, 0.0F, 0.0F, 0.5F };

  const GLfloat  offset   = 0.4F * w;
  const GLfloat  radius   = 0.8F * w;

  const GLfloat  c1       = radius * 0.965926F;
  const GLfloat  c2       = radius * 0.866025F;
  const GLfloat  c3       = radius * 0.707107F;
  const GLfloat  c4       = radius * 0.500000F;
  const GLfloat  c5       = radius * 0.258819F;
  const GLfloat  c1i      = radius - c1;
  const GLfloat  c2i      = radius - c2;
  const GLfloat  c3i      = radius - c3;
  const GLfloat  c4i      = radius - c4;
  const GLfloat  c5i      = radius - c5;

  GLFunctions1*  funcs    = gtx.findFunctions<GLFunctions1> ();

  if ( ! funcs )
  {
    return;
  }

  funcs->glNormal   ( 0.0F, 0.0F, 1.0F );

  funcs->glBegin    ( GL_QUADS );
  {
    funcs->glColor4 ( DARK );
    funcs->glVertex ( x1 + offset,          y0          );
    funcs->glVertex ( x0 + offset + radius, y0          );
    funcs->glVertex ( x0 + offset + radius, y0 - offset );
    funcs->glVertex ( x1 + offset,          y0 - offset );
  }
  funcs->glEnd ();

  funcs->glBegin    ( GL_QUADS );
  {
    funcs->glColor4 ( DARK  );
    funcs->glVertex ( x1,                   y1 - offset - radius );
    funcs->glVertex ( x1,                   y0                   );
    funcs->glVertex ( x1 + offset,          y0                   );
    funcs->glVertex ( x1 + offset,          y1 - offset - radius );
  }
  funcs->glEnd ();

  funcs->glBegin    ( GL_QUADS );
  {
    funcs->glColor4 ( DARK  );
    funcs->glVertex ( x1 + offset,          y0 - offset          );
    funcs->glVertex ( x0 + offset + radius, y0 - offset          );
    funcs->glColor4 ( LIGHT );
    funcs->glVertex ( x0 + offset + radius, y0 - offset - radius );
    funcs->glVertex ( x1 + offset,          y0 - offset - radius );
  }
  funcs->glEnd ();

  funcs->glBegin    ( GL_QUADS );
  {
    funcs->glColor4 ( DARK  );
    funcs->glVertex ( x1 + offset,          y1 - offset - radius );
    funcs->glVertex ( x1 + offset,          y0 - offset          );
    funcs->glColor4 ( LIGHT );
    funcs->glVertex ( x1 + offset + radius, y0 - offset          );
    funcs->glVertex ( x1 + offset + radius, y1 - offset - radius );
  }
  funcs->glEnd ();

  funcs->glBegin    ( GL_QUADS );
  {
    funcs->glColor4 ( DARK  );
    funcs->glVertex ( x0 + offset + radius, y0 - offset );
    funcs->glVertex ( x0 + offset + radius, y0          );
    funcs->glColor4 ( LIGHT );
    funcs->glVertex ( x0 + offset,          y0          );
    funcs->glVertex ( x0 + offset,          y0 - offset );
  }
  funcs->glEnd ();

  funcs->glBegin    ( GL_QUADS );
  {
    funcs->glColor4 ( DARK  );
    funcs->glVertex ( x1,                   y1 - offset - radius );
    funcs->glVertex ( x1 + offset,          y1 - offset - radius );
    funcs->glColor4 ( LIGHT );
    funcs->glVertex ( x1 + offset,          y1 - offset          );
    funcs->glVertex ( x1,                   y1 - offset          );
  }
  funcs->glEnd ();

  funcs->glBegin    ( GL_TRIANGLE_FAN );
  {
    funcs->glColor4 ( DARK  );
    funcs->glVertex ( x0 + offset + radius, y0 - offset           );
    funcs->glColor4 ( LIGHT );
    funcs->glVertex ( x0 + offset,          y0 - offset           );
    funcs->glVertex ( x0 + offset + c1i,    y0 - offset - c5      );
    funcs->glVertex ( x0 + offset + c2i,    y0 - offset - c4      );
    funcs->glVertex ( x0 + offset + c3i,    y0 - offset - c3      );
    funcs->glVertex ( x0 + offset + c4i,    y0 - offset - c2      );
    funcs->glVertex ( x0 + offset + c5i,    y0 - offset - c1      );
    funcs->glVertex ( x0 + offset + radius, y0 - offset - radius  );
  }
  funcs->glEnd ();

  funcs->glBegin    ( GL_TRIANGLE_FAN );
  {
    funcs->glColor4 ( DARK  );
    funcs->glVertex ( x1 + offset,          y0 - offset          );
    funcs->glColor4 ( LIGHT );
    funcs->glVertex ( x1 + offset,          y0 - offset - radius );
    funcs->glVertex ( x1 + offset + c5,     y0 - offset - c1     );
    funcs->glVertex ( x1 + offset + c4,     y0 - offset - c2     );
    funcs->glVertex ( x1 + offset + c3,     y0 - offset - c3     );
    funcs->glVertex ( x1 + offset + c2,     y0 - offset - c4     );
    funcs->glVertex ( x1 + offset + c1,     y0 - offset - c5     );
    funcs->glVertex ( x1 + offset + radius, y0 - offset          );
  }
  funcs->glEnd ();

  funcs->glBegin    ( GL_TRIANGLE_FAN );
  {
    funcs->glColor4 ( DARK  );
    funcs->glVertex ( x1 + offset,          y1 - offset - radius );
    funcs->glColor4 ( LIGHT );
    funcs->glVertex ( x1 + offset + radius, y1 - offset - radius );
    funcs->glVertex ( x1 + offset + c1,     y1 - offset - c5i    );
    funcs->glVertex ( x1 + offset + c2,     y1 - offset - c4i    );
    funcs->glVertex ( x1 + offset + c3,     y1 - offset - c3i    );
    funcs->glVertex ( x1 + offset + c4,     y1 - offset - c2i    );
    funcs->glVertex ( x1 + offset + c5,     y1 - offset - c1i    );
    funcs->glVertex ( x1 + offset,          y1 - offset          );
  }
  funcs->glEnd ();
}


//-----------------------------------------------------------------------
//   renderShadow_v3
//-----------------------------------------------------------------------


void          renderShadow_v3

  ( GLContext&  gtx,
    GLfloat     x0,
    GLfloat     y0,
    GLfloat     x1,
    GLfloat     y1,
    GLfloat     w )

{
  const GLfloat  DARK_VAL    = 0.5F;
  const GLfloat  LIGHT_VAL   = 0.0F;

  const GLfloat  offset      = 0.4F * w;
  const GLfloat  radius      = 0.8F * w;

  const GLfloat  c1          = radius * 0.965926F;
  const GLfloat  c2          = radius * 0.866025F;
  const GLfloat  c3          = radius * 0.707107F;
  const GLfloat  c4          = radius * 0.500000F;
  const GLfloat  c5          = radius * 0.258819F;
  const GLfloat  c1i         = radius - c1;
  const GLfloat  c2i         = radius - c2;
  const GLfloat  c3i         = radius - c3;
  const GLfloat  c4i         = radius - c4;
  const GLfloat  c5i         = radius - c5;

  GLfloat        vertPos[24] = { 0.0F };
  GLfloat        colors [32] = { 0.0F };

  // First quad.

  vertPos[ 0] = x1 + offset;
  vertPos[ 1] = y0;
  colors [ 3] = DARK_VAL;

  vertPos[ 3] = x0 + offset + radius;
  vertPos[ 4] = y0;
  colors [ 7] = DARK_VAL;

  vertPos[ 6] = x0 + offset + radius;
  vertPos[ 7] = y0 - offset;
  colors [11] = DARK_VAL;

  vertPos[ 9] = x1 + offset;
  vertPos[10] = y0 - offset;
  colors [15] = DARK_VAL;

  ShaderDraw::draw ( gtx, 12, vertPos, colors,
                     ShaderDraw::TRIANGLE_FAN );

  // Second quad.

  vertPos[ 0] = x1;
  vertPos[ 1] = y1 - offset - radius;
  colors [ 3] = DARK_VAL;

  vertPos[ 3] = x1;
  vertPos[ 4] = y0;
  colors [ 7] = DARK_VAL;

  vertPos[ 6] = x1 + offset;
  vertPos[ 7] = y0;
  colors [11] = DARK_VAL;

  vertPos[ 9] = x1 + offset;
  vertPos[10] = y1 - offset - radius;
  colors [15] = DARK_VAL;

  ShaderDraw::draw ( gtx, 12, vertPos, colors,
                     ShaderDraw::TRIANGLE_FAN );

  // Third quad.

  vertPos[ 0] = x1 + offset;
  vertPos[ 1] = y0 - offset;
  colors [ 3] = DARK_VAL;

  vertPos[ 3] = x0 + offset + radius;
  vertPos[ 4] = y0 - offset;
  colors [ 7] = DARK_VAL;

  vertPos[ 6] = x0 + offset + radius;
  vertPos[ 7] = y0 - offset - radius;
  colors [11] = LIGHT_VAL;

  vertPos[ 9] = x1 + offset;
  vertPos[10] = y0 - offset - radius;
  colors [15] = LIGHT_VAL;

  ShaderDraw::draw ( gtx, 12, vertPos, colors,
                     ShaderDraw::TRIANGLE_FAN );

  // Fourth quad.

  vertPos[ 0] = x1 + offset;
  vertPos[ 1] = y1 - offset - radius;
  colors [ 3] = DARK_VAL;

  vertPos[ 3] = x1 + offset;
  vertPos[ 4] = y0 - offset;
  colors [ 7] = DARK_VAL;

  vertPos[ 6] = x1 + offset + radius;
  vertPos[ 7] = y0 - offset;
  colors [11] = LIGHT_VAL;

  vertPos[ 9] = x1 + offset + radius;
  vertPos[10] = y1 - offset - radius;
  colors [15] = LIGHT_VAL;

  ShaderDraw::draw ( gtx, 12, vertPos, colors,
                     ShaderDraw::TRIANGLE_FAN );

  // Fifth quad.

  vertPos[ 0] = x0 + offset + radius;
  vertPos[ 1] = y0 - offset;
  colors [ 3] = DARK_VAL;

  vertPos[ 3] = x0 + offset + radius;
  vertPos[ 4] = y0;
  colors [ 7] = DARK_VAL;

  vertPos[ 6] = x0 + offset;
  vertPos[ 7] = y0;
  colors [11] = LIGHT_VAL;

  vertPos[ 9] = x0 + offset;
  vertPos[10] = y0 - offset;
  colors [15] = LIGHT_VAL;

  ShaderDraw::draw ( gtx, 12, vertPos, colors,
                     ShaderDraw::TRIANGLE_FAN );

  // Sixth quad.

  vertPos[ 0] = x1;
  vertPos[ 1] = y1 - offset - radius;
  colors [ 3] = DARK_VAL;

  vertPos[ 3] = x1 + offset;
  vertPos[ 4] = y1 - offset - radius;
  colors [ 7] = DARK_VAL;

  vertPos[ 6] = x1 + offset;
  vertPos[ 7] = y1 - offset;
  colors [11] = LIGHT_VAL;

  vertPos[ 9] = x1;
  vertPos[10] = y1 - offset;
  colors [15] = LIGHT_VAL;

  ShaderDraw::draw ( gtx, 12, vertPos, colors,
                     ShaderDraw::TRIANGLE_FAN );

  // First triangle fan.

  vertPos[ 0] = x0 + offset + radius;
  vertPos[ 1] = y0 - offset;
  colors [ 3] = DARK_VAL;

  vertPos[ 3] = x0 + offset;
  vertPos[ 4] = y0 - offset;
  colors [ 7] = LIGHT_VAL;

  vertPos[ 6] = x0 + offset + c1i;
  vertPos[ 7] = y0 - offset - c5;
  colors [11] = LIGHT_VAL;

  vertPos[ 9] = x0 + offset + c2i;
  vertPos[10] = y0 - offset - c4;
  colors [15] = LIGHT_VAL;

  vertPos[12] = x0 + offset + c3i;
  vertPos[13] = y0 - offset - c3;
  colors [19] = LIGHT_VAL;

  vertPos[15] = x0 + offset + c4i;
  vertPos[16] = y0 - offset - c2;
  colors [23] = LIGHT_VAL;

  vertPos[18] = x0 + offset + c5i;
  vertPos[19] = y0 - offset - c1;
  colors [27] = LIGHT_VAL;

  vertPos[21] = x0 + offset + radius;
  vertPos[22] = y0 - offset - radius;
  colors [31] = LIGHT_VAL;

  ShaderDraw::draw ( gtx, 24, vertPos, colors,
                     ShaderDraw::TRIANGLE_FAN );

  // Second triangle fan.

  vertPos[ 0] = x1 + offset;
  vertPos[ 1] = y0 - offset;
  colors [ 3] = DARK_VAL;

  vertPos[ 3] = x1 + offset;
  vertPos[ 4] = y0 - offset - radius;
  colors [ 7] = LIGHT_VAL;

  vertPos[ 6] = x1 + offset + c5;
  vertPos[ 7] = y0 - offset - c1;
  colors [11] = LIGHT_VAL;

  vertPos[ 9] = x1 + offset + c4;
  vertPos[10] = y0 - offset - c2;
  colors [15] = LIGHT_VAL;

  vertPos[12] = x1 + offset + c3;
  vertPos[13] = y0 - offset - c3;
  colors [19] = LIGHT_VAL;

  vertPos[15] = x1 + offset + c2;
  vertPos[16] = y0 - offset - c4;
  colors [23] = LIGHT_VAL;

  vertPos[18] = x1 + offset + c1;
  vertPos[19] = y0 - offset - c5;
  colors [27] = LIGHT_VAL;

  vertPos[21] = x1 + offset + radius;
  vertPos[22] = y0 - offset;
  colors [31] = LIGHT_VAL;

  ShaderDraw::draw ( gtx, 24, vertPos, colors,
                     ShaderDraw::TRIANGLE_FAN );

  // Third triangle fan.

  vertPos[ 0] = x1 + offset;
  vertPos[ 1] = y1 - offset - radius;
  colors [ 3] = DARK_VAL;

  vertPos[ 3] = x1 + offset + radius;
  vertPos[ 4] = y1 - offset - radius;
  colors [ 7] = LIGHT_VAL;

  vertPos[ 6] = x1 + offset + c1;
  vertPos[ 7] = y1 - offset - c5i;
  colors [11] = LIGHT_VAL;

  vertPos[ 9] = x1 + offset + c2;
  vertPos[10] = y1 - offset - c4i;
  colors [15] = LIGHT_VAL;

  vertPos[12] = x1 + offset + c3;
  vertPos[13] = y1 - offset - c3i;
  colors [19] = LIGHT_VAL;

  vertPos[15] = x1 + offset + c4;
  vertPos[16] = y1 - offset - c2i;
  colors [23] = LIGHT_VAL;

  vertPos[18] = x1 + offset + c5;
  vertPos[19] = y1 - offset - c1i;
  colors [27] = LIGHT_VAL;

  vertPos[21] = x1 + offset;
  vertPos[22] = y1 - offset;
  colors [31] = LIGHT_VAL;

  ShaderDraw::draw ( gtx, 24, vertPos, colors,
                     ShaderDraw::TRIANGLE_FAN );
}


JEM_END_PACKAGE( gl )
