
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
#include <jem/gl/render.h>
#include <jem/gl/Array.h>
#include <jem/gl/Color.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/ShaderDraw.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   forward declaration of functions
//-----------------------------------------------------------------------


static void   renderDisk_v1

  ( GLContext&  gtx,
    GLfloat     radius,
    int         quality );

static void   renderDisk_v3

  ( GLContext&  gtx,
    GLfloat     radius,
    int         quality );


//-----------------------------------------------------------------------
//   renderDisk
//-----------------------------------------------------------------------


void          renderDisk

  ( GLContext&  gtx,
    GLfloat     radius,
    int         quality )

{
  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    renderDisk_v3 ( gtx, radius, quality );
  }
  else
  {
    renderDisk_v1 ( gtx, radius, quality );
  }
}


//-----------------------------------------------------------------------
//   renderDisk_v1
//-----------------------------------------------------------------------


void          renderDisk_v1

  ( GLContext&  gtx,
    GLfloat     radius,
    int         quality )

{
  JEM_ASSERT ( radius > 0.0F );

  GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

  if ( ! funcs )
  {
    return;
  }

  funcs->glNormal ( 0.0F, 0.0F, 1.0F );
  funcs->glBegin  ( GL_TRIANGLE_FAN );
  funcs->glVertex (   0.0F, 0.0F );
  funcs->glVertex ( radius, 0.0F );

  if ( quality <= 1 )
  {
    const GLfloat  a = radius * 0.866025F;
    const GLfloat  b = radius * 0.500000F;

    funcs->glVertex (       a,       b );
    funcs->glVertex (       b,       a );
    funcs->glVertex (    0.0F,  radius );
    funcs->glVertex (      -b,       a );
    funcs->glVertex (      -a,       b );
    funcs->glVertex ( -radius,    0.0F );
    funcs->glVertex (      -a,      -b );
    funcs->glVertex (      -b,      -a );
    funcs->glVertex (    0.0F, -radius );
    funcs->glVertex (       b,      -a );
    funcs->glVertex (       a,      -b );
  }
  else
  {
    GLfloat  x, y, t;
    int      a, da;

    da = 90 / (quality + 1);
    da = da - (90 % da);

    if ( da < 3 )
    {
      da = 3;
    }

    for ( a = da; a < 360; a += da )
    {
      t = (GLfloat) (RADS_PER_DEGREE * a);
      x = (GLfloat) (radius * std::cos ( t ));
      y = (GLfloat) (radius * std::sin ( t ));

      funcs->glVertex ( x, y );
    }
  }

  funcs->glVertex ( radius, 0.0F );

  funcs->glEnd ();
}


//-----------------------------------------------------------------------
//   renderDisk
//-----------------------------------------------------------------------


void          renderDisk_v3

  ( GLContext&  gtx,
    GLfloat     radius,
    int         quality )

{
  JEM_ASSERT ( radius > 0.0F );

  if ( quality <= 1 )
  {
    const GLfloat  a = radius * 0.866025F;
    const GLfloat  b = radius * 0.500000F;

    GLfloat        vertPos[42] = { 0.0F };
    GLfloat        normals[42] = { 0.0F };

    for ( idx_t i = 0; i < 42; i += 3 )
    {
      normals[i + 2] = 1.0F;
    }

    vertPos[ 0] =  0.0F;   vertPos[ 1] =  0.0F;
    vertPos[ 3] =  radius; vertPos[ 4] =  0.0F;
    vertPos[ 6] =  a;      vertPos[ 7] =  b;
    vertPos[ 9] =  b;      vertPos[10] =  a;
    vertPos[12] =  0.0F;   vertPos[13] =  radius;
    vertPos[15] = -b;      vertPos[16] =  a;
    vertPos[18] = -a;      vertPos[19] =  b;
    vertPos[21] = -radius; vertPos[22] =  0.0F;
    vertPos[24] = -a;      vertPos[25] = -b;
    vertPos[27] = -b;      vertPos[28] = -a;
    vertPos[30] =  0.0F;   vertPos[31] = -radius;
    vertPos[33] =  b;      vertPos[34] = -a;
    vertPos[36] =  a;      vertPos[37] = -b;
    vertPos[39] =  radius; vertPos[40] =  0.0F;

    ShaderDraw::draw ( gtx, 42, vertPos, normals,
                       gtx.getColor (),
                       ShaderDraw::TRIANGLE_FAN );
  }
  else
  {
    GLfloatMatrix  buffer;
    GLfloatVector  vertPos;
    GLfloatVector  normals;
    GLfloat        x, y, t;
    int            a, da;
    int            n;


    da = 90 / (quality + 1);
    da = da - (90 % da);

    if ( da < 3 )
    {
      da = 3;
    }

    n = 3 * (360 / da);

    buffer .resize ( n + 6, 2 );
    vertPos.ref    ( buffer[0] );
    normals.ref    ( buffer[1] );

    vertPos = 0.0F;
    normals = 0.0F;

    for ( idx_t i = 0; i < normals.size(); i += 3 )
    {
      normals[i + 2] = 1.0F;
    }

    vertPos[0] = 0.0F;   vertPos[1] = 0.0F;
    vertPos[3] = radius; vertPos[4] = 0.0F;
    n          = 6;

    for ( a = da; a < 360; a += da )
    {
      t  = (GLfloat) (RADS_PER_DEGREE * (double) a);
      x  = (GLfloat) (radius * std::cos ( t ));
      y  = (GLfloat) (radius * std::sin ( t ));

      vertPos[n]     = x;
      vertPos[n + 1] = y;

      n             += 3;
    }

    vertPos[n] = radius; vertPos[n + 1] = 0.0F;
    n         += 3;

    ShaderDraw::draw ( gtx, vertPos, normals,
                       gtx.getColor (),
                       ShaderDraw::TRIANGLE_FAN );
  }
}


JEM_END_PACKAGE( gl )
