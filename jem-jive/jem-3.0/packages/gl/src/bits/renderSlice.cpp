
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


static void   renderSlice_v1

  ( GLContext&  gtx,
    GLfloat     alpha,
    GLfloat     beta,
    GLfloat     radius,
    int         quality );

static void   renderSlice_v3

  ( GLContext&  gtx,
    GLfloat     alpha,
    GLfloat     beta,
    GLfloat     radius,
    int         quality );


//-----------------------------------------------------------------------
//   renderSlice
//-----------------------------------------------------------------------


void          renderSlice

  ( GLContext&  gtx,
    GLfloat     alpha,
    GLfloat     beta,
    GLfloat     radius,
    int         quality )

{
  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    renderSlice_v3 ( gtx, alpha, beta, radius, quality );
  }
  else
  {
    renderSlice_v1 ( gtx, alpha, beta, radius, quality );
  }
}


//-----------------------------------------------------------------------
//   renderSlice_v1
//-----------------------------------------------------------------------


void          renderSlice_v1

  ( GLContext&  gtx,
    GLfloat     alpha,
    GLfloat     beta,
    GLfloat     radius,
    int         quality )

{
  JEM_ASSERT ( radius > 0.0F );

  GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

  GLfloat        x, y, t, dt;
  int            i, n, da;

  if ( ! funcs )
  {
    return;
  }

  if ( beta <= alpha )
  {
    return;
  }

  da = 90 / (quality + 1);
  da = da - (90 % da);

  if ( da < 3 )
  {
    da = 3;
  }

  n  = (int) ((beta - alpha) / (GLfloat) da);

  if ( n < 1 )
  {
    n = 1;
  }

  dt    = (beta - alpha) / (GLfloat) n;
  dt    = (GLfloat) (dt    * RADS_PER_DEGREE);
  alpha = (GLfloat) (alpha * RADS_PER_DEGREE);

  funcs->glNormal ( 0.0F, 0.0F, 1.0F );
  funcs->glBegin  ( GL_TRIANGLE_FAN );
  funcs->glVertex ( 0.0F, 0.0F );

  for ( i = 0; i <= n; i++ )
  {
    t = alpha + (GLfloat) i * dt;
    x = (GLfloat) (radius * std::cos ( t ));
    y = (GLfloat) (radius * std::sin ( t ));

    funcs->glVertex ( x, y );
  }

  funcs->glEnd ();
}


//-----------------------------------------------------------------------
//   renderSlice_v3
//-----------------------------------------------------------------------


void          renderSlice_v3

  ( GLContext&  gtx,
    GLfloat     alpha,
    GLfloat     beta,
    GLfloat     radius,
    int         quality )

{
  JEM_ASSERT ( radius > 0.0F );

  GLfloat        x, y, t, dt;
  int            i, n, da;

  GLfloatMatrix  buffer;
  GLfloatVector  vertPos;
  GLfloatVector  normals;


  if ( beta <= alpha )
  {
    return;
  }

  da = 90 / (quality + 1);
  da = da - (90 % da);

  if ( da < 3 )
  {
    da = 3;
  }

  n  = (int) ((beta - alpha) / (GLfloat) da);

  if ( n < 1 )
  {
    n = 1;
  }

  buffer .resize ( 3 * (n + 2), 2 );
  vertPos.ref    ( buffer[0] );
  normals.ref    ( buffer[1] );

  dt      = (beta - alpha) / (GLfloat) n;
  dt      = (GLfloat) (dt    * RADS_PER_DEGREE);
  alpha   = (GLfloat) (alpha * RADS_PER_DEGREE);

  // Starts at (0.0F; 0.0F)

  vertPos = 0.0F;
  normals = 0.0F;

  normals[2] = 1.0F;

  for ( i = 0; i <= n; i++ )
  {
    t = alpha + (GLfloat) i * dt;
    x = (GLfloat) (radius * std::cos ( t ));
    y = (GLfloat) (radius * std::sin ( t ));

    vertPos[3*(i+1)]     = x;
    vertPos[3*(i+1) + 1] = y;

    normals[3*(i+1) + 2] = 1.0F;
  }

  ShaderDraw::draw ( gtx, vertPos, normals,
                     gtx.getColor (),
                     ShaderDraw::TRIANGLE_FAN );
}


JEM_END_PACKAGE( gl )
