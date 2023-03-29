
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


#include <jem/gl/render.h>
#include <jem/gl/Box.h>
#include <jem/gl/Color.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/ShaderDraw.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   renderBoxEdges
//-----------------------------------------------------------------------


void            renderBoxEdges

  ( GLContext&    gtx,
    const Box&    box )

{
  const Transform*  tr      = nullptr;
  const int         rank    = box.rank ();
  idx_t             count   = 0;
  GLfloat           pos[72] = { 0.0F };

  Vertex3f          a, b;


  if ( rank == 0 )
  {
    return;
  }

  tr = box.getDrawData ( a, b );

  if ( tr )
  {
    gtx.pushTransform ();
    gtx.transform     ( *tr );
  }

  switch ( rank )
  {
  case 1:

    if ( b[0] > a[0] )
    {
      GLfloat*  part = pos + count;

      count  += 6;

      part[0] = a[0];
      part[3] = b[0];
    }

    break;

  case 2:

    if ( b[0] > a[0] )
    {
      GLfloat*  part = pos + count;

      count  += 12;

      part[0] = a[0]; part[ 1] = a[1];
      part[3] = b[0]; part[ 4] = a[1];

      part[6] = a[0]; part[ 7] = b[1];
      part[9] = b[0]; part[10] = b[1];
    }

    if ( b[1] > a[1] )
    {
      GLfloat*  part = pos + count;

      count  += 12;

      part[0] = a[0]; part[ 1] = a[1];
      part[3] = a[0]; part[ 4] = b[1];

      part[6] = b[0]; part[ 7] = a[1];
      part[9] = b[0]; part[10] = b[1];
    }

    break;

  case 3:

    if ( b[0] > a[0] )
    {
      GLfloat*  part = pos + count;

      count   += 24;

      part[ 0] = a[0]; part[ 1] = a[1]; part[ 3] = a[2];
      part[ 3] = b[0]; part[ 4] = a[1]; part[ 5] = a[2];

      part[ 6] = a[0]; part[ 7] = b[1]; part[ 8] = a[2];
      part[ 9] = b[0]; part[10] = b[1]; part[11] = a[2];

      part[12] = a[0]; part[13] = a[1]; part[14] = b[2];
      part[15] = b[0]; part[16] = a[1]; part[17] = b[2];

      part[18] = a[0]; part[19] = b[1]; part[20] = b[2];
      part[21] = b[0]; part[22] = b[1]; part[23] = b[2];
    }

    if ( b[1] > a[1] )
    {
      GLfloat*  part = pos + count;

      count   += 24;

      part[ 0] = a[0]; part[ 1] = a[1]; part[ 3] = a[2];
      part[ 3] = a[0]; part[ 4] = b[1]; part[ 5] = a[2];

      part[ 6] = b[0]; part[ 7] = a[1]; part[ 8] = a[2];
      part[ 9] = b[0]; part[10] = b[1]; part[11] = a[2];

      part[12] = a[0]; part[13] = a[1]; part[14] = b[2];
      part[15] = a[0]; part[16] = b[1]; part[17] = b[2];

      part[18] = b[0]; part[19] = a[1]; part[20] = b[2];
      part[21] = b[0]; part[22] = b[1]; part[23] = b[2];
    }

    if ( b[2] > a[2] )
    {
      GLfloat*  part = pos + count;

      count   += 24;

      part[ 0] = a[0]; part[ 1] = a[1]; part[ 3] = a[2];
      part[ 3] = a[0]; part[ 4] = a[1]; part[ 5] = b[2];

      part[ 6] = a[0]; part[ 7] = b[1]; part[ 8] = a[2];
      part[ 9] = a[0]; part[10] = b[1]; part[11] = b[2];

      part[12] = b[0]; part[13] = a[1]; part[14] = a[2];
      part[15] = b[0]; part[16] = a[1]; part[17] = b[2];

      part[18] = b[0]; part[19] = b[1]; part[20] = a[2];
      part[21] = b[0]; part[22] = b[1]; part[23] = b[2];
    }

    break;
  }

  if      ( ! count )
  {
    // Do nothing
  }
  else if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, count, pos,
                       gtx.getColor (),
                       ShaderDraw::LINES );
  }
  else
  {
    GLFuncs1*  funcs = gtx.findFunctions<GLFuncs1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_LINES );

      for ( idx_t i = 0; i < count; i += 3 )
      {
        funcs->glVertex3 ( pos + i );
      }

      funcs->glEnd ();
    }
  }

  if ( tr )
  {
    gtx.popTransform ();
  }
}


//-----------------------------------------------------------------------
//   renderBoxFaces
//-----------------------------------------------------------------------


void            renderBoxFaces

  ( GLContext&    gtx,
    const Box&    box )

{
  const Transform*  tr       = nullptr;
  const int         rank     = box.rank ();
  idx_t             count    = 0;
  GLfloat           pos[108] = { 0.0F };
  GLfloat           dir[108] = { 0.0F };

  Vertex3f          a, b;


  if ( rank < 2 )
  {
    return;
  }

  tr = box.getDrawData ( a, b );

  if ( tr )
  {
    gtx.pushTransform ();
    gtx.transform     ( *tr );
  }

  switch ( rank )
  {
  case 2:

    if ( b[0] > a[0] && b[1] > a[1] )
    {
      GLfloat*  partPos = pos + count;
      GLfloat*  partDir = dir + count;

      count      += 18;

      partPos[ 0] = a[0]; partPos[ 1] = a[1];
      partPos[ 3] = b[0]; partPos[ 4] = a[1];
      partPos[ 6] = b[0]; partPos[ 7] = b[1];

      partPos[ 9] = b[0]; partPos[10] = b[1];
      partPos[12] = a[0]; partPos[13] = b[1];
      partPos[15] = a[0]; partPos[16] = a[1];

      partDir[ 2] = 1.0F;
      partDir[ 5] = 1.0F;
      partDir[ 8] = 1.0F;
      partDir[11] = 1.0F;
      partDir[14] = 1.0F;
      partDir[17] = 1.0F;
    }

    break;

  case 3:

    if ( b[0] > a[0] && b[2] > a[2] )
    {
      GLfloat*  partPos = pos + count;
      GLfloat*  partDir = dir + count;

      count      +=  36;

      partPos[ 0] = a[0]; partPos[ 1] = a[1]; partPos[ 2] = a[2];
      partPos[ 3] = b[0]; partPos[ 4] = a[1]; partPos[ 5] = a[2];
      partPos[ 6] = b[0]; partPos[ 7] = a[1]; partPos[ 8] = b[2];

      partPos[ 9] = b[0]; partPos[10] = a[1]; partPos[11] = b[2];
      partPos[12] = a[0]; partPos[13] = a[1]; partPos[14] = b[2];
      partPos[15] = a[0]; partPos[16] = a[1]; partPos[17] = a[2];

      partDir[ 1] = -1.0F;
      partDir[ 4] = -1.0F;
      partDir[ 7] = -1.0F;
      partDir[10] = -1.0F;
      partDir[13] = -1.0F;
      partDir[16] = -1.0F;

      partPos[18] = a[0]; partPos[19] = b[1]; partPos[20] = a[2];
      partPos[21] = a[0]; partPos[22] = b[1]; partPos[23] = b[2];
      partPos[24] = b[0]; partPos[25] = b[1]; partPos[26] = b[2];

      partPos[27] = b[0]; partPos[28] = b[1]; partPos[29] = b[2];
      partPos[30] = b[0]; partPos[31] = b[1]; partPos[32] = a[2];
      partPos[33] = a[0]; partPos[34] = b[1]; partPos[35] = a[2];

      partDir[19] =  1.0F;
      partDir[22] =  1.0F;
      partDir[25] =  1.0F;
      partDir[28] =  1.0F;
      partDir[31] =  1.0F;
      partDir[34] =  1.0F;
    }

    if ( b[0] > a[0] && b[1] > a[1] )
    {
      GLfloat*  partPos = pos + count;
      GLfloat*  partDir = dir + count;

      count      +=  36;

      partPos[ 0] = a[0]; partPos[ 1] = a[1]; partPos[ 2] = a[2];
      partPos[ 3] = a[0]; partPos[ 4] = b[1]; partPos[ 5] = a[2];
      partPos[ 6] = b[0]; partPos[ 7] = b[1]; partPos[ 8] = a[2];

      partPos[ 9] = b[0]; partPos[10] = b[1]; partPos[11] = a[2];
      partPos[12] = b[0]; partPos[13] = a[1]; partPos[14] = a[2];
      partPos[15] = a[0]; partPos[16] = a[1]; partPos[17] = a[2];

      partDir[ 2] = -1.0F;
      partDir[ 5] = -1.0F;
      partDir[ 8] = -1.0F;
      partDir[11] = -1.0F;
      partDir[14] = -1.0F;
      partDir[17] = -1.0F;

      partPos[18] = a[0]; partPos[19] = a[1]; partPos[20] = b[2];
      partPos[21] = b[0]; partPos[22] = a[1]; partPos[23] = b[2];
      partPos[24] = b[0]; partPos[25] = b[1]; partPos[26] = b[2];

      partPos[27] = b[0]; partPos[28] = b[1]; partPos[29] = b[2];
      partPos[30] = a[0]; partPos[31] = b[1]; partPos[32] = b[2];
      partPos[33] = a[0]; partPos[34] = a[1]; partPos[35] = b[2];

      partDir[20] = 1.0F;
      partDir[23] = 1.0F;
      partDir[26] = 1.0F;
      partDir[29] = 1.0F;
      partDir[32] = 1.0F;
      partDir[35] = 1.0F;
    }

    if ( b[1] > a[1] && b[2] > a[2] )
    {
      GLfloat*  partPos = pos + count;
      GLfloat*  partDir = dir + count;

      count      +=  36;

      partPos[ 0] = b[0]; partPos[ 1] = a[1]; partPos[ 2] = a[2];
      partPos[ 3] = b[0]; partPos[ 4] = b[1]; partPos[ 5] = a[2];
      partPos[ 6] = b[0]; partPos[ 7] = b[1]; partPos[ 8] = b[2];

      partPos[ 9] = b[0]; partPos[10] = b[1]; partPos[11] = b[2];
      partPos[12] = b[0]; partPos[13] = a[1]; partPos[14] = b[2];
      partPos[15] = b[0]; partPos[16] = a[1]; partPos[17] = a[2];

      partDir[ 0] = 1.0F;
      partDir[ 3] = 1.0F;
      partDir[ 6] = 1.0F;
      partDir[ 9] = 1.0F;
      partDir[12] = 1.0F;
      partDir[15] = 1.0F;

      partPos[18] = a[0]; partPos[19] = a[1]; partPos[20] = a[2];
      partPos[21] = a[0]; partPos[22] = a[1]; partPos[23] = b[2];
      partPos[24] = a[0]; partPos[25] = b[1]; partPos[26] = b[2];

      partPos[27] = a[0]; partPos[28] = b[1]; partPos[29] = b[2];
      partPos[30] = a[0]; partPos[31] = b[1]; partPos[32] = a[2];
      partPos[33] = a[0]; partPos[34] = a[1]; partPos[35] = a[2];

      partDir[18] = -1.0F;
      partDir[21] = -1.0F;
      partDir[24] = -1.0F;
      partDir[27] = -1.0F;
      partDir[30] = -1.0F;
      partDir[33] = -1.0F;
    }

    break;
  }

  if      ( ! count )
  {
    // Do nothing
  }
  else if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    ShaderDraw::draw ( gtx, count, pos, dir,
                       gtx.getColor (),
                       ShaderDraw::TRIANGLES );
  }
  else
  {
    GLFuncs1*  funcs = gtx.findFunctions<GLFuncs1> ();

    if ( funcs )
    {
      funcs->glBegin ( GL_TRIANGLES );

      for ( idx_t i = 0; i < count; i += 3 )
      {
        funcs->glNormal3 ( dir + i );
        funcs->glVertex3 ( pos + i );
      }

      funcs->glEnd ();
    }
  }

  if ( tr )
  {
    gtx.popTransform ();
  }
}


JEM_END_PACKAGE( gl )

