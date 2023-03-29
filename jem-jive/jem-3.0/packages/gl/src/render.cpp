
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
#include <jem/gl/render.h>
#include <jem/gl/Rect.h>
#include <jem/gl/GLContext.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class CanvasScope
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


CanvasScope::~CanvasScope ()
{
  GLFuncs1Core*  funcs = gtx_.findFunctions<GLFuncs1Core> ();

  gtx_.popTransform ( GLContext::PROJECTION );
  gtx_.popTransform ();
  gtx_.setLighting  ( lighting_ );

  if ( funcs )
  {
    funcs->glDepthFunc ( depthFunc_ );
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void CanvasScope::init_ ( GLfloat clip0, GLfloat clip1 )
{
  GLFuncs1Core*  funcs = gtx_.findFunctions<GLFuncs1Core> ();
  Rect           vport = gtx_.getViewport                 ();

  GLint          zfunc;
  GLdouble       x, y, w, h;

  x          = (GLdouble) vport.x;
  y          = (GLdouble) vport.y;
  w          = (GLdouble) vport.width;
  h          = (GLdouble) vport.height;
  width      = vport.width;
  height     = vport.height;
  nearClip   = clip0;
  farClip    = clip1;
  lighting_  = gtx_.getLighting ();
  depthFunc_ = GL_LESS;

  gtx_.pushTransform ( GLContext::PROJECTION );
  gtx_.setIdentity   ( GLContext::PROJECTION );
  gtx_.setOrtho      ( x, x + w, y, y + h,
                       (GLdouble) clip0, (GLdouble) clip1,
                       GLContext::PROJECTION );
  gtx_.pushTransform ();
  gtx_.setIdentity   ();
  gtx_.setLighting   ( false );

  if ( funcs )
  {
    funcs->glGetParameter ( GL_DEPTH_FUNC, zfunc );
    funcs->glDepthFunc    ( GL_ALWAYS );

    depthFunc_ = (GLenum) zfunc;
  }
}


JEM_END_PACKAGE( gl )
