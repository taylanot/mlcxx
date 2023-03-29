
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


#include <jem/base/limits.h>
#include <jem/gl/Rect.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/GLVersion.h>
#include <jem/gl/StdGLSetup.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class StdGLSetup
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdGLSetup::StdGLSetup ()
{
  zscaleSet_ = false;
}


StdGLSetup::~StdGLSetup ()
{}


//-----------------------------------------------------------------------
//   initGL
//-----------------------------------------------------------------------


void StdGLSetup::initGL ( GLContext& gtx )
{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();


  zscaleSet_ = false;

  gtx.setAlphaTest ( 0.05F );

  if ( funcs )
  {
    // Clear the current error flag. If it has been set, some external
    // code must responsible.

    for ( int i = 0; i < 10; i++ )
    {
      GLenum  err = funcs->glGetError ();

      if ( err == GL_NO_ERROR )
      {
        break;
      }
    }

    funcs->glEnable      ( GL_BLEND );
    funcs->glBlendFunc   ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    funcs->glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL );
  }

  if ( ! gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLFuncs1*  funcs1 = gtx.findFunctions<GLFuncs1> ();

    // The fixed pipeline of OpenGL 1.1 is used.

    if ( funcs1 )
    {
      funcs1->glColorMaterial ( GL_FRONT_AND_BACK,
                                GL_AMBIENT_AND_DIFFUSE );
      funcs1->glEnable        ( GL_COLOR_MATERIAL );

      // Make sure that the OpenGL matrices are sane.

      funcs1->glMatrixMode    ( GL_TEXTURE );
      funcs1->glLoadIdentity  ();
      funcs1->glMatrixMode    ( GL_PROJECTION );
      funcs1->glLoadIdentity  ();
      funcs1->glMatrixMode    ( GL_MODELVIEW );
      funcs1->glLoadIdentity  ();
    }
  }

  checkGLError ( gtx, JEM_FUNC );
}


//-----------------------------------------------------------------------
//   initView
//-----------------------------------------------------------------------


void StdGLSetup::initView

  ( GLContext&   gtx,
    const Rect&  r )

{
  GLFuncs1*  funcs = gtx.findFunctions<GLFuncs1> ();

  if ( ! funcs )
  {
    return;
  }

  if ( ! zscaleSet_ && r.width > 0 && r.height > 0 )
  {
    GLboolean  ztest = funcs->glIsEnabled ( GL_DEPTH_TEST );

    if ( ztest == GL_TRUE )
    {
      // This is a work-around for a stupid bug in ATI's implementation
      // of glReadPixels.

      GLfloat  d;

      funcs->glClear      ( GL_DEPTH_BUFFER_BIT );
      funcs->glFlush      ();
      funcs->glReadPixels ( 0, 0, 1, 1, GL_DEPTH_COMPONENT,
                            GL_FLOAT, & d );

      if ( ! gtx.checkVersion( GLVersion::VERSION_3_3 ) &&
           ! isTiny( d - 1.0 ) )
      {
        funcs->glPixelTransfer ( GL_DEPTH_SCALE, 1.0F / d );
      }

      zscaleSet_ = true;
    }
  }

  checkGLError ( gtx, JEM_FUNC );
}


JEM_END_PACKAGE( gl )
