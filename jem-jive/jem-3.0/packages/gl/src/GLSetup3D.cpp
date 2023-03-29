
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


#include <jem/gl/utilities.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/GLFunctions1_1.h>
#include <jem/gl/GLVersion.h>
#include <jem/gl/GLSetup3D.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLSetup3D
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


GLSetup3D::GLSetup3D ()
{}


GLSetup3D::~GLSetup3D ()
{}


//-----------------------------------------------------------------------
//   initGL
//-----------------------------------------------------------------------


void GLSetup3D::initGL ( GLContext& gtx )
{
  GLFuncs1_1Core*  funcs = gtx.findFunctions<GLFuncs1_1Core> ();

  Super::initGL ( gtx );

  if ( funcs )
  {
    funcs->glEnable        ( GL_DEPTH_TEST );
    funcs->glPolygonOffset ( 1.0, 1.0 );

    if ( ! gtx.checkVersion( GLVersion::VERSION_3_3 ) )
    {
#ifdef JEM_OS_WINDOWS
      funcs->glEnable ( GL_NORMALIZE );
#else
      funcs->glEnable ( GL_RESCALE_NORMAL );
#endif
    }
  }

  checkGLError ( gtx, JEM_FUNC );
}


JEM_END_PACKAGE( gl )
