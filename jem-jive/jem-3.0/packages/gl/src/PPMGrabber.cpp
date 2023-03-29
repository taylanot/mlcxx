
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
#include <jem/base/ClassTemplate.h>
#include <jem/io/OutputStream.h>
#include <jem/util/Flex.h>
#include <jem/gl/Rect.h>
#include <jem/gl/GrabScope.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/PPMGrabber.h>


JEM_DEFINE_CLASS( jem::gl::PPMGrabber );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class PPMGrabber
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


PPMGrabber::PPMGrabber ()
{}


PPMGrabber::~PPMGrabber ()
{}


//-----------------------------------------------------------------------
//   grabFrame
//-----------------------------------------------------------------------


void PPMGrabber::grabFrame

  ( OutputStream&   out,
    RenderContext&  rtx,
    GfxNode&        scene,
    Rect&           rect )

{
  GrabScope  scope ( rtx.getGLContext() );

  scope.drawFrame  ( rtx, scene );
  scope.adjustRect ( rect );
  saveFrame        ( out, rtx, rect );
}


//-----------------------------------------------------------------------
//   saveFrame
//-----------------------------------------------------------------------


void PPMGrabber::saveFrame

  ( OutputStream&   out,
    RenderContext&  rtx,
    const Rect&     rect )

{
  using jem::util::Flex;

  JEM_PRECHECK ( rect.isValid() );

  GLContext&     gtx   = rtx.getGLContext ();
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

  if ( funcs )
  {
    const idx_t  w      = (idx_t) rect.width;
    const idx_t  h      = (idx_t) rect.height;

    String       header = String::format ( "P6\n%d %d\n255\n", w, h );

    Flex<byte>   buf    ( 3 * w * h );

    funcs->glReadPixels ( rect.x, rect.y, rect.width, rect.height,
                          GL_RGB, GL_UNSIGNED_BYTE, buf.addr() );

    out.write ( header.addr(), header.size() );

    for ( idx_t i = h - 1; i >= 0; i-- )
    {
      out.write ( buf.addr( i * 3 * w ), 3 * w );
    }
  }
}


JEM_END_PACKAGE( gl )
