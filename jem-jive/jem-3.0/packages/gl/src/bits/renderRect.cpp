
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
#include <jem/gl/Rect.h>
#include <jem/gl/Color.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/ShaderDraw.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   renderRect
//-----------------------------------------------------------------------


void           renderRect

  ( GLContext&   gtx,
    GLfloat      x0,
    GLfloat      y0,
    GLfloat      x1,
    GLfloat      y1 )

{
  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    const GLfloat  pos[12] = { x0, y0, 0.0F, x1, y0, 0.0F,
                               x1, y1, 0.0F, x0, y1, 0.0F };

    ShaderDraw::draw ( gtx, 12, pos,
                       gtx.getColor (),
                       ShaderDraw::TRIANGLE_FAN );
  }
  else
  {
    GLFunctions1*  funcs = gtx.findFunctions<GLFunctions1> ();

    if ( funcs )
    {
      funcs->glRect ( x0, y0, x1, y1 );
    }
  }
}


void            renderRect

  ( GLContext&    gtx,
    const Rect&   rect )

{
  renderRect ( gtx,
               (GLfloat)  rect.x,
               (GLfloat)  rect.y,
               (GLfloat) (rect.x + rect.width),
               (GLfloat) (rect.y + rect.height) );
}


JEM_END_PACKAGE( gl )
