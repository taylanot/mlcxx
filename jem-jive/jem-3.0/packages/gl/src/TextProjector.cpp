
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


#include <jem/gl/GLContext.h>
#include <jem/gl/TextProjector.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class TextProjector
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


TextProjector::TextProjector

  ( const Font&   font,
    Alignment     hAlign,
    Alignment     vAlign ) :

    Super ( font, hAlign, vAlign )

{
  proj_.setIdentity ();

  active_ = false;
}


//-----------------------------------------------------------------------
//   begin
//-----------------------------------------------------------------------


void TextProjector::begin ( GLContext& gtx )
{
  Transform  p, t;

  if ( active_ )
  {
    end ( gtx );
  }

  vport_ = gtx.getViewport  ();
  p      = gtx.getTransform ( GLContext::PROJECTION );
  t      = gtx.getTransform ();

  matmul            ( proj_, p, t );
  gtx.pushTransform ( GLContext::PROJECTION );
  gtx.setOrtho      ( vport_, 0.0, -1.0,
                      GLContext::PROJECTION );
  gtx.pushTransform ();
  gtx.setIdentity   ();

  Super::begin      ( gtx );

  active_ = true;
}


void TextProjector::begin

  ( GLContext&        gtx,
    const Transform&  proj )

{
  if ( active_ )
  {
    end ( gtx );
  }

  vport_ = gtx.getViewport ();

  gtx.pushTransform ( GLContext::PROJECTION );
  gtx.setOrtho      ( vport_, 0.0, -1.0,
                      GLContext::PROJECTION );
  gtx.pushTransform ();
  gtx.setIdentity   ();

  Super::begin      ( gtx );

  proj_   = proj;
  active_ = true;
}


//-----------------------------------------------------------------------
//   end
//-----------------------------------------------------------------------


void TextProjector::end ( GLContext& gtx )
{
  if ( active_ )
  {
    Super::end       ( gtx );

    gtx.popTransform ( GLContext::PROJECTION );
    gtx.popTransform ();

    active_ = false;
  }
}


//-----------------------------------------------------------------------
//   setCursor
//-----------------------------------------------------------------------


void TextProjector::setCursor

  ( GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  Vertex3d  v ( x, y, z );
  Vertex3d  w;

  // Project the cursor position to device independent coordinates.

  if ( project( w, proj_, v ) )
  {
    // Transform the cursor position to window coordinates.

    w[0] = 0.5 * (w[0] + 1.0) * vport_.width;
    w[1] = 0.5 * (w[1] + 1.0) * vport_.height;
    w[2] = 0.5 * (w[2] + 1.0);

    xPos = vport_.x + (GLint) w[0];
    yPos = vport_.y + (GLint) w[1];
    zPos = (GLfloat) w[2];
  }
  else
  {
    // Make the cursor invisible.

    xPos = -1;
    yPos = -1;
    zPos = -1.0;
  }
}


JEM_END_PACKAGE( gl )
