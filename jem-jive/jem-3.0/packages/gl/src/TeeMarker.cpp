
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
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/Box.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/TeeMarker.h>


JEM_DEFINE_CLASS( jem::gl::TeeMarker );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class TeeMarker
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const byte  TeeMarker::INACTIVE_   =    0x0;
const byte  TeeMarker::DRAW_QUADS_ = 1 << 0;
const byte  TeeMarker::DRAW_LINES_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


TeeMarker::TeeMarker

  ( GLfloat       size,
    const Color&  color )

{
  size_    = size;
  angle_   = 0.0F;
  color_   = clamped ( color );

  geom_[0] = 0.0F;
  geom_[1] = 0.0F;
  lwidth_  = 0.0F;
  state_   = INACTIVE_;
}


TeeMarker::~TeeMarker ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String TeeMarker::toString () const
{
  return "tee";
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void TeeMarker::configure ( const Properties& props )
{
  props.find ( size_,  PropNames::SIZE  );
  props.find ( angle_, PropNames::ANGLE );

  color_.configure ( PropNames::COLOR, props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void TeeMarker::getConfig ( const Properties& conf ) const
{
  conf.set ( PropNames::SIZE,  size_  );
  conf.set ( PropNames::ANGLE, angle_ );

  color_.getConfig ( PropNames::COLOR, conf  );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void TeeMarker::listProps ( const Properties& conf ) const
{
  conf.set ( PropNames::SIZE,
             "The size of the tee markers. A negative size is "
             "relative to the object for which the markers are "
             "drawn" );
  conf.set ( PropNames::ANGLE,
             "The rotation angle (in degrees) around the z-axis "
             "(in body coordinates)" );
  conf.set ( PropNames::COLOR,
             "The color of the markers" );
}


//-----------------------------------------------------------------------
//   beginDraw
//-----------------------------------------------------------------------


void TeeMarker::beginDraw

  ( GLContext&  gtx,
    const Box&  box,
    idx_t       count )

{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

  double         alpha;
  double         len;


  state_ = INACTIVE_;

  if      ( size_ < 0.0 )
  {
    len = -size_ * box.size ();
  }
  else if ( size_ > 0.0 )
  {
    len =  size_;
  }
  else
  {
    return;
  }

  alpha    = angle_ * 3.141592653589793 / 180.0;
  geom_[0] = (GLfloat) (0.5 * len * std::cos( alpha ) );
  geom_[1] = (GLfloat) (0.5 * len * std::sin( alpha ) );
  state_  |= DRAW_QUADS_;

  gtx.pushAttribs ( GLContext::COLOR    |
                    GLContext::LIGHTING |
                    GLContext::LINE_WIDTH );
  gtx.setColor    ( color_ );
  gtx.setLighting ( false );

  if ( count > 1000 )
  {
    lwidth_ = 0.0F;
  }
  else
  {
    lwidth_ = gtx.getLineWidth ();
    state_ |= DRAW_LINES_;

    gtx.setLineWidth ( 1.0F );
  }

  if ( funcs )
  {
    blend_ = funcs->glIsEnabled ( GL_BLEND );

    if ( (blend_ == GL_TRUE) && color_.isOpaque() )
    {
      funcs->glDisable ( GL_BLEND );
    }
  }
}


//-----------------------------------------------------------------------
//   endDraw
//-----------------------------------------------------------------------


void TeeMarker::endDraw ( GLContext& gtx )
{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

  if ( state_ )
  {
    gtx.popAttribs ();

    if ( state_ & DRAW_LINES_ )
    {
      gtx.setLineWidth ( lwidth_ );
    }

    if ( funcs )
    {
      if ( blend_ == GL_TRUE )
      {
        funcs->glEnable ( GL_BLEND );
      }
    }

    state_ = INACTIVE_;
  }
}


//-----------------------------------------------------------------------
//   drawMarker
//-----------------------------------------------------------------------


void TeeMarker::drawMarker

  ( GLContext&      gtx,
    idx_t           tag,
    const GLfloat*  pos )

{
  const GLfloat  a     = 0.15F;
  const GLfloat  b     = 1.15F;
  const GLfloat  z     = pos[2];
  GLFunctions1*  funcs = nullptr;

  GLfloat        vertPos[36];

  GLfloat        x1, y1;
  GLfloat        x2, y2;
  GLfloat        dx, dy;


  if ( state_ )
  {
    return;
  }

  if ( ! gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    funcs = gtx.findFunctions<GLFunctions1> ();
  }

  if ( state_ & DRAW_QUADS_ )
  {
    if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
    {
      x1 = pos[0] - geom_[0];
      y1 = pos[1] - geom_[1];
      x2 = pos[0] + geom_[0];
      y2 = pos[1] + geom_[1];
      dx =      a * geom_[1];
      dy =     -a * geom_[0];

      vertPos[ 0] = x1 + dx; vertPos[ 1] = y1 + dy; vertPos[ 2] = z;
      vertPos[ 3] = x2 + dx; vertPos[ 4] = y2 + dy; vertPos[ 5] = z;
      vertPos[ 6] = x2 - dx; vertPos[ 7] = y2 - dy; vertPos[ 8] = z;

      vertPos[ 9] = x2 - dx; vertPos[10] = y2 - dy; vertPos[11] = z;
      vertPos[12] = x1 - dx; vertPos[13] = y1 - dy; vertPos[14] = z;
      vertPos[15] = x1 + dx; vertPos[16] = y1 + dy; vertPos[17] = z;

      x1 = pos[0];
      y1 = pos[1];
      x2 = x1 - b * geom_[1];
      y2 = y1 + b * geom_[0];
      dx =      a * geom_[0];
      dy =      a * geom_[1];

      vertPos[18] = x1 + dx; vertPos[19] = y1 + dy; vertPos[20] = z;
      vertPos[21] = x2 + dx; vertPos[22] = y2 + dy; vertPos[23] = z;
      vertPos[24] = x2 - dx; vertPos[25] = y2 - dy; vertPos[26] = z;

      vertPos[27] = x2 - dx; vertPos[28] = y2 - dy; vertPos[29] = z;
      vertPos[30] = x1 - dx; vertPos[31] = y1 - dy; vertPos[32] = z;
      vertPos[33] = x1 + dx; vertPos[34] = y1 + dy; vertPos[35] = z;

      ShaderDraw::draw ( gtx, 36, vertPos,
                         gtx.getColor (),
                         ShaderDraw::TRIANGLES );
    }
    else if ( funcs )
    {
      funcs->glBegin ( GL_QUADS );
      {
        x1 = pos[0] - geom_[0];
        y1 = pos[1] - geom_[1];
        x2 = pos[0] + geom_[0];
        y2 = pos[1] + geom_[1];
        dx =      a * geom_[1];
        dy =     -a * geom_[0];

        funcs->glVertex ( x1 + dx, y1 + dy, z );
        funcs->glVertex ( x2 + dx, y2 + dy, z );
        funcs->glVertex ( x2 - dx, y2 - dy, z );
        funcs->glVertex ( x1 - dx, y1 - dy, z );

        x1 = pos[0];
        y1 = pos[1];
        x2 = x1 - b * geom_[1];
        y2 = y1 + b * geom_[0];
        dx =      a * geom_[0];
        dy =      a * geom_[1];

        funcs->glVertex ( x1 + dx, y1 + dy, z );
        funcs->glVertex ( x2 + dx, y2 + dy, z );
        funcs->glVertex ( x2 - dx, y2 - dy, z );
        funcs->glVertex ( x1 - dx, y1 - dy, z );
      }
      funcs->glEnd ();
    }
  }

  if ( state_ & DRAW_LINES_ )
  {
    if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
    {
      x1 = pos[0] - geom_[0];
      y1 = pos[1] - geom_[1];
      x2 = pos[0] + geom_[0];
      y2 = pos[1] + geom_[1];

      vertPos[0] = x1; vertPos[ 1] = y1; vertPos[ 2] = z;
      vertPos[3] = x2; vertPos[ 4] = y2; vertPos[ 5] = z;

      x1 = pos[0];
      y1 = pos[1];
      x2 = x1 - b * geom_[1];
      y2 = y1 + b * geom_[0];

      vertPos[6] = x1; vertPos[ 7] = y1; vertPos[ 8] = z;
      vertPos[9] = x2; vertPos[10] = y2; vertPos[11] = z;

      ShaderDraw::draw ( gtx, 12, vertPos,
                         gtx.getColor(), ShaderDraw::LINES );
    }
    else if ( funcs )
    {
      funcs->glBegin ( GL_LINES );
      {
        x1 = pos[0] - geom_[0];
        y1 = pos[1] - geom_[1];
        x2 = pos[0] + geom_[0];
        y2 = pos[1] + geom_[1];

        funcs->glVertex ( x1, y1, z );
        funcs->glVertex ( x2, y2, z );

        x1 = pos[0];
        y1 = pos[1];
        x2 = x1 - b * geom_[1];
        y2 = y1 + b * geom_[0];

        funcs->glVertex ( x1, y1, z );
        funcs->glVertex ( x2, y2, z );
      }
      funcs->glEnd ();
    }
  }
}


//-----------------------------------------------------------------------
//   setColor
//-----------------------------------------------------------------------


void TeeMarker::setColor ( const Color& color )
{
  color_ = clamped ( color );
}


//-----------------------------------------------------------------------
//   getColor
//-----------------------------------------------------------------------


Color TeeMarker::getColor () const
{
  return color_;
}


//-----------------------------------------------------------------------
//   setSize
//-----------------------------------------------------------------------


void TeeMarker::setSize ( GLfloat size )
{
  size_ = size;
}


//-----------------------------------------------------------------------
//   setAngle
//-----------------------------------------------------------------------


void TeeMarker::setAngle ( GLfloat angle )
{
  angle_ = angle;
}


JEM_END_PACKAGE( gl )
