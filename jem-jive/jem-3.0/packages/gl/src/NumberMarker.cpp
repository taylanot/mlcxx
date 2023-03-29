
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
#include <jem/base/Integer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/NumberMarker.h>


JEM_DEFINE_CLASS( jem::gl::NumberMarker );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class NumberMarker
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


NumberMarker::NumberMarker ( const Color& color )
{
  color_   =  clamped ( color );
  gtxID_   =  0;
  zPrec_   = -1.0;
  zFunc_   =  GL_LESS;
  counter_ =  0;
}


NumberMarker::~NumberMarker ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String NumberMarker::toString () const
{
  return "number";
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void NumberMarker::configure ( const Properties& props )
{
  writer.font.configure ( PropNames::FONT,  props );
  color_.     configure ( PropNames::COLOR, props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void NumberMarker::getConfig ( const Properties& conf ) const
{
  writer.font.getConfig ( PropNames::FONT,  conf );
  color_     .getConfig ( PropNames::COLOR, conf );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void NumberMarker::listProps ( const Properties& conf ) const
{
  conf.set ( PropNames::FONT,
             "The font to be used for the number markers" );
  conf.set ( PropNames::COLOR,
             "The color of the number markers" );
}


//-----------------------------------------------------------------------
//   setFont
//-----------------------------------------------------------------------


void NumberMarker::setFont ()
{
  writer.font = Font::getDefault ();
}


//-----------------------------------------------------------------------
//   beginDraw
//-----------------------------------------------------------------------


void NumberMarker::beginDraw

  ( GLContext&  gtx,
    const Box&  box,
    idx_t       count )

{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

  GLint          zFunc;

  if ( funcs )
  {
    if ( ! gtx.hasID( gtxID_ ) )
    {
      GLint  zBits = jem::max ( (GLint) 8, getDepthBits( gtx ) );

      gtxID_ = gtx.getID ();
      zPrec_ = (GLfloat) (10.0 * ::pow ( 2.0, (double) -zBits ));
    }

    funcs->glGetParameter ( GL_DEPTH_FUNC, &zFunc );

    zFunc_ = (GLenum) zFunc;

    if ( zFunc_ != GL_ALWAYS )
    {
      funcs->glDepthFunc ( GL_ALWAYS );
    }
  }

  counter_ = 0;

  gtx.pushAttribs ( GLContext::COLOR | GLContext::LIGHTING );
  gtx.setLighting ( false );
  gtx.setColor    ( color_ );
  writer.begin    ( gtx );
  timer_.reset    ();
  timer_.start    ();
}


//-----------------------------------------------------------------------
//   endDraw
//-----------------------------------------------------------------------


void NumberMarker::endDraw ( GLContext& gtx )
{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

  timer_.stop    ();
  writer.end     ( gtx );
  gtx.popAttribs ();

  if ( funcs )
  {
    if ( zFunc_ != GL_ALWAYS )
    {
      funcs->glDepthFunc ( zFunc_ );
    }
  }
}


//-----------------------------------------------------------------------
//   drawMarker
//-----------------------------------------------------------------------


void NumberMarker::drawMarker

  ( GLContext&      gtx,
    idx_t           tag,
    const GLfloat*  pos )

{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

  char           buf[64];

  writer.setCursor ( pos[0], pos[1], pos[2] );

  if ( ! writer.isVisible() )
  {
    return;
  }

  idx_t  k = Integer::print ( buf, (idx_t) sizeof(buf), (lint) tag );

  if      ( zFunc_ == GL_ALWAYS )
  {
    writer.zPos -= zPrec_;
  }
  else if ( funcs )
  {
    GLfloat  depth;
    GLfloat  delta;

    funcs->glReadPixels ( writer.xPos, writer.yPos, 1, 1,
                          GL_DEPTH_COMPONENT, GL_FLOAT, &depth );

    delta = zPrec_ + (GLfloat) std::fabs( 0.001 * depth );

    if ( (writer.zPos - delta) > depth )
    {
      return;
    }

    // As the call to glReadPixels can be quite slow, only call it a
    // limited number of times. After that, turn on the depth test and
    // let OpenGL decide what to draw.

    counter_++;

    if ( counter_ >= 256 )
    {
      counter_ = 0;

      if ( timer_.toDouble() > 0.5  )
      {
        funcs->glDepthFunc ( zFunc_ );

        zFunc_ = GL_ALWAYS;
      }
    }
  }

  writer.writeLine ( gtx, buf, k );
}


//-----------------------------------------------------------------------
//   setColor
//-----------------------------------------------------------------------


void NumberMarker::setColor ( const Color& color )
{
  color_ = color;

  clamp ( color_ );
}


//-----------------------------------------------------------------------
//   getColor
//-----------------------------------------------------------------------


Color NumberMarker::getColor () const
{
  return color_;
}


JEM_END_PACKAGE( gl )
