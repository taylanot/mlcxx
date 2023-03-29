
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


#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/render.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/PointMarker.h>


JEM_DEFINE_CLASS( jem::gl::PointMarker );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class PointMarker
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const GLfloat  PointMarker::MAX_SIZE = 10.0;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PointMarker::PointMarker

  ( GLfloat       size,
    const Color&  color )

{
  size_   = clamped ( size, MAX_SIZE );
  color_  = clamped ( color );
  active_ = false;
}


PointMarker::~PointMarker ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String PointMarker::toString () const
{
  return "point";
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void PointMarker::configure ( const Properties& props )
{
  double  sz;

  if ( props.find( sz, PropNames::SIZE, 0.0, (double) MAX_SIZE ) )
  {
    size_ = (GLfloat) sz;
  }

  color_.configure ( PropNames::COLOR, props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void PointMarker::getConfig ( const Properties& conf ) const
{
  conf.set ( PropNames::SIZE, (double) size_ );

  color_.getConfig ( PropNames::COLOR, conf );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void PointMarker::listProps ( const Properties& conf ) const
{
  conf.set ( PropNames::SIZE,
             "The size (int pixels) of the point markers" );
  conf.set ( PropNames::COLOR,
             "The color of the point markers" );
}


//-----------------------------------------------------------------------
//   beginDraw
//-----------------------------------------------------------------------


void PointMarker::beginDraw

  ( GLContext&  gtx,
    const Box&  box,
    idx_t       count )

{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

  if ( funcs && (size_ > 0.1F) )
  {
    active_ = true;
  }
  else
  {
    active_ = false;
  }

  coords_.clear   ();
  coords_.reserve ( 3 * count );
}


//-----------------------------------------------------------------------
//   endDraw
//-----------------------------------------------------------------------


void PointMarker::endDraw ( GLContext& gtx )
{
  if ( active_ )
  {
    GLFuncs1Core&   funcs  = gtx.getFunctions<GLFuncs1Core> ();
    const idx_t     count  = coords_.size      ();
    const GLfloat*  coords = coords_.addr      ();
    GLboolean       blend0 = funcs.glIsEnabled ( GL_BLEND );

    AttribScope     scope  ( gtx, GLContext::COLOR    |
                                  GLContext::LIGHTING |
                                  GLContext::POINT_SIZE );

    gtx.setColor     ( color_ );
    gtx.setLighting  ( false );
    gtx.setPointSize ( size_ );

    if ( (blend0 == GL_TRUE) && color_.isOpaque() )
    {
      funcs.glDisable ( GL_BLEND );
    }

    if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
    {
      ShaderDraw::draw ( gtx, count, coords,
                         color_, ShaderDraw::POINTS );
    }
    else
    {
      GLFuncs1*  funcs1 = gtx.findFunctions<GLFuncs1> ();

      if ( funcs1 )
      {
        funcs1->glBegin ( GL_POINTS );

        for ( idx_t i = 0; i < count; i += 3 )
        {
          funcs1->glVertex3 ( coords + i );
        }

        funcs1->glEnd ();
      }
    }

    if ( blend0 == GL_TRUE )
    {
      funcs.glEnable ( GL_BLEND );
    }
  }
}


//-----------------------------------------------------------------------
//   drawMarker
//-----------------------------------------------------------------------


void PointMarker::drawMarker

  ( GLContext&      gtx,
    idx_t           tag,
    const GLfloat*  pos )

{
  if ( active_ )
  {
    GLfloat*  coords = coords_.extend ( 3 );

    coords[0] = pos[0];
    coords[1] = pos[1];
    coords[2] = pos[2];
  }
}


//-----------------------------------------------------------------------
//   setColor
//-----------------------------------------------------------------------


void PointMarker::setColor ( const Color& color )
{
  color_ = clamped ( color );
}


//-----------------------------------------------------------------------
//   getColor
//-----------------------------------------------------------------------


Color PointMarker::getColor () const
{
  return color_;
}


//-----------------------------------------------------------------------
//   setSize
//-----------------------------------------------------------------------


void PointMarker::setSize ( GLfloat size )
{
  size_ = clamped ( size, MAX_SIZE );
}


JEM_END_PACKAGE( gl )
