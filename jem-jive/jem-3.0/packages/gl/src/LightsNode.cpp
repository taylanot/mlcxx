
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/Box.h>
#include <jem/gl/Lights.h>
#include <jem/gl/Actions.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/Transformable.h>
#include <jem/gl/LightsNode.h>


JEM_DEFINE_CLASS( jem::gl::LightsNode );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class LightsNode
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


LightsNode::LightsNode ( const String& name ) :

  Super ( name )

{
  static_assert ( LIGHT_COUNT >= 3, "invalid number of lights" );

  dirtyBits_ = 0;

  for ( int i = 0; i < LIGHT_COUNT; i++ )
  {
    lights_[i].pos     =  0.0;
    lights_[i].color   =  Color ( 0.2F );
    lights_[i].glidx   =  GL_LIGHT0 - 1;
    lights_[i].enabled =  false;
  }

  for ( int i = 0; i < 3; i++ )
  {
    lights_[i].pos[i]  = 1.0;
    lights_[i].enabled = true;
    dirtyBits_        |= (LIGHT0 << i);
  }

  lights_[2].color  = Color ( 1.0 );
  ambient_          = Color ( 0.2F );
  dirtyBits_       |= (LIGHT0 << LIGHT_COUNT);
}


LightsNode::~LightsNode ()
{}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool LightsNode::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  if ( action == (int) GLActions::REDRAW ||
       action == (int) GLActions::REDRAW_FAST )
  {
    if ( dirtyBits_ )
    {
      initLights_ ( rtx.getGLContext() );
    }

    return true;
  }

  if ( action == (int) GLActions::INIT_GL )
  {
    GLContext&     gtx   = rtx.getGLContext ();
    GLFunctions1*  funcs = getFunctions_    ( gtx );

    // Make sure that the lights are re-enabled when the INIT_GL
    // action is requested a second time (when switching to another
    // OpenGL context, for instance).

    dirtyBits_ = 0;

    for ( int i = 0; i < LIGHT_COUNT; i++ )
    {
      lights_[i].glidx = GL_LIGHT0 - 1;

      if ( lights_[i].enabled )
      {
        dirtyBits_ |= (LIGHT0 << i);
      }
    }

    dirtyBits_ |= (LIGHT0 << LIGHT_COUNT);

    if ( funcs )
    {
      funcs->glEnable     ( GL_LIGHTING );
      funcs->glLightModel ( GL_LIGHT_MODEL_TWO_SIDE, 1 );
    }

    initLights_ ( gtx );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   getVisibility
//-----------------------------------------------------------------------


void LightsNode::getVisibility

  ( bool&             viz,
    Box&              vbox,
    const Transform&  proj ) const

{
  viz = true;

  vbox.makeEmpty ();
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void LightsNode::configure ( const Properties& props )
{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    String      lightName;
    Vertex3f    dir;

    for ( int i = 0; i < LIGHT_COUNT; i++ )
    {
      Light_&  light = lights_[i];

      lightName = PropNames::LIGHT + String ( i );

      dir[0] = -light.pos[0];
      dir[1] = -light.pos[1];
      dir[2] = -light.pos[2];

      if ( myProps.contains( lightName ) )
      {
        Properties  lightProps = myProps.getProps ( lightName );

        light.color.configure ( PropNames::COLOR,
                                lightProps );
        Vertex    ::configure ( dir, PropNames::DIRECTION,
                                lightProps );

        lightProps .find ( light.enabled, PropNames::ENABLE );

        light.pos[0] = -dir[0];
        light.pos[1] = -dir[1];
        light.pos[2] = -dir[2];

        dirtyBits_  |= (LIGHT0 << i);
      }
    }

    if ( ambient_.configure( PropNames::AMBIENT, myProps ) )
    {
      dirtyBits_ |= (LIGHT0 << LIGHT_COUNT);
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void LightsNode::getConfig ( const Properties& conf ) const
{
  Properties  myConf = conf.makeProps ( myName_ );

  Vertex3f    dir;


  for ( int i = 0; i < LIGHT_COUNT; i++ )
  {
    Properties     lightConf =

      myConf.makeProps ( PropNames::LIGHT + String(i) );

    const Light_&  light = lights_[i];

    dir[0] = -light.pos[0];
    dir[1] = -light.pos[1];
    dir[2] = -light.pos[2];

    light.color.getConfig ( PropNames::COLOR,     lightConf );
    Vertex    ::getConfig ( PropNames::DIRECTION, lightConf, dir );

    lightConf.set ( PropNames::ENABLE, light.enabled );
  }

  ambient_.getConfig ( PropNames::AMBIENT, myConf );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void LightsNode::listProps ( const Properties& conf ) const
{
  Properties  myConf = conf.makeProps ( myName_ );

  for ( int i = 0; i < LIGHT_COUNT; i++ )
  {
    Properties  lightConf =

      myConf.makeProps ( PropNames::LIGHT + String(i) );

    lightConf.set ( PropNames::COLOR,
                    "The color of this light" );
    lightConf.set ( PropNames::DIRECTION,
                    "The direction of this light" );
    lightConf.set ( PropNames::ENABLE,
                    "Whether this light is enabled" );
  }

  myConf.set ( PropNames::AMBIENT,
               "The color of the ambient light" );
}


//-----------------------------------------------------------------------
//   enable
//-----------------------------------------------------------------------


void LightsNode::enable

  ( int   mask,
    bool  choice )

{
  int  idx = 0;

  while ( (mask > 0) && (idx < LIGHT_COUNT) )
  {
    if ( mask & 1 )
    {
      if ( lights_[idx].enabled != choice )
      {
        lights_[idx].enabled = choice;
        dirtyBits_          |= (LIGHT0 << idx);
      }
    }

    mask >>= 1;
    idx++;
  }
}


//-----------------------------------------------------------------------
//   isEnabled
//-----------------------------------------------------------------------


bool LightsNode::isEnabled ( int idx ) const
{
  return lights_[Lights::getLightIndex( idx )].enabled;
}


//-----------------------------------------------------------------------
//   setColor
//-----------------------------------------------------------------------


void LightsNode::setColor

  ( int           mask,
    const Color&  color )

{
  int  idx = 0;

  while ( (mask > 0) && (idx < LIGHT_COUNT) )
  {
    if ( mask & 1 )
    {
      lights_[idx].color = clamped ( color );
      dirtyBits_        |= (LIGHT0 << idx);
    }

    mask >>= 1;
    idx++;
  }
}


//-----------------------------------------------------------------------
//   getColor
//-----------------------------------------------------------------------


Color LightsNode::getColor ( int idx ) const
{
  return lights_[Lights::getLightIndex( idx )].color;
}


//-----------------------------------------------------------------------
//   setDirection
//-----------------------------------------------------------------------


void LightsNode::setDirection

  ( int              mask,
    const Vertex3f&  dir )

{
  int  idx = 0;

  while ( (mask > 0) && (idx < LIGHT_COUNT) )
  {
    if ( mask & 1 )
    {
      Light_&  light = lights_[idx];

      light.pos[0] = -dir[0];
      light.pos[1] = -dir[1];
      light.pos[2] = -dir[2];

      dirtyBits_  |= (LIGHT0 << idx);
    }

    mask >>= 1;
    idx++;
  }
}


//-----------------------------------------------------------------------
//   getDirection
//-----------------------------------------------------------------------


Vertex3f LightsNode::getDirection ( int idx ) const
{
  const Light_&  light = lights_[Lights::getLightIndex( idx )];

  Vertex3f       dir;

  dir[0] = -light.pos[0];
  dir[1] = -light.pos[1];
  dir[2] = -light.pos[2];

  return dir;
}


//-----------------------------------------------------------------------
//   setAmbient
//-----------------------------------------------------------------------


void LightsNode::setAmbient ( const Color& color )
{
  ambient_    = clamped ( color );
  dirtyBits_ |= (LIGHT0 << LIGHT_COUNT);
}


//-----------------------------------------------------------------------
//   getAmbient
//-----------------------------------------------------------------------


Color LightsNode::getAmbient () const noexcept
{
  return ambient_;
}


//-----------------------------------------------------------------------
//   initLights_
//-----------------------------------------------------------------------


void LightsNode::initLights_ ( GLContext& gtx )
{
  GLFunctions1*  funcs  = getFunctions_         ( gtx );
  Ref<Lights>    lights = gtx    .getLights     ();
  int            mask   = lights->getActiveMask ();


  for ( int i = 0; i < LIGHT_COUNT; i++ )
  {
    if ( dirtyBits_ & (LIGHT0 << i) )
    {
      Light_&   light = lights_[i];
      Color     col   = light.color;
      int       idx   = LIGHT0 << i;
      Vertex3f  dir;

      dir[0] = -light.pos[0];
      dir[1] = -light.pos[1];
      dir[2] = -light.pos[2];

      if ( light.enabled )
      {
        mask |= idx;
      }
      else
      {
        mask &= ~idx;
        col   = Color::black ();
      }

      lights->setColor     ( idx, col );
      lights->setDirection ( idx, dir );

      if ( funcs )
      {
        initLight_ ( gtx, light, *funcs );
      }
    }
  }

  lights->setActiveMask ( mask );

  if ( dirtyBits_ & (LIGHT0 << LIGHT_COUNT) )
  {
    lights->setAmbient ( ambient_ );

    if ( funcs )
    {
      funcs->glLightModel ( GL_LIGHT_MODEL_AMBIENT,
                            ambient_.addr() );
    }
  }

  dirtyBits_ = 0;

  checkGLError ( gtx, JEM_FUNC );
}


//-----------------------------------------------------------------------
//   initLight_
//-----------------------------------------------------------------------


void LightsNode::initLight_

  ( GLContext&     gtx,
    Light_&        light,
    GLFunctions1&  funcs )

{
  int  i = 0;


  if ( ! light.enabled )
  {
    if ( light.glidx >= GL_LIGHT0 )
    {
      funcs.glDisable ( (GLenum) light.glidx );

      light.glidx = (int) GL_LIGHT0 - 1;
    }

    return;
  }

  if ( light.enabled && light.glidx < GL_LIGHT0 )
  {
    for ( i = 0; i < GL_MAX_LIGHTS; i++ )
    {
      if ( funcs.glIsEnabled( (GLenum) (GL_LIGHT0 + i) ) == GL_FALSE )
      {
        break;
      }
    }

    if ( i >= GL_MAX_LIGHTS )
    {
      print ( System::warn(), myName_,
              " : failed to enable another light\n" );

      light.enabled = false;

      return;
    }

    light.glidx = (int) GL_LIGHT0 + i;

    funcs.glEnable ( (GLenum) light.glidx );
  }

  i = light.glidx;

  funcs.glLight ( (GLenum) i, GL_DIFFUSE,  light.color.addr() );
  funcs.glLight ( (GLenum) i, GL_SPECULAR, light.color.addr() );

  gtx.pushTransform ();
  gtx.setIdentity   ();
  funcs.glLight     ( (GLenum) i, GL_POSITION, light.pos.addr() );
  gtx.popTransform  ();
}


//-----------------------------------------------------------------------
//   getGLFunctions_
//-----------------------------------------------------------------------


GLFunctions1* LightsNode::getFunctions_ ( const GLContext& gtx )
{
  GLFunctions1*  funcs = nullptr;

  if ( ! gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    funcs = gtx.findFunctions<GLFunctions1> ();
  }

  return funcs;
}


JEM_END_PACKAGE( gl )
