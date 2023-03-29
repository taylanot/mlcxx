
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


#include <jem/gl/GLException.h>
#include <jem/gl/Lights.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Lights
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Lights::Lights ()
{
  static_assert ( LIGHT_COUNT >= 3, "invalid number of lights" );

  for ( int i = 0; i < LIGHT_COUNT; i++ )
  {
    orient_[i] = 0.0F;
  }

  for ( int i = 0; i < 3; i++ )
  {
    color_ [i]    =  Color ( 0.2F );
    orient_[i][i] = -1.0F;
  }

  color_[2]    = Color ( 1.0F );
  ambient_     = Color ( 0.2F );
  activeMask_  = ALL_LIGHTS;
}


Lights::~Lights ()
{}


//-----------------------------------------------------------------------
//   setColor
//-----------------------------------------------------------------------


void Lights::setColor

  ( int           mask,
    const Color&  color )

{
  int  idx = 0;

  while ( (mask > 0) && (idx < LIGHT_COUNT) )
  {
    if ( mask & 1 )
    {
      color_[idx] = color;
    }

    mask >>= 1;
    idx++;
  }
}


//-----------------------------------------------------------------------
//   setDirection
//-----------------------------------------------------------------------


void Lights::setDirection

  ( int              mask,
    const Vertex3f&  dir )

{
  int  idx = 0;

  while ( (mask > 0) && (idx < LIGHT_COUNT) )
  {
    if ( mask & 1 )
    {
      orient_[idx] = dir;
    }

    mask >>= 1;
    idx++;
  }
}


//-----------------------------------------------------------------------
//   setAmbient
//-----------------------------------------------------------------------


void Lights::setAmbient ( const Color& color )
{
  ambient_ = color;
}


//-----------------------------------------------------------------------
//   getLightIndex
//-----------------------------------------------------------------------


int Lights::getLightIndex ( int idx )
{
  switch ( idx )
  {
  case LIGHT0:

    return 0;

  case LIGHT1:

    return 1;

  case LIGHT2:

    return 2;

  default:

    throw GLException ( JEM_FUNC, "invalid light index" );
  }

  return -1;
}


JEM_END_PACKAGE( gl )
