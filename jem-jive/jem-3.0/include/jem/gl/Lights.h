
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

#ifndef JEM_GL_LIGHTS_H
#define JEM_GL_LIGHTS_H

#include <jem/base/Object.h>
#include <jem/gl/Color.h>
#include <jem/gl/Vertex.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Lights
//-----------------------------------------------------------------------


class Lights : public Object
{
 public:

  static const int        LIGHT0      = 1 << 0;
  static const int        LIGHT1      = 1 << 1;
  static const int        LIGHT2      = 1 << 2;
  static const int        ALL_LIGHTS  = LIGHT0 | LIGHT1 | LIGHT2;

  static const int        LIGHT_COUNT = 3;


                          Lights         ();

  inline const Color&     getColor

    ( int                   idx )           const;

  void                    setColor

    ( int                   mask,
      const Color&          color );

  inline const Vertex3f&  getDirection

    ( int                   idx )           const;

  void                    setDirection

    ( int                   mask,
      const Vertex3f&       dir );

  inline const Color&     getAmbient     () const noexcept;

  void                    setAmbient

    ( const Color&          color );

  inline int              getActiveMask  () const noexcept;

  inline void             setActiveMask

    ( int                   mask );

  static int              getLightIndex

    ( int                   idx );


 protected:

  virtual                 ~Lights        ();


 private:

  Color                   color_  [LIGHT_COUNT];
  Vertex3f                orient_ [LIGHT_COUNT];
  Color                   ambient_;
  int                     activeMask_;

};


//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getColor
//-----------------------------------------------------------------------


inline const Color& Lights::getColor ( int idx ) const
{
  return color_[getLightIndex( idx )];
}


//-----------------------------------------------------------------------
//   getDirection
//-----------------------------------------------------------------------


inline const Vertex3f& Lights::getDirection ( int idx ) const
{
  return orient_[getLightIndex( idx )];
}


//-----------------------------------------------------------------------
//   getAmbient
//-----------------------------------------------------------------------


inline const Color& Lights::getAmbient () const noexcept
{
  return ambient_;
}


//-----------------------------------------------------------------------
//   getActiveMask
//-----------------------------------------------------------------------


inline int Lights::getActiveMask () const noexcept
{
  return activeMask_;
}


//-----------------------------------------------------------------------
//   setActiveMask
//-----------------------------------------------------------------------


inline void Lights::setActiveMask ( int mask )
{
  activeMask_ = mask;
}


JEM_END_PACKAGE( gl )

#endif
