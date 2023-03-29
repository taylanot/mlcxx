
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

#ifndef JEM_GL_COLORMAP_H
#define JEM_GL_COLORMAP_H

#include <jem/base/Object.h>
#include <jem/gl/Array.h>
#include <jem/gl/Color.h>


JEM_BEGIN_PACKAGE( gl )


class GLContext;


//-----------------------------------------------------------------------
//   class ColorMap
//-----------------------------------------------------------------------


class ColorMap : public Object
{
 public:

  JEM_DECLARE_CLASS       ( ColorMap, Object );


                            ColorMap      ();

  explicit                  ColorMap

    ( const GLfloatMatrix&    colors );

  explicit                  ColorMap

    ( GLfloatMatrix&&         colors );

  inline void               enable

    ( GLContext&              gtx );

  inline void               disable

    ( GLContext&              gtx );

  void                      setColors

    ( const GLfloatMatrix&    colors );

  void                      setColors

    ( GLfloatMatrix&&         colors );

  inline GLuint             getTexID      () const noexcept;
  inline GLfloatMatrix      getColors     () const noexcept;

  Color                     getColor

    ( GLfloat                 idx )          const noexcept;

  GLfloat                   getMinAlpha   () const noexcept;

  void                      makeSmooth

    ( bool                    choice = true );

  inline bool               isSmooth      () const noexcept;


 protected:

  virtual                  ~ColorMap      ();


 private:

  void                      enable_

    ( GLContext&              gtx );

  void                      disable_

    ( GLContext&              gtx );

  GLuint                    initTexture_

    ( GLContext&              gtx );

  static GLubyteMatrix      getTexImage_

    ( const GLfloatMatrix&    colors );


 private:

  class                     Texture_;

  GLfloatMatrix             colors_;
  Ref<Texture_>             texture_;
  GLuint                    texID_;
  bool                      smooth_;
  bool                      enabled_;

};


//-----------------------------------------------------------------------
//   class ColorMapScope
//-----------------------------------------------------------------------


class ColorMapScope
{
 public:

  explicit inline           ColorMapScope

    ( GLContext&              gtx,
      ColorMap&               cmap );

  inline                   ~ColorMapScope ();


 private:

  GLContext&                gtx_;
  ColorMap&                 cmap_;

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class ColorMap
//=======================================================================

//-----------------------------------------------------------------------
//   enable
//-----------------------------------------------------------------------


inline void ColorMap::enable ( GLContext& gtx )
{
  if ( ! enabled_ )
  {
    enable_ ( gtx );
  }
}


//-----------------------------------------------------------------------
//   disable
//-----------------------------------------------------------------------


inline void ColorMap::disable ( GLContext& gtx )
{
  if ( enabled_ )
  {
    disable_ ( gtx );
  }
}


//-----------------------------------------------------------------------
//   getTexID
//-----------------------------------------------------------------------


inline GLuint ColorMap::getTexID () const noexcept
{
  if ( enabled_ )
  {
    return texID_;
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   getColors
//-----------------------------------------------------------------------


inline GLfloatMatrix ColorMap::getColors () const noexcept
{
  return colors_;
}


//-----------------------------------------------------------------------
//   isSmooth
//-----------------------------------------------------------------------


inline bool ColorMap::isSmooth () const noexcept
{
  return smooth_;
}


//=======================================================================
//   class ColorMapScope
//=======================================================================


inline ColorMapScope::ColorMapScope

  ( GLContext&  gtx,
    ColorMap&   cmap ) :

    gtx_  ( gtx ),
    cmap_ ( cmap )

{
  cmap_.enable ( gtx );
}


inline ColorMapScope::~ColorMapScope ()
{
  cmap_.disable ( gtx_ );
}


JEM_END_PACKAGE( gl )

#endif
