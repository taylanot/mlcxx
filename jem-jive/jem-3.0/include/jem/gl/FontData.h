
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

#ifndef JEM_GL_FONTDATA_H
#define JEM_GL_FONTDATA_H

#include <jem/defines.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class FontData
//-----------------------------------------------------------------------


class FontData
{
 public:

  static const int        FIRST_CHAR     =   32;
  static const int        LAST_CHAR      =  127;
  static const int        CHAR_COUNT     =   96;
  static const int        BYTES_PER_CHAR =    3;
  static const GLubyte    MAGIC_BYTE     = 0x66;

  static const char*      TYPE_NAMES         [3];
  static const char*      SIZE_NAMES         [3];
  static const GLubyte*   FONT_TABLE         [9];

  static const GLubyte    MONOSPACE_SMALL     [];
  static const GLubyte    MONOSPACE_MEDIUM    [];
  static const GLubyte    MONOSPACE_LARGE     [];

  static const GLubyte    PROPORTIONAL_SMALL  [];
  static const GLubyte    PROPORTIONAL_MEDIUM [];
  static const GLubyte    PROPORTIONAL_LARGE  [];

  static const GLubyte    SYMBOL_SMALL        [];
  static const GLubyte    SYMBOL_MEDIUM       [];
  static const GLubyte    SYMBOL_LARGE        [];


  static inline
    const GLubyte*        getTexData

    ( const GLubyte*        fnData );

  static inline
    const GLubyte*        getCharData

    ( int                   ch,
      const GLubyte*        fnData );

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getTexData
//-----------------------------------------------------------------------


inline const GLubyte* FontData::getTexData ( const GLubyte* fnData )
{
  const int  offset = 3 + CHAR_COUNT * BYTES_PER_CHAR;

  return (fnData + offset);
}


//-----------------------------------------------------------------------
//   getCharData
//-----------------------------------------------------------------------


inline const GLubyte* FontData::getCharData

  ( int             ch,
    const GLubyte*  fnData )

{
  if ( ch < FIRST_CHAR || ch > LAST_CHAR )
  {
    ch = LAST_CHAR;
  }

  const int  offset = 3 + (ch - FIRST_CHAR) * BYTES_PER_CHAR;

  return (fnData + offset);
};


JEM_END_PACKAGE( gl )

#endif
