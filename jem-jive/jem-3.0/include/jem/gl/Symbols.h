
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

#ifndef JEM_GL_SYMBOLS_H
#define JEM_GL_SYMBOLS_H

#include <jem/base/String.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Symbols
//-----------------------------------------------------------------------


class Symbols
{
 public:

  static const char       NONE;
  static const char       ASTERIX;
  static const char       BOX;
  static const char       CIRCLE;
  static const char       DIAMOND;
  static const char       DISC;
  static const char       FEMALE;
  static const char       MALE;
  static const char       PLUS;
  static const char       SNOW_FLAKE;
  static const char       SQUARE;
  static const char       STAR;
  static const char       TRIANGLE;
  static const char       TRIANGLE_UP;
  static const char       TRIANGLE_DOWN;
  static const char       TRIANGLE_LEFT;
  static const char       TRIANGLE_RIGHT;


  static char             byName

    ( const String&         name )       noexcept;

  static String           getName

    ( char                  symbol )     noexcept;

  static String           getChoice   ();


 private:

  struct                  Symbol_;

  static const Symbol_    SYMBOLS_ [];

};


JEM_END_PACKAGE( gl )

#endif
