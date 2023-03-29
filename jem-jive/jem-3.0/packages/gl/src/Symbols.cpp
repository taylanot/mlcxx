
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


#include <jem/base/StringBuffer.h>
#include <jem/gl/Symbols.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   struct Symbols::Symbol_
//=======================================================================


struct Symbols::Symbol_
{
  const char*   name;
  const char    code;
};


//=======================================================================
//   class Symbols_
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char  Symbols::NONE           =  0;
const char  Symbols::ASTERIX        = 38;
const char  Symbols::BOX            = 46;
const char  Symbols::CIRCLE         = 58;
const char  Symbols::DIAMOND        = 81;
const char  Symbols::DISC           = 53;
const char  Symbols::FEMALE         = 33;
const char  Symbols::MALE           = 34;
const char  Symbols::PLUS           = 35;
const char  Symbols::SNOW_FLAKE     = 41;
const char  Symbols::SQUARE         = 67;
const char  Symbols::STAR           = 70;
const char  Symbols::TRIANGLE       = 54;
const char  Symbols::TRIANGLE_UP    = 54;
const char  Symbols::TRIANGLE_DOWN  = 55;
const char  Symbols::TRIANGLE_LEFT  = 56;
const char  Symbols::TRIANGLE_RIGHT = 57;


const Symbols::Symbol_  Symbols::SYMBOLS_ [] =
{
  { "asterix",        ASTERIX        },
  { "box",            BOX            },
  { "circle",         CIRCLE         },
  { "diamond",        DIAMOND        },
  { "disc",           DISC           },
  { "female",         FEMALE         },
  { "male",           MALE           },
  { "plus",           PLUS           },
  { "snow flake",     SNOW_FLAKE     },
  { "square",         SQUARE         },
  { "star",           STAR           },
  { "triangle",       TRIANGLE       },
  { "up triangle",    TRIANGLE_UP    },
  { "down triangle",  TRIANGLE_DOWN  },
  { "left triangle",  TRIANGLE_LEFT  },
  { "right triangle", TRIANGLE_RIGHT },
  { nullptr,          0              }
};


//-----------------------------------------------------------------------
//   byName
//-----------------------------------------------------------------------


char Symbols::byName ( const String& name ) noexcept
{
  char  symbol = NONE;

  for ( int i = 0; SYMBOLS_[i].name; i++ )
  {
    if ( SYMBOLS_[i].name == name )
    {
      symbol = SYMBOLS_[i].code;
      break;
    }
  }

  return symbol;
}


//-----------------------------------------------------------------------
//   getName
//-----------------------------------------------------------------------


String Symbols::getName ( char symbol ) noexcept
{
  const char*  name = "";

  for ( int i = 0; SYMBOLS_[i].name; i++ )
  {
    if ( SYMBOLS_[i].code == symbol )
    {
      name = SYMBOLS_[i].name;
      break;
    }
  }

  return name;
}


//-----------------------------------------------------------------------
//   getChoice
//-----------------------------------------------------------------------


String Symbols::getChoice ()
{
  StringBuffer  buf;

  buf.reserve ( 128 );

  for ( int i = 0; SYMBOLS_[i].name; i++ )
  {
    if ( i > 0 )
    {
      if ( SYMBOLS_[i + 1].name )
      {
        print ( buf, ", " );
      }
      else
      {
        print ( buf, " or " );
      }
    }

    print ( buf, '`', SYMBOLS_[i].name, '\'' );
  }

  return buf.toString ();
}


JEM_END_PACKAGE( gl )
