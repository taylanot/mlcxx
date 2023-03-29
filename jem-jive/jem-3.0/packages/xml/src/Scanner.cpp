
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


#include <cctype>
#include <jem/base/limits.h>
#include <jem/xml/Scanner.h>


JEM_BEGIN_PACKAGE( xml )


//=======================================================================
//   class Scanner
//=======================================================================

//-----------------------------------------------------------------------
//   skipXmlWhite
//-----------------------------------------------------------------------


void Scanner::skipXmlWhite ()
{
  int  c = read ();


  while ( c >= 0 )
  {
    if ( std::isspace(c) )
    {
      if ( c == '\n' )
      {
        lineno++;
      }
    }
    else if ( c == '<' && scan( "!--", 3 ) )
    {
      skipComment_ ( false );
    }
    else
    {
      break;
    }

    c = read ();
  }

  unread ( c );
}


//-----------------------------------------------------------------------
//   readName
//-----------------------------------------------------------------------


void Scanner::readName ()
{
  int  c = read ();

  while ( c >= 0 && (std::isalnum(c) ||
                     c == '_' || c == '-' || c == ':') )
  {
    token += (char) c;
    c      = read  ();
  }

  unread ( c );
}


//-----------------------------------------------------------------------
//   readComment
//-----------------------------------------------------------------------


void Scanner::readComment ()
{
  skipComment_ ( true );
}


//-----------------------------------------------------------------------
//   skipComment_
//-----------------------------------------------------------------------


void Scanner::skipComment_ ( bool store )
{
  lint  line  = lineno;
  int   level = 1;

  while ( level > 0 )
  {
    int  c = read ();

    if ( c < 0 )
    {
      parseError (
        String::format (
          "unterminated comment block starting at line %d", line
        )
      );
    }

    if ( store )
    {
      token += (char) c;
    }

    if ( c == '\n' )
    {
      lineno++;
    }

    if ( c == '<' && scan( "!--", 3 ) )
    {
      if ( level >= maxOf( level ) )
      {
        parseError ( "too many nested comment blocks" );
      }

      level++;

      if ( store )
      {
        token += "!--";
      }
    }
    else if ( c == '-' && scan( "->", 2 ) )
    {
      level--;

      if ( store )
      {
        token += "->";
      }
    }
  }
}


JEM_END_PACKAGE( xml )
