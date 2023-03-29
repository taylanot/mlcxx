
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


#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/Writer.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserState.h>
#include <jem/xml/TextLog.h>


JEM_BEGIN_PACKAGE( xml )


//=======================================================================
//   class TextLog
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


TextLog::TextLog

  ( const Ref<Writer>& output ) :

    output_ ( output ),
    prefix_ ( "  %s, line %l: " )

{
  if ( ! output_ )
  {
    output_ = & System::info ();
  }

  tagCount_ = 0;
}


TextLog::TextLog

  ( const String&       prefix,
    const Ref<Writer>&  output ) :

    output_ ( output ),
    prefix_ ( prefix )

{
  if ( ! output_ )
  {
    output_ = & System::info ();
  }

  if ( ! checkPrefix_ ( prefix ) )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "illegal prefix string: `%s\'", prefix
      )
    );
  }

  tagCount_ = 0;
}


TextLog::~TextLog ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void TextLog::reset ()
{
  timer_.reset ();
  timer_.start ();

  tagCount_ = 0;
}


//-----------------------------------------------------------------------
//   logEvent
//-----------------------------------------------------------------------


void TextLog::logEvent

  ( State&         state,
    Event          event,
    const String&  param )

{
  Writer&  out = * output_;

  if ( event == TAG_START ||
       event == TAG_END   ||
       event == TAG_SKIP )
  {
    if      ( tagCount_ < 50 )
    {
      tagCount_++;
    }
    else if ( timer_.toDouble() > 1.0 )
    {
      timer_.reset ();
    }
    else
    {
      return;
    }
  }

  switch ( event )
  {
  case TAG_START :

    printPrefix_ ( state );

    print ( out, "reading tag `", param, "\' ...\n" );

    break;

  case TAG_END :

    printPrefix_ ( state );

    print ( out, "done reading tag `", param, "\'\n" );

    break;

  case TAG_SKIP :

    printPrefix_ ( state );

    print ( out, "skipping tag `", param, "\' ...\n" );

    break;

  case PROGRESS :

    if ( timer_.toDouble() > 2.5 )
    {
      tagCount_ = 0;

      timer_.reset ();
      printPrefix_ ( state );

      print ( out, param, '\n' );
    }

    break;

  case SOURCE_PUSHED :

    printPrefix_ ( state );

    print ( out, "reading from `", param, "\' ...\n" );

    break;

  case SOURCE_POPPED :

    printPrefix_ ( state );

    print ( out, "leaving `", param, "\'.\n" );

    break;

  default:

    printPrefix_ ( state );

    print ( out, param, '\n' );
  }

  out.flush ();
}


//-----------------------------------------------------------------------
//   printPrefix_
//-----------------------------------------------------------------------


void TextLog::printPrefix_ ( State& state )
{
  Writer&      out = * output_;
  const char*  px  = prefix_.addr ();
  const idx_t  n   = prefix_.size ();

  idx_t        i, k;


  for ( i = 0; i < n; i++ )
  {
    if ( px[i] != '%' )
    {
      out.write ( px[i] );

      continue;
    }

    i++;

    if      ( px[i] == '%' )
    {
      out.write ( px[i] );
    }
    else if ( px[i] == 'l' )
    {
      print ( out, state.input->getLineNumber() );
    }
    else if ( px[i] == 's' )
    {
      const String&  source = state.sourceStack.top ();

      k = source.size ();

      if ( k > 12 )
      {
        print ( out, "...", source[ slice(k - 9,END) ] );
      }
      else
      {
        print ( out, source );
      }
    }
    else
    {
      throw Error ( JEM_FUNC, "illegal prefix string" );
    }
  }
}


//-----------------------------------------------------------------------
//   checkPrefix_
//-----------------------------------------------------------------------


bool TextLog::checkPrefix_ ( const String& px )
{
  const idx_t  n = px.size ();

  bool         valid;
  idx_t        i;

  for ( i = 0; i < n; i++ )
  {
    if ( px[i] == '%' )
    {
      i++;

      if ( i == n )
      {
        return false;
      }

      valid = ( px[i] == '%' ||
                px[i] == 'l' ||
                px[i] == 's' );

      if ( ! valid )
      {
        return false;
      }
    }
  }

  return true;
}


JEM_END_PACKAGE( xml )
