
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
#include <jem/base/ClassTemplate.h>
#include <jem/io/FileName.h>
#include <jem/io/FileReader.h>
#include <jem/io/IOException.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserState.h>
#include <jem/xml/IncludeTagParser.h>


using jem::io::Reader;


JEM_DEFINE_CLASS( jem::xml::IncludeTagParser );


JEM_BEGIN_PACKAGE( xml )


//=======================================================================
//   class IncludeTagParser
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  IncludeTagParser::TAG_NAME  = "Include";
const int    IncludeTagParser::MAX_DEPTH = 64;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


IncludeTagParser::IncludeTagParser () noexcept :

  Super ( "source | required" )

{
  required_ = true;
}


IncludeTagParser::~IncludeTagParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void IncludeTagParser::reset ()
{
  source_   = "";
  required_ = true;
}


//-----------------------------------------------------------------------
//   processTag
//-----------------------------------------------------------------------


void IncludeTagParser::processTag ( State& state )
{
  using jem::io::FileName;
  using jem::io::IOException;

  Tokenizer&      input = * state.input;

  Ref<Tokenizer>  newInput;
  Ref<Reader>     file;

  String          fname    = source_;
  bool            required = required_;


  source_   = "";
  required_ = true;

  if ( state.parserStack.size() == 0 )
  {
    return;
  }

  if ( state.sourceStack.size() > MAX_DEPTH )
  {
    inputError (
      state,
      String::format (
        "maximum include depth (%d) exceeded", MAX_DEPTH
      )
    );
  }

  if ( ! FileName::isAbsolute( fname ) )
  {
    fname = FileName::join (
      FileName::getDirName ( state.sourceStack.top() ),
      fname
    );

    fname = FileName::simplify ( fname );
  }

  try
  {
    file = openFile ( state, fname );
  }
  catch ( const IOException& e )
  {
    if ( required )
    {
      if ( isLenient_ )
      {
        print ( System::warn(), e.what(), '\n' );

        return;
      }
      else
      {
        inputError ( state, e.what() );
      }
    }
    else
    {
      return;
    }
  }

  newInput = newInstance<Tokenizer> ( file );

  newInput->setOptions ( input.getOptions() );

  pushInput ( state, newInput, fname );
}


//-----------------------------------------------------------------------
//   openFile
//-----------------------------------------------------------------------


Ref<Reader> IncludeTagParser::openFile

  ( State&         state,
    const String&  fname )

{
  using jem::io::FileReader;

  return newInstance<FileReader> ( fname );
}


//-----------------------------------------------------------------------
//   handleAttribute
//-----------------------------------------------------------------------


void IncludeTagParser::handleAttribute

  ( State&         state,
    const String&  name,
    const String&  value,
    idx_t          index )

{
  switch ( index )
  {
  case 0:

    if ( value.size() == 0 )
    {
      parseError ( state, "empty file name" );
    }

    source_ = value;

    break;

  case 1:

    required_ = parseBoolean ( state, name, value );

    break;

  default:

    throw Error (
      JEM_FUNC,
      "oops, unexpected attribute index: " + String ( index )
    );
  }
}


JEM_END_PACKAGE( xml )
