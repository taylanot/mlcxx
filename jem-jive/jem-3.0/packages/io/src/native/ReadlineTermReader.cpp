
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <jem/base/Once.h>
#include <jem/base/Mutex.h>
#include <jem/base/CString.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/File.h>
#include <jem/io/InputStream.h>
#include <jem/io/EOFException.h>
#include <jem/io/WordCompleter.h>
#include <jem/io/config.h>
#include "native/ReadlineTermReader.h"

extern "C"
{
  #include <readline/history.h>
  #include <readline/readline.h>
}


JEM_DEFINE_CLASS( jem::io::StdTermReader );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   non-member functions and data
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


static WordCompleter*  completer_;


//-----------------------------------------------------------------------
//   compGenerator
//-----------------------------------------------------------------------


extern "C"
{
  static char*    nextCompletion_

    ( const char*   word,
      int           state )

  {
    char*  cstr = 0;

    try
    {
      String       comp = completer_->nextCompletion ();
      const idx_t  len  = comp.size ();

      if ( len )
      {
        cstr = (char*) std::malloc ( (size_t) len + 1);

        if ( cstr )
        {
          std::memcpy ( cstr, comp.addr(), (size_t) len );

          cstr[len] = '\0';
        }
      }
    }
    catch ( ... ) {}

    return cstr;
  }

  static char**   listCompletions_

    ( const char*   word,
      int           start,
      int           end )

  {
   if ( ! completer_ )
    {
      return (char**) 0;
    }

    const char*  buf = ::rl_line_buffer;

    int          i   = 0;
    int          j   = start;


    while ( j > i && std::isspace( buf[j - 1] ) )
    {
      j--;
    }

    while ( i < j && std::isspace( buf[i] ) )
    {
      i++;
    }

    try
    {
      String  prev ( buf + i, j - i );

      completer_->newWord ( clone( word ), prev );
    }
    catch ( ... )
    {
      return (char**) 0;
    }

#ifdef JEM_USE_GNU_READLINE

    return ::rl_completion_matches ( word, nextCompletion_ );

#else

    return ::completion_matches    ( word, nextCompletion_ );

#endif
  }
}


//=======================================================================
//   class StdTermReader::Data_
//=======================================================================


class StdTermReader::Data_
{
 public:

  Mutex         mutex;
  String        lastLine;

};


//=======================================================================
//   class Cleanup_
//=======================================================================


class StdTermReader::Cleanup_
{
 public:

  inline       ~Cleanup_  ();

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline StdTermReader::Cleanup_::~Cleanup_ ()
{
  if ( data_ && data_->mutex.locked() )
  {
    // Restore the terminal settings.

#ifdef JEM_USE_GNU_READLINE

    ::rl_free_line_state      ();
    ::rl_cleanup_after_signal ();

#else

    ::rl_deprep_terminal ();

#endif
  }
}


//=======================================================================
//   class StdTermReader
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


StdTermReader::Data_*    StdTermReader::data_ = 0;
StdTermReader::Cleanup_  StdTermReader::cleanup_;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdTermReader::StdTermReader ()
{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, init_ );
}


StdTermReader::~StdTermReader ()
{}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t StdTermReader::poll ( const Time& timeout )
{
  return File::getStdin()->poll ( timeout );
}


//-----------------------------------------------------------------------
//   readLine
//-----------------------------------------------------------------------


String StdTermReader::readLine

  ( const String&   prompt,
    WordCompleter*  wcomp )

{
  Lock<Mutex>  lock ( data_->mutex );

  const idx_t  n = prompt.size ();

  String       line;
  char*        str;


  if ( wcomp )
  {
    completer_ = wcomp;
    ::rl_attempted_completion_function = listCompletions_;
    ::rl_completion_append_character   = '\0';
  }

  if      ( n == 0 )
  {
    str = ::readline ( 0 );
  }
  else if ( n <= 64 )
  {
    char  buf[64 + 1];

    std::memcpy ( buf, prompt.addr(), (size_t) n );

    buf[n] = '\0';

    str = ::readline ( buf );
  }
  else
  {
    str = ::readline ( makeCString( prompt ) );
  }

  completer_ = nullptr;
  ::rl_attempted_completion_function = nullptr;
  ::rl_completion_append_character   = ' ';

  if ( ! str )
  {
    throw EOFException (
      "terminal",
      "end of file encountered while reading from the terminal"
    );
  }

  line = jem::clone ( str );

  if ( line != data_->lastLine && line.size() > 0 )
  {
    ::add_history ( str );

    data_->lastLine = line;
  }

  std::free ( str );

  return line;
}


//-----------------------------------------------------------------------
//   clearHistory
//-----------------------------------------------------------------------


void StdTermReader::clearHistory ()
{
  Lock<Mutex>  lock ( data_->mutex );

  ::clear_history ();
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> StdTermReader::clone () const
{
  return const_cast<Self*> ( this );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void StdTermReader::init_ ()
{
  data_ = new Data_ ();

  ::rl_ignore_completion_duplicates = 0;

  ::using_history ();
}


JEM_END_PACKAGE( io )
