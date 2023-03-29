
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


#include <jem/base/ClassTemplate.h>
#include <jem/base/Time.h>
#include <jem/base/Integer.h>
#include <jem/io/PrefixWriter.h>


JEM_DEFINE_CLASS( jem::io::PrefixWriter );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class PrefixWriter
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PrefixWriter::PrefixWriter

  ( const Ref<Writer>&  out,
    const String&       prefix,
    Options             options ) :

    output_ ( out ),
    buffer_ ( (options & FLUSH_LINES) ? 128 : 512 )

{
  JEM_PRECHECK ( out );

  lineno_       = 1_lint;
  atStart_      = true;
  flushLines_   = (options & FLUSH_LINES);
  simplePrefix_ = true;

  setPrefix ( prefix );
}


PrefixWriter::PrefixWriter ( const Self& rhs ) :

  output_ ( rhs.output_->dup() ),
  buffer_ ( rhs.buffer_.size() ),
  prefix_ ( rhs.prefix_        )

{
  lineno_       = rhs.lineno_;
  atStart_      = true;
  flushLines_   = rhs.flushLines_;
  simplePrefix_ = rhs.simplePrefix_;
}


PrefixWriter::~PrefixWriter ()
{
  buffer_.flushNoThrow ( *output_ );
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> PrefixWriter::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void PrefixWriter::close ()
{
  buffer_ .flush ( *output_ );
  output_->close ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void PrefixWriter::flush ()
{
  buffer_. flush ( *output_ );
  output_->flush ();
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void PrefixWriter::write ( int c )
{
  if ( atStart_ && c != '\n' )
  {
    writePrefix_ ();
  }

  buffer_.write ( *output_, c );

  if ( c == '\n' )
  {
    if ( flushLines_ )
    {
      buffer_. flush ( *output_ );
      output_->flush ();
    }

    atStart_ = true;
  }
}


void PrefixWriter::write ( int c, idx_t n )
{
  if ( atStart_ && c != '\n' )
  {
    writePrefix_ ();
  }

  buffer_.write ( *output_, c, n );

  if ( c == '\n' )
  {
    if ( flushLines_ )
    {
      buffer_. flush ( *output_ );
      output_->flush ();
    }

    atStart_ = true;
  }
}


void PrefixWriter::write ( const char* buf, idx_t n )
{
  bool  doFlush = false;


  for ( idx_t i = 0; i < n; i++ )
  {
    int  c = buf[i];

    if ( atStart_ && c != '\n' )
    {
      writePrefix_ ();
    }

    buffer_.write ( *output_, c );

    if ( c == '\n' )
    {
      doFlush  = flushLines_;
      atStart_ = true;
    }
  }

  if ( doFlush )
  {
    buffer_. flush ( *output_ );
    output_->flush ();
  }
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void PrefixWriter::writeNoThrow

  ( const char*  buf,
    idx_t        n ) noexcept

{
  buffer_.writeNoThrow ( *output_, buf, n );
}


//-----------------------------------------------------------------------
//   setPrefix
//-----------------------------------------------------------------------


void PrefixWriter::setPrefix ( const String& prefix )
{
  prefix_       =  prefix;
  simplePrefix_ = (prefix.find( '%' ) < 0);
}


//-----------------------------------------------------------------------
//   setLineNumber
//-----------------------------------------------------------------------


void PrefixWriter::setLineNumber ( lint lineno )
{
  lineno_ = lineno;
}


//-----------------------------------------------------------------------
//   writePrefix_
//-----------------------------------------------------------------------


void PrefixWriter::writePrefix_ ()
{
  const idx_t  n = prefix_.size ();
  const char*  p = prefix_.addr ();


  if ( simplePrefix_ )
  {
    buffer_.write ( *output_, p, n );
  }
  else
  {
    idx_t  i = 0;

    while ( i < n )
    {
      char  c = p[i++];

      if ( c != '%' )
      {
        buffer_.write ( *output_, c );
        continue;
      }

      if ( i == n )
      {
        c = '%';
      }
      else
      {
        c = p[i++];
      }

      switch ( c )
      {
      case '%' :

        buffer_.write ( *output_, c );
        break;

      case 'l' :

        writeLineno_ ();
        break;

      case 't' :

        writeTime_ ();
        break;

      case 'd' :

        writeDate_ ();
        break;
      }
    }
  }

  lineno_++;

  atStart_ = (n == 0);
}


//-----------------------------------------------------------------------
//   writeLineno_
//-----------------------------------------------------------------------


void PrefixWriter::writeLineno_ ()
{
  const  int  N = 64;

  char   buf[N];
  idx_t  n;

  n = Integer::print ( buf, N, lineno_ );

  buffer_.write ( *output_, buf, n );
}


//-----------------------------------------------------------------------
//   writeTime_
//-----------------------------------------------------------------------


void PrefixWriter::writeTime_ ()
{
  Time    t = Time::now ();
  String  s = t.toString ( "%H:%M:%S" );

  buffer_.write ( *output_, s.addr(), s.size() );
}


//-----------------------------------------------------------------------
//   writeDate_
//-----------------------------------------------------------------------


void PrefixWriter::writeDate_ ()
{
  Time    t = Time::now ();
  String  s = t.toString ( "%c" );

  buffer_.write ( *output_, s.addr(), s.size() );
}


JEM_END_PACKAGE( io )
