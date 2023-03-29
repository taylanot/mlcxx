
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
#include <jem/base/assert.h>
#include <jem/base/MemCache.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Flex.h>
#include <jem/io/params.h>
#include <jem/io/PrintWriter.h>


JEM_DEFINE_CLASS( jem::io::PrintWriter );


JEM_BEGIN_PACKAGE( io )


using jem::util::Flex;


//=======================================================================
//   class PrintWriter::Settings_
//=======================================================================


class PrintWriter::Settings_
{
 public:

  NumberFormat          nformat;

  Options               options;
  int                   indentWidth;
  int                   pageWidth;

};


//=======================================================================
//   class PrintWriter::Data_
//=======================================================================


class PrintWriter::Data_
{
 public:


  explicit              Data_

    ( const Ref<Writer>&  out,
      idx_t               bufsize );

                        Data_

    ( const Data_&        rhs );

                       ~Data_     ();

  inline bool           overfull  () const;
  void                  flush     ();
  inline void           newLine   ();
  inline void           indent    ();

  inline void           write

    ( int                 c );

  void                  write

    ( int                 c,
      idx_t               n );

  void                  write

    ( const char*         buf,
      idx_t               n );


 public:

  Ref<Writer>           output;

  Flex<int>             marginStack;
  Flex<Settings_>       settingStack;

  char*                 buffer;

  idx_t                 capacity;
  idx_t                 last;
  lint                  lineno;
  lint                  column;
  int                   margin;
  lint                  blankLines;

  Options               options;
  int                   indentWidth;
  int                   pageWidth;

  ThrowMode             throwMode;

};


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PrintWriter::Data_::Data_

  ( const Ref<Writer>&  out,
    idx_t               bufsize )

{
  output       = out;
  capacity     = max ( bufsize, (idx_t) NumberFormat::MIN_BUFSIZE );
  last         = 0;
  lineno       = 1_lint;
  column       = 0;
  margin       = 0;
  blankLines   = 0;
  indentWidth  = INDENT_WIDTH;
  pageWidth    = PAGE_WIDTH;
  throwMode    = CAN_THROW;

  buffer       = (char*) MemCache::alloc ( (size_t) capacity );
}


PrintWriter::Data_::Data_ ( const Data_& rhs )
{
  output       = rhs.output->dup ();
  marginStack  = rhs.marginStack;
  settingStack = rhs.settingStack;
  capacity     = rhs.capacity;
  last         = 0;
  lineno       = rhs.lineno;
  column       = 0;
  margin       = rhs.margin;
  blankLines   = rhs.blankLines;
  options      = rhs.options;
  indentWidth  = rhs.indentWidth;
  pageWidth    = rhs.pageWidth;
  throwMode    = rhs.throwMode;

  buffer       = (char*) MemCache::alloc ( (size_t) capacity );
}


PrintWriter::Data_::~Data_ ()
{
  if ( last > 0 )
  {
    output->writeNoThrow ( buffer, last );
  }

  MemCache::dealloc ( buffer, (size_t) capacity );

  buffer = nullptr;
}


//-----------------------------------------------------------------------
//   overfull
//-----------------------------------------------------------------------


inline bool PrintWriter::Data_::overfull () const
{
  if ( options & NO_WORD_WRAP )
  {
    return false;
  }
  else
  {
    return ((last + column) > pageWidth);
  }
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void PrintWriter::Data_::flush ()
{
  idx_t  n = last;

  column += (lint) n;
  last    = 0;

  if ( throwMode == CAN_THROW )
  {
    output->write        ( buffer, n );
  }
  else
  {
    output->writeNoThrow ( buffer, n );
  }
}


//-----------------------------------------------------------------------
//   newLine
//-----------------------------------------------------------------------


inline void PrintWriter::Data_::newLine ()
{
  if ( (last == capacity) || (options & FLUSH_LINES))
  {
    flush ();
  }

  if ( last + column == 0 )
  {
    blankLines++;
  }

  buffer[last++] = '\n';
  column         = (lint) -last;

  lineno++;
}


//-----------------------------------------------------------------------
//   indent
//-----------------------------------------------------------------------


inline void PrintWriter::Data_::indent ()
{
  if ( last + column == 0 )
  {
    blankLines = 0;

    if ( margin )
    {
      write ( ' ', margin );
    }
  }
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


inline void PrintWriter::Data_::write ( int c )
{
  if ( last == capacity )
  {
    flush ();
  }

  buffer[last++] = (char) c;
}


void PrintWriter::Data_::write ( int c, idx_t n )
{
  while ( true )
  {
    idx_t  k = last + n;

    if ( k <= capacity )
    {
      for ( ; last < k; last++ )
      {
        buffer[last] = (char) c;
      }

      break;
    }
    else
    {
      k  = capacity;
      n -= (k - last);

      for ( ; last < k; last++ )
      {
        buffer[last] = (char) c;
      }

      flush ();
    }
  }
}


void PrintWriter::Data_::write ( const char* buf, idx_t n )
{
  while ( true )
  {
    idx_t  k = capacity - last;

    if ( n <= k )
    {
      std::memcpy ( buffer + last, buf, (size_t) n );

      last += n;
      break;
    }
    else
    {
      std::memcpy ( buffer + last, buf, (size_t) k );

      last += k;
      buf  += k;
      n    -= k;

      flush ();
    }
  }
}


//=======================================================================
//   class PrintWriter
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  PrintWriter::PAGE_WIDTH    = 256;
const int  PrintWriter::INDENT_WIDTH  =   2;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PrintWriter::PrintWriter

  ( const Ref<Writer>&  out,
    idx_t               bufsize )

{
  JEM_PRECHECK ( out );

  if ( bufsize <= 0 )
  {
    bufsize = DEFAULT_BUFSIZE;
  }

  data_ = new Data_ ( out, bufsize );
}


PrintWriter::PrintWriter ( const Self& rhs ) :

  nformat ( rhs.nformat )

{
  data_ = new Data_ ( *rhs.data_ );
}


PrintWriter::~PrintWriter ()
{
  delete data_;
  data_ = nullptr;
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> PrintWriter::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void PrintWriter::close ()
{
  data_->flush         ();
  data_->output->close ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void PrintWriter::flush ()
{
  data_->flush         ();
  data_->output->flush ();
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void PrintWriter::write ( int c )
{
  Data_&  d = * data_;


  if      ( c == '\n' )
  {
    d.newLine ();
  }
  else if ( std::isspace(c) && d.overfull() )
  {
    d.newLine ();
  }
  else
  {
    d.indent  ();
    d.write   ( c );
  }
}


void PrintWriter::write ( int c, idx_t n )
{
  Data_&  d = * data_;


  if ( n <= 0 )
  {
    return;
  }

  if      ( c == '\n' )
  {
    d.newLine ();

    n--;

    d.write ( c, n );

    d.lineno     += (lint) n;
    d.blankLines += (lint) n;
  }
  else if ( std::isspace(c) && d.overfull() )
  {
    d.newLine ();
  }
  else
  {
    d.indent  ();
    d.write   ( c, n );
  }
}


void PrintWriter::write ( const char* buf, idx_t n )
{
  Data_&  d = * data_;

  idx_t   i = 0;


  while ( i < n )
  {
    if ( buf[i] == '\n' )
    {
      d.newLine ();
      i++;
    }
    else if ( std::isspace(buf[i]) && d.overfull() )
    {
      d.newLine ();
      i++;

      while ( i < n && (buf[i] == ' ' || buf[i] == '\t') )
      {
        i++;
      }
    }
    else
    {
      d.indent ();
      d.write  ( buf[i++] );

      for ( ; i < n; i++ )
      {
        if ( buf[i] == '\n' )
        {
          break;
        }
        if ( std::isspace(buf[i]) && d.overfull() )
        {
          break;
        }

        d.write ( buf[i] );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void PrintWriter::writeNoThrow ( const char* buf, idx_t n ) noexcept
{
  data_->throwMode = NO_THROW;

  Self::write ( buf, n );

  data_->throwMode = CAN_THROW;
}


//-----------------------------------------------------------------------
//   flushBuffer
//-----------------------------------------------------------------------


void PrintWriter::flushBuffer ()
{
  data_->flush ();
}


//-----------------------------------------------------------------------
//   printBool
//-----------------------------------------------------------------------


void PrintWriter::printBool ( bool value )
{
  Data_&  d = * data_;


  d.indent ();

  if ( value )
  {
    if ( d.options & NUMERIC_BOOLS )
    {
      d.write ( '1' );
    }
    else
    {
      d.write ( "true",  4 );
    }
  }
  else
  {
    if ( d.options & NUMERIC_BOOLS )
    {
      d.write ( '0' );
    }
    else
    {
      d.write ( "false", 5 );
    }
  }
}


//-----------------------------------------------------------------------
//   printInt
//-----------------------------------------------------------------------


void PrintWriter::printInt ( lint i )
{
  Data_&  d = * data_;

  idx_t   n;


  d.indent ();

  if ( d.last + NumberFormat::MIN_BUFSIZE > d.capacity )
  {
    d.flush ();
  }

  n       = nformat.print ( d.buffer + d.last, i );
  d.last += n;
}


//-----------------------------------------------------------------------
//   printFloat (double)
//-----------------------------------------------------------------------


void PrintWriter::printFloat ( double value )
{
  Data_&         d  = * data_;
  NumberFormat&  nf =   nformat;

  int  n;


  d.indent ();

  if ( d.last + NumberFormat::MIN_BUFSIZE > d.capacity )
  {
    d.flush ();
  }

  n = std::sprintf ( d.buffer + d.last,
                     nf.format_, nf.floatWidth_, nf.fracDigits_, value );

  d.last += n;
}


//-----------------------------------------------------------------------
//   printLine
//-----------------------------------------------------------------------


void PrintWriter::printLine ()
{
  data_->newLine ();
}


//-----------------------------------------------------------------------
//   printSpace
//-----------------------------------------------------------------------


void PrintWriter::printSpace ()
{
  Data_&  d = * data_;

  if ( d.overfull() )
  {
    d.newLine ();
  }
  else
  {
    d.indent ();
    d.write  ( ' ' );
  }
}


//-----------------------------------------------------------------------
//   printSpaces
//-----------------------------------------------------------------------


void PrintWriter::printSpaces ( idx_t n )
{
  Data_&  d = * data_;

  if ( d.overfull() )
  {
    d.newLine ();
  }
  else
  {
    d.indent ();
    d.write  ( ' ', n );
  }
}


//-----------------------------------------------------------------------
//   printBlankLine
//-----------------------------------------------------------------------


void PrintWriter::printBlankLine ()
{
  Data_&  d = * data_;

  while ( d.blankLines < 1_lint )
  {
    d.newLine ();
  }
}


//-----------------------------------------------------------------------
//   printBlankLines
//-----------------------------------------------------------------------


void PrintWriter::printBlankLines ( idx_t n )
{
  Data_&  d = * data_;

  while ( d.blankLines < n )
  {
    d.newLine ();
  }
}


//-----------------------------------------------------------------------
//   pushSettings
//-----------------------------------------------------------------------


void PrintWriter::pushSettings ()
{
  Data_&     d = * data_;
  Settings_  s;

  s.nformat     = nformat;
  s.options     = d.options;
  s.indentWidth = d.indentWidth;
  s.pageWidth   = d.pageWidth;

  d.settingStack.pushBack ( s );
}


//-----------------------------------------------------------------------
//   popSettings
//-----------------------------------------------------------------------


void PrintWriter::popSettings ()
{
  if ( data_->settingStack.size() > 0 )
  {
    Data_&            d = * data_;
    const Settings_&  s = d.settingStack.back ();

    nformat       = s.nformat;
    d.options     = s.options;
    d.indentWidth = s.indentWidth;
    d.pageWidth   = s.pageWidth;

    d.settingStack.popBack ();
  }
}


//-----------------------------------------------------------------------
//   incrIndentLevel
//-----------------------------------------------------------------------


int PrintWriter::incrIndentLevel ()
{
  Data_&  d = * data_;

  d.marginStack.pushBack ( d.margin );

  d.margin += d.indentWidth;

  return (int) d.marginStack.size ();
}


//-----------------------------------------------------------------------
//   decrIndentLevel
//-----------------------------------------------------------------------


int PrintWriter::decrIndentLevel ()
{
  Data_&  d = * data_;


  if ( d.marginStack.size() )
  {
    d.margin = d.marginStack.back ();

    d.marginStack.popBack ();
  }

  return (int) d.marginStack.size ();
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void PrintWriter::setOption

  ( Option  option,
    bool    yesno )

{
  data_->options.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void PrintWriter::setOptions ( Options options )
{
  data_->options = options;
}


//-----------------------------------------------------------------------
//   setPageWidth
//-----------------------------------------------------------------------


void PrintWriter::setPageWidth ( int width )
{
  JEM_PRECHECK ( width > 0 );

  data_->pageWidth = width;
}


//-----------------------------------------------------------------------
//   setIndentWidth
//-----------------------------------------------------------------------


void PrintWriter::setIndentWidth ( int width )
{
  JEM_PRECHECK ( width >= 0 );

  data_->indentWidth = width;
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


PrintWriter::Options PrintWriter::getOptions () const noexcept
{
  return data_->options;
}


//-----------------------------------------------------------------------
//   getPageWidth
//-----------------------------------------------------------------------


int PrintWriter::getPageWidth () const noexcept
{
  return data_->pageWidth;
}


//-----------------------------------------------------------------------
//   getIndentWidth
//-----------------------------------------------------------------------


int PrintWriter::getIndentWidth () const noexcept
{
  return data_->indentWidth;
}


//-----------------------------------------------------------------------
//   getIndentLevel
//-----------------------------------------------------------------------


int PrintWriter::getIndentLevel () const noexcept
{
  return (int) data_->marginStack.size ();
}


//-----------------------------------------------------------------------
//   getMarginWidth
//-----------------------------------------------------------------------


int PrintWriter::getMarginWidth () const noexcept
{
  return data_->margin;
}


//-----------------------------------------------------------------------
//   getLineNumber
//-----------------------------------------------------------------------


lint PrintWriter::getLineNumber () const noexcept
{
  return data_->lineno;
}


//-----------------------------------------------------------------------
//   getColumnNumber
//-----------------------------------------------------------------------


lint PrintWriter::getColumnNumber () const noexcept
{
  const Data_&  d = * data_;

  return jem::max ( (lint) (d.column + d.last), (lint) d.margin );
}


//-----------------------------------------------------------------------
//   atStartOfLine
//-----------------------------------------------------------------------


bool PrintWriter::atStartOfLine () const noexcept
{
  const Data_&  d = * data_;

  return (d.column + d.last <= d.margin);
}


//=======================================================================
//   related classes
//=======================================================================


const Fill        dots      = { '.', 48 };
const Fill        spaces    = { ' ', 48 };
const Indent      indent    = {};
const Outdent     outdent   = {};
const BlankLine   blankLine = {};


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


void            print

  ( PrintWriter&  out,
    const Fill&   fill )

{
  const int  n = (int) (fill.endCol - out.getColumnNumber());

  if ( n > 0 )
  {
    out.write ( ' ' );
    out.write ( '.', n );
    out.write ( ' ' );
  }
  else
  {
    out.write ( ' ' );
  }
}


JEM_END_PACKAGE( io )
