
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


#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/io/TextInput.h>
#include <jem/base/MemCache.h>
#include <jem/base/AllocatorUtils.h>
#include <jem/base/Integer.h>
#include <jem/base/Float.h>
#include <jem/base/Character.h>
#include <jem/base/String.h>


JEM_BEGIN_PACKAGE_BASE


using jem::io::DataInput;
using jem::io::DataOutput;
using jem::io::TextInput;


//=======================================================================
//   class String::Buffer_
//=======================================================================


class String::Buffer_
{
 public:

                          Buffer_   ()       noexcept;
                         ~Buffer_   ()       noexcept;

  inline void             pushBack

    ( char                  c )              noexcept;

  inline void             pushBack

    ( char                  c,
      idx_t                 n )              noexcept;

  inline void             pushBack

    ( const char*           str,
      idx_t                 len )            noexcept;

  inline idx_t             size     () const noexcept;
  inline char*             addr     () const noexcept;


 private:

  void                     grow_

    ( idx_t                  size )          noexcept;


 private:

  char*                   buffer_;
  idx_t                   size_;
  idx_t                   capacity_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


String::Buffer_::Buffer_ () noexcept
{
  size_     = 0;
  capacity_ = 64;
  buffer_   = (char*) MemCache::allocNoThrow ( (size_t) capacity_ );
}


String::Buffer_::~Buffer_ () noexcept
{
  MemCache::dealloc ( buffer_, (size_t) capacity_ );
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


inline void String::Buffer_::pushBack ( char c ) noexcept
{
  if ( size_ == capacity_ )
  {
    grow_ ( size_ + 1 );
  }

  buffer_[size_++] = c;
}


inline void String::Buffer_::pushBack ( char c, idx_t n ) noexcept
{
  if ( size_ + n > capacity_ )
  {
    grow_ ( size_ + n );
  }

  std::memset ( buffer_ + size_, c, (size_t) n );

  size_ += n;
}


inline void String::Buffer_::pushBack

  ( const char*  str,
    idx_t        len ) noexcept

{
  if ( size_ + len > capacity_ )
  {
    grow_ ( size_ + len );
  }

  std::memcpy ( buffer_ + size_, str, (size_t) len );

  size_ += len;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t String::Buffer_::size () const noexcept
{
  return size_;
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


inline char* String::Buffer_::addr () const noexcept
{
  return buffer_;
}


//-----------------------------------------------------------------------
//   grow_
//-----------------------------------------------------------------------


void String::Buffer_::grow_ ( idx_t n ) noexcept
{
  char*  tmp;
  idx_t  cap;

  cap  = max ( 2 * capacity_, n );
  tmp  = (char*) MemCache::allocNoThrow ( (size_t) cap );

  std::memcpy       ( tmp, buffer_, (size_t) size_ );
  MemCache::dealloc ( buffer_, (size_t) capacity_  );

  buffer_   = tmp;
  capacity_ = cap;
}


//=======================================================================
//   class String::Chunk_
//=======================================================================


class String::Chunk_
{
 public:

  typedef Chunk_          Self;


  static inline Self*     alloc

    ( char**                str,
      idx_t                 len )      noexcept;

  static inline Self*     realloc

    ( Self*                 chunk,
      idx_t                 len )      noexcept;

  inline char*            getData   () noexcept;

  static inline void      addRef

    ( Self*                 chunk )    noexcept;

  static inline void      delRef

    ( Self*                 chunk )    noexcept;


 private:

  explicit inline         Chunk_

    ( size_t                msize )    noexcept;


 private:

  const size_t            msize_;
  std::atomic_int         refcount_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline String::Chunk_::Chunk_ ( size_t msize ) noexcept :

  msize_    ( msize ),
  refcount_ ( 1 )

{}


//-----------------------------------------------------------------------
//   alloc
//-----------------------------------------------------------------------


inline String::Chunk_* String::Chunk_::alloc

  ( char**  str,
    idx_t   len ) noexcept

{
  Self*   chunk;
  size_t  offset;
  char*   mem;

  offset = AllocatorUtils::align ( sizeof(Self) );
  mem    = (char*) MemCache::allocNoThrow ( offset + (size_t) len );
  chunk  = new (mem) Self ( offset + (size_t) len );
  *str   = mem + offset;

  return chunk;
}


//-----------------------------------------------------------------------
//   realloc
//-----------------------------------------------------------------------


inline String::Chunk_* String::Chunk_::realloc

  ( Self*  chunk,
    idx_t  len ) noexcept

{
  size_t  offset;
  char*   mem;

  offset = AllocatorUtils::align ( sizeof(Self) );
  mem    = (char*)

    MemCache::reallocNoThrow ( chunk, chunk->msize_,
                                      offset + (size_t) len );

  return new (mem) Self ( offset + (size_t) len );
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline char* String::Chunk_::getData () noexcept
{
  size_t  offset = AllocatorUtils::align ( sizeof(Self) );

  return (((char*) this) + offset);
}


//-----------------------------------------------------------------------
//   addRef
//-----------------------------------------------------------------------


inline void String::Chunk_::addRef ( Self* chunk ) noexcept
{
  chunk->refcount_.fetch_add ( 1, std::memory_order_relaxed );
}


//-----------------------------------------------------------------------
//   delRef
//-----------------------------------------------------------------------


inline void String::Chunk_::delRef ( Self* chunk ) noexcept
{
  int  count =

    chunk->refcount_.fetch_sub ( 1, std::memory_order_release );

  if ( count == 1 )
  {
    std::atomic_thread_fence ( std::memory_order_acquire );
    MemCache::dealloc        ( chunk, chunk->msize_ );
  }
}


//=======================================================================
//   class String::Utils_
//=======================================================================


class String::Utils_
{
 public:

  static inline char*     init

    ( String*               str,
      idx_t                 len )      noexcept;

  static inline void      free

    ( String*               str )      noexcept;

  static void             format

    ( Buffer_&              buf,
      const char*           fmt,
      const Argument_*      args,
      int                   argc )     noexcept;

  static void             padAndAdd

    ( Buffer_&              buf,
      idx_t                 width,
      const char*           str,
      idx_t                 len )      noexcept;

};


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


inline char* String::Utils_::init

  ( String*  str,
    idx_t    len ) noexcept

{
  if ( len <= TinyString::MAXLEN )
  {
    str->chunk_ = nullptr;
    str->str_   = str->buffer_;
  }
  else
  {
    str->chunk_ = Chunk_::alloc ( &str->str_, len );
  }

  str->size_ = len;

  return str->str_;
}


//-----------------------------------------------------------------------
//   free
//-----------------------------------------------------------------------


inline void String::Utils_::free ( String* str ) noexcept
{
  if ( str->size_ > TinyString::MAXLEN && str->chunk_ )
  {
    Chunk_::delRef ( str->chunk_ );
  }
}


//-----------------------------------------------------------------------
//   format
//-----------------------------------------------------------------------


void String::Utils_::format

  ( Buffer_&          xbuf,
    const char*       fmt,
    const Argument_*  args,
    int               argc ) noexcept

{
  char  buf[64];
  char  flg[16];

  int   wsign;
  int   width;
  int   prec;
  int   iarg;
  int   i, n;


  flg[0] = '%';
  iarg   = 0;

  for ( ; *fmt != '\0'; fmt++ )
  {
    if ( *fmt != '%' )
    {
      xbuf.pushBack ( *fmt );
      continue;
    }

    fmt++;

    i     = 1;
    wsign = 1;

    do
    {
      n = i;

      switch ( *fmt )
      {
      case '-':

        wsign = -wsign;

      case '#':
      case '0':
      case ' ':
      case '+':

        flg[i++] = *fmt++;
      }
    }
    while ( i > n && i < 4 );

    width = 0;

    if      ( *fmt == '*' )
    {
      if ( iarg < argc && args[iarg].type == Argument_::INT )
      {
        width = (int) args[iarg].intValue;
      }

      iarg++;
      fmt++;
    }
    else if ( std::isdigit(*fmt) )
    {
      width = *fmt - '0';
      fmt++;

      while ( std::isdigit(*fmt) )
      {
        width = 10 * width + *fmt - '0';
        fmt++;

        if ( width > 1000 )
        {
          while ( std::isdigit(*fmt) ) fmt++;
        }
      }

      width *= wsign;
    }

    prec = -1;

    if ( *fmt == '.' )
    {
      prec = 0;
      fmt++;

      if ( *fmt == '*' )
      {
        if ( iarg < argc && args[iarg].type == Argument_::INT )
        {
          prec = (int) args[iarg].intValue;
        }

        iarg++;
        fmt++;
      }
      else
      {
        while ( std::isdigit(*fmt) )
        {
          prec = 10 * prec + *fmt - '0';
          fmt++;

          if ( prec > 100 )
          {
            while ( std::isdigit(*fmt) ) fmt++;
          }
        }
      }

      if      ( prec < 0 )
      {
        prec = 0;
      }
      else if ( prec > 20 )
      {
        prec = 20;
      }
    }

    if ( *fmt == 'h' || *fmt == 'l')
    {
      fmt++;
    }

    if ( *fmt == '%' )
    {
      xbuf.pushBack ( '%' );
      continue;
    }

    if ( iarg >= argc )
    {
      goto invalid_arg;
    }

    switch ( *fmt )
    {
    case 'd':
    case 'i':
    case 'o':
    case 'u':
    case 'x':
    case 'X':

      {
        lint  value;

        if      ( args[iarg].type == Argument_::INT )
        {
          value = args[iarg].intValue;
        }
        else if ( args[iarg].type == Argument_::FLOAT )
        {
          value = (lint) args[iarg].fltValue;
        }
        else
        {
          goto invalid_arg;
        }

        if ( prec >= 0 )
        {
          flg[i++] = '.';
          flg[i++] = '*';
          flg[i++] = 'l';

          if ( sizeof(lint) > sizeof(long) )
          {
            flg[i++] = 'l';
          }

          flg[i++] = *fmt;
          flg[i++] = '\0';
          n        = std::sprintf ( buf, flg, prec, value );
        }
        else
        {
          flg[i++] = 'l';

          if ( sizeof(lint) > sizeof(long) )
          {
            flg[i++] = 'l';
          }

          flg[i++] = *fmt;
          flg[i++] = '\0';
          n        = std::sprintf ( buf, flg, value );
        }

        padAndAdd ( xbuf, width, buf, n );
      }

      break;

    case 'f':
    case 'g':
    case 'G':
    case 'e':
    case 'E':

      {
        char    fspec = *fmt;
        double  value;

        if      ( args[iarg].type == Argument_::INT )
        {
          value = (double) args[iarg].intValue;
        }
        else if ( args[iarg].type == Argument_::FLOAT )
        {
          value = args[iarg].fltValue;
        }
        else
        {
          goto invalid_arg;
        }

        if ( fspec == 'f' )
        {
          // This is to avoid a possible  buffer overflow.

          if ( std::fabs( value ) < 1.0e-10 ||
               std::fabs( value ) > 1.0e+10 )
          {
            fspec = 'g';
          }
        }

        if ( prec >= 0 )
        {
          flg[i++] = '.';
          flg[i++] = '*';
          flg[i++] = fspec;
          flg[i++] = '\0';
          n        = std::sprintf ( buf, flg, prec, value );
        }
        else
        {
          flg[i++] = fspec;
          flg[i++] = '\0';
          n        = std::sprintf ( buf, flg, value );
        }

        padAndAdd ( xbuf, width, buf, n );
      }

      break;

    case 'c':

      {
        int  value;

        if ( args[iarg].type == Argument_::INT )
        {
          value = (int) args[iarg].intValue;
        }
        else
        {
          goto invalid_arg;
        }

        if ( width > 1 )
        {
          xbuf.pushBack ( ' ', width - 1 );
        }

        xbuf.pushBack ( (char) value );

        if ( width < -1 )
        {
          xbuf.pushBack ( ' ', -width - 1 );
        }
      }

      break;

    case 's':

      {
        const char*  str;
        idx_t        len = 0;

        if      ( args[iarg].type == Argument_::CHARS )
        {
          str = (const char*) args[iarg].ptrValue;

          if ( str )
          {
            len = (idx_t) std::strlen ( str );
          }
        }
        else if ( args[iarg].type == Argument_::STRING )
        {
          str = ((const String*) args[iarg].ptrValue)->addr ();
          len = ((const String*) args[iarg].ptrValue)->size ();
        }
        else
        {
          goto invalid_arg;
        }

        if ( prec >= 0 && prec < len )
        {
          len = prec;
        }

        padAndAdd ( xbuf, width, str, len );
      }

      break;

    case 'p':

      {
        const void*  value = args[iarg].ptrValue;

        n = std::sprintf ( buf, "%p", value );

        padAndAdd ( xbuf, width, buf, n );
      }

      break;

    default:

      goto invalid_arg;
    }

    iarg++;
    continue;

invalid_arg:

    width = (width < 0)   ? -width : width;
    width = (width > 100) ? 100    : width;

    xbuf.pushBack ( '*', max( 3, width ) );
    iarg++;
  }
}


//-----------------------------------------------------------------------
//   padAndAdd
//-----------------------------------------------------------------------


void String::Utils_::padAndAdd

  ( Buffer_&     buf,
    idx_t        width,
    const char*  str,
    idx_t        len ) noexcept

{
  if ( width > len )
  {
    buf.pushBack ( ' ', width - len );
  }

#ifndef NDEBUG

  // Make sure that the string is valid.

  for ( idx_t i = 0; i < len; i++ )
  {
    if ( str[i] == '\0' )
    {
      std::fprintf (
        stderr,
        "FATAL ERROR in jem::String::format: "
        "unexpected null-character in String argument.\n"
      );

      std::fprintf ( stderr, "Aborting program.\n" );
      std::abort   ();
    }
  }

#endif

  buf.pushBack ( str, len );

  if ( -width > len )
  {
    buf.pushBack ( ' ', -width - len );
  }
}


//=======================================================================
//   Class String
//=======================================================================

//-----------------------------------------------------------------------
//   default constructor
//-----------------------------------------------------------------------


String::String () noexcept
{
  size_  = 0;
  chunk_ = nullptr;
  str_   = buffer_;
}


//-----------------------------------------------------------------------
//   primitive type conversion constructors
//-----------------------------------------------------------------------


String::String ( bool value ) noexcept
{
  if ( value )
  {
    Utils_::init ( this, 4 );

    str_[0] = 't';
    str_[1] = 'r';
    str_[2] = 'u';
    str_[3] = 'e';
  }
  else
  {
    Utils_::init ( this, 5 );

    str_[0] = 'f';
    str_[1] = 'a';
    str_[2] = 'l';
    str_[3] = 's';
    str_[4] = 'e';
  }
}


String::String ( char value ) noexcept
{
  Utils_::init ( this, 1 );

  str_[0] = value;
}


String::String ( byte value ) noexcept
{
  char   buf[BUFSIZE_];

  idx_t  len = Integer::print ( buf, BUFSIZE_, value, 16 );

  std::memcpy ( Utils_::init( this, len ), buf, (size_t) len );
}


String::String ( short value ) noexcept :

  String ( (lint) value )

{}


String::String ( int value ) noexcept :

  String ( (lint) value )

{}


String::String ( long value ) noexcept :

  String ( (lint) value )

{}


String::String ( lint value ) noexcept
{
  if ( value >= 0_lint && value < 10_lint )
  {
    size_   = 1;
    chunk_  = nullptr;
    str_    = buffer_;
    str_[0] = Character::LOWERCASE_DIGITS[value];
  }
  else
  {
    char   buf[BUFSIZE_];

    idx_t  len = Integer::print ( buf, BUFSIZE_, value );

    std::memcpy ( Utils_::init( this, len ), buf, (size_t) len );
  }
}


String::String ( idx_t value ) noexcept :

  String ( (lint) value )

{}


String::String ( float value ) noexcept :

  String ( (double) value )

{}


String::String ( double value ) noexcept
{
  char   buf[BUFSIZE_];

  idx_t  len = Float::print ( buf, BUFSIZE_, value );

  std::memcpy ( Utils_::init( this, len ), buf, (size_t) len );
}


//-----------------------------------------------------------------------
//   character array constructor
//-----------------------------------------------------------------------


String::String ( const char* rhs ) noexcept
{
  chunk_ = nullptr;

  if ( ! rhs )
  {
    size_ = 0;
    str_  = buffer_;
  }
  else
  {
    idx_t  n = 0;

    while ( rhs[n] != '\0' )
    {
      n++;
    }

    size_ = n;

    if ( n > TinyString::MAXLEN )
    {
      cstr_ = rhs;
    }
    else
    {
      str_ = buffer_;

      for ( idx_t i = 0; i < n; i++ )
      {
        str_[i] = rhs[i];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   character array copy constructor
//-----------------------------------------------------------------------


String::String ( const char* rhs, idx_t len ) noexcept
{
  JEM_ASSERT_NOTHROW ( len >= 0 );

  std::memcpy ( Utils_::init( this, len ), rhs, (size_t) len );
}


//-----------------------------------------------------------------------
//   tiny string copy constructor
//-----------------------------------------------------------------------


String::String ( const TinyString& str ) noexcept
{
  const int  mid = TinyString::MAXLEN / 2;
  int        n;


  chunk_ = (Chunk_*) str.data;
  str_   = buffer_;

  if ( str_[mid] == '\0' )
  {
    for ( n = 0; n < mid; n++ )
    {
      if ( str_[n] == '\0' )
      {
        break;
      }
    }
  }
  else
  {
    for ( n = mid + 1; n < TinyString::MAXLEN; n++ )
    {
      if ( str_[n] == '\0' )
      {
        break;
      }
    }
  }

  size_ = n;
}


//-----------------------------------------------------------------------
//   iterator copy constructors
//-----------------------------------------------------------------------


String::String ( char* first, char* last ) noexcept
{
  JEM_ASSERT_NOTHROW ( last >= first );

  const idx_t  len = (idx_t) (last - first);

  std::memcpy ( Utils_::init( this, len ), first, (size_t) len );
}


String::String

  ( const char*  first,
    const char*  last ) noexcept

{
  JEM_ASSERT_NOTHROW ( last >= first );

  const idx_t  len = (idx_t) (last - first);

  std::memcpy ( Utils_::init( this, len ), first, (size_t) len );
}


//-----------------------------------------------------------------------
//   string concatenation constructor
//-----------------------------------------------------------------------


String::String

  ( const String&  lhs,
    const String&  rhs ) noexcept

{
  char*  dest = Utils_::init ( this, lhs.size_ + rhs.size_ );

  std::memcpy ( dest, lhs.cstr_, (size_t) lhs.size_ );

  dest += lhs.size_;

  std::memcpy ( dest, rhs.cstr_, (size_t) rhs.size_ );
}


//-----------------------------------------------------------------------
//   copy constructor
//-----------------------------------------------------------------------


String::String ( const String& rhs ) noexcept
{
  size_  = rhs.size_;
  chunk_ = rhs.chunk_;

  if ( size_ <= TinyString::MAXLEN )
  {
    str_ = buffer_;
  }
  else
  {
    str_ = rhs.str_;

    if ( chunk_ )
    {
      Chunk_::addRef ( chunk_ );
    }
  }
}


//-----------------------------------------------------------------------
//   move constructor
//-----------------------------------------------------------------------


String::String ( String&& rhs ) noexcept
{
  size_      = rhs.size_;
  chunk_     = rhs.chunk_;
  rhs.size_  = 0;
  rhs.chunk_ = nullptr;

  if ( size_ <= TinyString::MAXLEN )
  {
    str_     = buffer_;
  }
  else
  {
    str_     = rhs.str_;
    rhs.str_ = rhs.buffer_;
  }
}


//-----------------------------------------------------------------------
//   constructors for the StringBuffer class
//-----------------------------------------------------------------------


String::String ( const Nil& nil, idx_t len ) noexcept
{
  JEM_ASSERT_NOTHROW ( len >= 0 );

  Utils_::init ( this, len );
}


//-----------------------------------------------------------------------
//   slice constructors
//-----------------------------------------------------------------------


String::String

  ( const String&  rhs,
    idx_t          first,
    idx_t          last ) noexcept

{
  char*  str = rhs.str_ + first;
  idx_t  len;

  if ( last > rhs.size_ )
  {
    last = rhs.size_;
  }

  len   = last - first;
  size_ = len;

  if ( len <= TinyString::MAXLEN )
  {
    chunk_ = nullptr;

    for ( idx_t i = 0; i < len; i++ )
    {
      buffer_[i] = str[i];
    }

    str_ = buffer_;
  }
  else
  {
    str_   = str;
    chunk_ = rhs.chunk_;

    if ( chunk_ )
    {
      Chunk_::addRef ( chunk_ );
    }
  }
}


String::String

  ( const String&  rhs,
    idx_t          first,
    idx_t          last,
    idx_t          stride ) noexcept

{
  idx_t  len;

  if ( last > rhs.size_ )
  {
    last = rhs.size_;
  }

  len = last - first;

  if ( len > 0 )
  {
    len = 1 + (len - 1) / stride;
  }

  Utils_::init ( this, len );

  for ( idx_t i = 0; first < last; first += stride, i++ )
  {
    str_[i] = rhs.str_[first];
  }
}


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


String::~String () noexcept
{
  Utils_::free ( this );
}


//-----------------------------------------------------------------------
//   readFrom (from a DataInput)
//-----------------------------------------------------------------------


void String::readFrom ( DataInput& in )
{
  uidx_t  sz;

  decode ( in, sz );

  idx_t  len = (idx_t) sz;

  Utils_::free ( this );

  in.decode ( Utils_::init( this, len ), len );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void String::writeTo ( DataOutput& out ) const
{
  out.encode ( size_ );
  out.encode ( str_, size_ );
}


//-----------------------------------------------------------------------
//   readFrom (from a TextInput)
//-----------------------------------------------------------------------


void String::readFrom ( TextInput& in )
{
  char  buf[BUFSIZE_];
  int   c;
  int   i;

  in.skipWhite ();

  c = in.read  ();
  i = 0;

  while ( i < BUFSIZE_ && c >= 0 && ! std::isspace(c) )
  {
    buf[i++] = (char) c;
    c        = in.read ();
  }

  if ( c != -1 && ! std::isspace(c) )
  {
    Buffer_ xbuf;

    xbuf.pushBack ( buf, i );

    while ( c != -1 && ! std::isspace(c) )
    {
      xbuf.pushBack ( (char) c );
      c = in.read   ();
    }

    Utils_::free ( this );

    std::memcpy ( Utils_::init( this, xbuf.size() ),
                  xbuf.addr(),
                  (size_t) xbuf.size() );
  }
  else
  {
    Utils_::free ( this );

    std::memcpy ( Utils_::init( this, i ), buf, (size_t) i );
  }

  in.pushBack (c);
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


String& String::operator = ( const String& rhs ) noexcept
{
  const idx_t  rsize = rhs.size_;

  if      ( size_ <= TinyString::MAXLEN &&
            rsize <= TinyString::MAXLEN )
  {
    chunk_ = rhs.chunk_;
  }
  else if ( size_ <= TinyString::MAXLEN )
  {
    chunk_ = rhs.chunk_;
    str_   = rhs.str_;

    if ( chunk_ )
    {
      Chunk_::addRef ( chunk_ );
    }
  }
  else if ( rsize <= TinyString::MAXLEN )
  {
    if ( chunk_ )
    {
      Chunk_::delRef ( chunk_ );
    }

    chunk_ = rhs.chunk_;
    str_   = buffer_;
  }
  else if ( this != & rhs )
  {
    if ( chunk_ )
    {
      Chunk_::delRef ( chunk_ );
    }

    chunk_ = rhs.chunk_;
    str_   = rhs.str_;

    if ( chunk_ )
    {
      Chunk_::addRef ( chunk_ );
    }
  }

  size_ = rsize;

  return *this;
}


String& String::operator = ( const char* rhs ) noexcept
{
  const idx_t  len = sizeOf ( rhs );

  Utils_::free ( this );

  chunk_ = nullptr;
  size_  = len;

  if ( len <= TinyString::MAXLEN )
  {
    for ( idx_t i = 0; i < len; i++ )
    {
      buffer_[i] = rhs[i];
    }

    str_ = buffer_;
  }
  else
  {
    cstr_ = rhs;
  }

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void String::swap ( String& rhs ) noexcept
{
  jem::swap ( chunk_, rhs.chunk_ );
  jem::swap ( size_,  rhs.size_  );
  jem::swap ( str_,   rhs.str_   );

  if ( size_     <= TinyString::MAXLEN )
  {
    str_ = buffer_;
  }

  if ( rhs.size_ <= TinyString::MAXLEN )
  {
    rhs.str_ = rhs.buffer_;
  }
}


//-----------------------------------------------------------------------
//   slice operators
//-----------------------------------------------------------------------


String String::operator [] ( const Slice& s ) const noexcept
{
  JEM_ASSERT_NOTHROW ( s.first() <= s.last( size() ) );

  if ( s.stride() == 1 )
  {
    return String ( *this, s.first(), s.last() );
  }
  else
  {
    return String ( *this, s.first(), s.last(), s.stride() );
  }
}


//-----------------------------------------------------------------------
//   hasNull
//-----------------------------------------------------------------------


bool String::hasNull () const noexcept
{
  // Note that this string can refer to part of another string
  // that is null terminated. Hence the explicit check for a
  // terminating null character.

  return (chunk_ == nullptr &&
          size_ > TinyString::MAXLEN && cstr_[size_] == '\0');
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint String::hashValue () const noexcept
{
  const char* JEM_RESTRICT  str = cstr_;

  const idx_t  len = size_;
  ulint        val = (ulint) len;

  for ( int i = 0; i < TinyString::MAXLEN; i++ )
  {
    val = val + ((val << 3) + (ulint) str[i]);
  }

  if ( len > TinyString::MAXLEN )
  {
    for ( idx_t i = TinyString::MAXLEN; i < len; i++ )
    {
      val = val + ((val << 3) + (ulint) str[i]);
    }
  }

  return (lint) val;
}


//-----------------------------------------------------------------------
//   toUpper
//-----------------------------------------------------------------------


String String::toUpper () const noexcept
{
  const idx_t  len = size_;

  String       tmp ( NIL, len );

  const char*  JEM_RESTRICT  s = cstr_;
  char*        JEM_RESTRICT  t = tmp.str_;

  for ( idx_t i = 0; i < len; i++ )
  {
    t[i] = (char) toupper ( s[i] );
  }

  return tmp;
}


//-----------------------------------------------------------------------
//   toLower
//-----------------------------------------------------------------------


String String::toLower () const noexcept
{
  const idx_t  len = size_;

  String       tmp ( NIL, len );

  const char*  JEM_RESTRICT  s = cstr_;
  char*        JEM_RESTRICT  t = tmp.str_;

  for ( idx_t i = 0; i < len; i++ )
  {
    t[i] = (char) tolower ( s[i] );
  }

  return tmp;
}


//-----------------------------------------------------------------------
//   stripWhite
//-----------------------------------------------------------------------


String String::stripWhite () const noexcept
{
  const char*  s = cstr_;

  idx_t  i = 0;
  idx_t  j = size_;

  while ( i < j && std::isspace( s[i] ) )
  {
    i++;
  }
  while ( j > i && std::isspace( s[j - 1]) )
  {
    j--;
  }

  return String ( *this, i, j );
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool String::equals ( const String& rhs ) const noexcept
{
  idx_t  len = rhs.size_;

  if ( size_ != len )
  {
    return false;
  }

  if ( len <= TinyString::MAXLEN )
  {
    return (chunk_ == rhs.chunk_);
  }

  const char* JEM_RESTRICT  s = cstr_;
  const char* JEM_RESTRICT  r = rhs.cstr_;

  char   c;
  idx_t  i;

  len--;

  c = (s[0] ^ r[0]) | (s[len] ^ r[len]);

  if ( c )
  {
    return false;
  }

  for ( i = 1; i < len - 3; i += 4 )
  {
    c = ( (s[i + 0] ^ r[i + 0]) |
          (s[i + 1] ^ r[i + 1]) |
          (s[i + 2] ^ r[i + 2]) |
          (s[i + 3] ^ r[i + 3]) );

    if ( c )
    {
      return false;
    }
  }

  switch ( len - i )
  {
  case 0:

    c = 0;

    break;

  case 1:

    c = s[i] ^ r[i];

    break;

  case 2:

    c = ( (s[i + 0] ^ r[i + 0]) |
          (s[i + 1] ^ r[i + 1]) );

    break;

  case 3:

    c = ( (s[i + 0] ^ r[i + 0]) |
          (s[i + 1] ^ r[i + 1]) |
          (s[i + 2] ^ r[i + 2]) );

    break;
  }

  return ! c;
}


bool String::equals ( const char* r ) const noexcept
{
  idx_t  len = r ? (int) std::strlen( r ) : 0;

  if ( size_ != len )
  {
    return false;
  }

  if ( len == 0 )
  {
    return true;
  }

  const char* JEM_RESTRICT  s = cstr_;

  char   c;
  idx_t  i;

  len--;

  c = (s[0] ^ r[0]) | (s[len] ^ r[len]);

  if ( c )
  {
    return false;
  }

  for ( i = 1; i < len - 3; i += 4 )
  {
    c = ( (s[i + 0] ^ r[i + 0]) |
          (s[i + 1] ^ r[i + 1]) |
          (s[i + 2] ^ r[i + 2]) |
          (s[i + 3] ^ r[i + 3]) );

    if ( c )
    {
      return false;
    }
  }

  switch ( len - i )
  {
  case 0:

    c = 0;

    break;

  case 1:

    c = s[i] ^ r[i];

    break;

  case 2:

    c = ( (s[i + 0] ^ r[i + 0]) |
          (s[i + 1] ^ r[i + 1]) );

    break;

  case 3:

    c = ( (s[i + 0] ^ r[i + 0]) |
          (s[i + 1] ^ r[i + 1]) |
          (s[i + 2] ^ r[i + 2]) );

    break;
  }

  return ! c;
}


//-----------------------------------------------------------------------
//   equalsIgnoreCase
//-----------------------------------------------------------------------


bool String::equalsIgnoreCase ( const String& rhs ) const noexcept
{
  const idx_t  len = size_;

  if ( len != rhs.size_ )
  {
    return false;
  }

  const char*  s = cstr_;
  const char*  r = rhs.cstr_;

  for ( idx_t i = 0; i < len; i++ )
  {
    if ( toupper( s[i] ) != toupper( r[i] ) )
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   compareTo
//-----------------------------------------------------------------------


int String::compareTo ( const String& rhs ) const noexcept
{
  const char*  s = cstr_;
  const char*  r = rhs.cstr_;

  const idx_t  n = min ( size_, rhs.size_ );

  idx_t        i;


  for ( i = 0; i < n; i++ )
  {
    if ( s[i] != r[i] )
    {
      break;
    }
  }

  if      ( i == size_ && i == rhs.size_ )
  {
    return  0;
  }
  else if ( i == size_ )
  {
    return -1;
  }
  else if ( i == rhs.size_ )
  {
    return  1;
  }
  else if ( s[i] < r[i] )
  {
    return -1;
  }
  else
  {
    return  1;
  }
}


//-----------------------------------------------------------------------
//   compareToIgnoreCase
//-----------------------------------------------------------------------


int String::compareToIgnoreCase ( const String& rhs ) const noexcept
{
  const char*  s = cstr_;
  const char*  r = rhs.cstr_;

  const idx_t  n = min ( size_, rhs.size_ );

  idx_t        i;


  for ( i = 0; i < n; i++ )
  {
    if ( toupper( s[i] ) != toupper( r[i] ) )
    {
      break;
    }
  }

  if      ( i == size_ && i == rhs.size_ )
  {
    return  0;
  }
  else if ( i == size_ )
  {
    return -1;
  }
  else if ( i == rhs.size_ )
  {
    return  1;
  }
  else if ( toupper( s[i] ) < toupper( r[i] ) )
  {
    return -1;
  }
  else
  {
    return 1;
  }
}


//-----------------------------------------------------------------------
//   startsWith
//-----------------------------------------------------------------------


bool String::startsWith ( const String& word ) const noexcept
{
  const idx_t  k = word.size_ - 1;
  const idx_t  n = size_      - 1;

  if ( k < 0 || k > n )
  {
    return false;
  }

  const char*  s = cstr_;
  const char*  r = word.cstr_;

  if ( (s[0] ^ r[0]) | (s[k] ^ r[k]) )
  {
    return false;
  }

  for ( idx_t i = 1; i < k; i++ )
  {
    if ( s[i] != r[i] )
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   endsWith
//-----------------------------------------------------------------------


bool String::endsWith ( const String& word ) const noexcept
{
  const idx_t  k = word.size_ - 1;
  const idx_t  n = size_      - 1;

  if ( k < 0 || k > n )
  {
    return false;
  }

  const char*  s = cstr_ + (n - k);
  const char*  r = word.cstr_;

  if ( (s[0] ^ r[0]) | (s[k] ^ r[k]) )
  {
    return false;
  }

  for ( idx_t i = 1; i < k; i++ )
  {
    if ( s[i] != r[i] )
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   find (a substring)
//-----------------------------------------------------------------------


idx_t String::find

  ( const String&  needle,
    idx_t          fromIndex ) const noexcept

{
  const idx_t  k = needle.size_ - 1;
  const idx_t  n = size_        - k;

  if ( k < 0 || n <= 0 )
  {
    return -1;
  }

  const char*  s = cstr_;
  const char*  r = needle.cstr_;

  const char   a = r[0];
  const char   b = r[k];

  idx_t        i, j;


  i  = max ( 0_idx, fromIndex );
  s += i;

  while ( i < n )
  {
    if ( (s[0] ^ a) | (s[k] ^ b) )
    {
      goto next_iter;
    }

    for ( j = 1; j < k; j++ )
    {
      if ( s[j] != r[j] )
      {
        goto next_iter;
      }
    }

    return i;

   next_iter:

    i++;
    s++;
  }

  return -1;
}


//-----------------------------------------------------------------------
//   find (a character)
//-----------------------------------------------------------------------


idx_t String::find ( char needle, idx_t fromIndex ) const noexcept
{
  const char*  s = cstr_;
  const idx_t  n = size_;


  for ( idx_t i = max( 0_idx, fromIndex ); i < n; i++ )
  {
    if ( s[i] == needle )
    {
      return i;
    }
  }

  return -1;
}


//-----------------------------------------------------------------------
//   rfind (from the end of the string)
//-----------------------------------------------------------------------


idx_t String::rfind ( const String& needle ) const noexcept
{
  return rfind ( needle, size_ );
}


idx_t String::rfind ( char needle ) const noexcept
{
  return rfind ( needle, size_ );
}


//-----------------------------------------------------------------------
//   rfind (from a given index)
//-----------------------------------------------------------------------


idx_t String::rfind

  ( const String&  needle,
    idx_t          fromIndex ) const noexcept

{
  const idx_t  k = needle.size_ - 1;
  const idx_t  n = size_ - needle.size_;

  if ( k < 0 || n < 0 )
  {
    return -1;
  }

  const char*  s = cstr_;
  const char*  r = needle.cstr_;

  const char   a = r[0];
  const char   b = r[k];

  idx_t         i, j;


  i  = min ( n, fromIndex );
  s += i;

  while ( i >= 0 )
  {
    if ( (s[0] ^ a) | (s[k] ^ b) )
    {
      goto next_iter;
    }

    for ( j = 1; j < k; j++ )
    {
      if ( s[j] != r[j] )
      {
        goto next_iter;
      }
    }

    return i;

   next_iter:

    i--;
    s--;
  }

  return i;
}


idx_t String::rfind ( char needle, idx_t fromIndex ) const noexcept
{
  const char*  s = cstr_;

  for ( idx_t i = min( size_ - 1, fromIndex ); i >= 0; i-- )
  {
    if ( s[i] == needle )
    {
      return i;
    }
  }

  return -1;
}


//-----------------------------------------------------------------------
//   sizeOf
//-----------------------------------------------------------------------


idx_t String::sizeOf ( const char* str ) noexcept
{
  idx_t  len = 0;

  if ( str )
  {
    while ( str[len] != '\0' )
    {
      len++;
    }
  }

  return len;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


char* String::init_ ( idx_t len ) noexcept
{
  return Utils_::init ( this, len );
}


//-----------------------------------------------------------------------
//   free_
//-----------------------------------------------------------------------


void String::free_ () noexcept
{
  Utils_::free ( this );
}


//-----------------------------------------------------------------------
//   realloc_
//-----------------------------------------------------------------------


void String::realloc_ ( idx_t len ) noexcept
{
  if ( size_ <= TinyString::MAXLEN )
  {
    if ( len > TinyString::MAXLEN )
    {
      Chunk_*  c = Chunk_::alloc ( &str_, len );

      for ( int i = 0; i < TinyString::MAXLEN; i++ )
      {
        str_[i] = buffer_[i];
      }

      chunk_ = c;
    }
  }
  else
  {
    if ( len <= TinyString::MAXLEN )
    {
      Chunk_*  c = chunk_;

      for ( int i = 0; i < TinyString::MAXLEN; i++ )
      {
        buffer_[i] = cstr_[i];
      }

      if ( c )
      {
        Chunk_::delRef ( c );
      }

      str_ = buffer_;
    }
    else
    {
      if ( chunk_ )
      {
        chunk_ = Chunk_::realloc ( chunk_, len );
        str_   = chunk_->getData ();
      }
      else
      {
        const char*  s = cstr_;

        chunk_ = Chunk_::alloc ( &str_, len );

        std::memcpy ( str_, s, (size_t) min( len, size_ ) );
      }
    }
  }

  size_ = len;
}


//-----------------------------------------------------------------------
//   format_
//-----------------------------------------------------------------------


String String::format_

  ( const char*       fmt,
    const Argument_*  args,
    int               argc ) noexcept

{
  if ( ! fmt )
  {
    return String ();
  }
  else
  {
    Buffer_  buf;

    Utils_::format ( buf, fmt, args, argc );

    return String ( buf.addr(), buf.size() );
  }
}


//=======================================================================
//   Related functions
//=======================================================================

//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


String  clone ( const char* str ) noexcept
{
  if ( str )
  {
    return String ( str, String::sizeOf( str ) );
  }
  else
  {
    return String ();
  }
}


JEM_END_PACKAGE_BASE
