
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

#ifndef JEM_BASE_STRING_H
#define JEM_BASE_STRING_H

#include <jem/base/assert.h>
#include <jem/base/Slice.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   forward declarations
//-----------------------------------------------------------------------


class Nil;
class String;
class StringBuffer;

namespace io
{
  class TextInput;
  class DataInput;
  class DataOutput;
}


//-----------------------------------------------------------------------
//   class TinyString
//-----------------------------------------------------------------------


class TinyString
{
 public:

  static const int      MAXLEN = sizeof(void*);


  explicit inline       TinyString

    ( void*               str )           noexcept;

  explicit inline       TinyString

    ( const char          str[MAXLEN] )  noexcept;


 public:

  union
  {
    void* const         data;
    char                chars[MAXLEN];
  };

};


//-----------------------------------------------------------------------
//   class String
//-----------------------------------------------------------------------


class String
{
 public:

  friend class          StringBuffer;

  typedef char          ValueType;
  typedef const char*   Iterator;
  typedef const char*   ConstIterator;


                        String      ()       noexcept;

  explicit              String

    ( bool                value )            noexcept;

  explicit              String

    ( char                value )            noexcept;

  explicit              String

    ( byte                value )            noexcept;

  explicit              String

    ( short               value )            noexcept;

  explicit              String

    ( int                 value )            noexcept;

  explicit              String

    ( long                value )            noexcept;

  explicit              String

    ( lint                value )            noexcept;

  explicit              String

    ( idx_t               value )            noexcept;

  explicit              String

    ( float               value )            noexcept;

  explicit              String

    ( double              value )            noexcept;

                        String

    ( const char*         str )              noexcept;

                        String

    ( const char*         str,
      idx_t               len )              noexcept;

                        String

    ( const TinyString&   str )              noexcept;

                        String

    ( char*               first,
      char*               last )             noexcept;

                        String

    ( const char*         first,
      const char*         last )             noexcept;

  template <class InputIterator>

                        String

    ( InputIterator       first,
      InputIterator       last );

                        String

    ( const String&       lhs,
      const String&       rhs )              noexcept;

                        String

    ( const String&       rhs )              noexcept;

                        String

    ( String&&            rhs )              noexcept;

                       ~String      ()       noexcept;

  void                  readFrom

    ( io::DataInput&      in );

  void                  writeTo

    ( io::DataOutput&     out )        const;

  void                  readFrom

    ( io::TextInput&      in );

  String&               operator =

    ( const String&       rhs )              noexcept;

  inline String&        operator =

    ( String&&            rhs )              noexcept;

  String&               operator =

    ( const char*         rhs )              noexcept;

  void                  swap

    ( String&             rhs )              noexcept;

  inline Iterator       begin       () const noexcept;
  inline Iterator       end         () const noexcept;

  inline char           front       () const noexcept;
  inline char           back        () const noexcept;

  template <class I>
    inline char         operator    []

    ( I                   i )          const noexcept;

  String                operator    []

    ( const Slice&        s )          const noexcept;

  inline String         operator    []

    ( const SliceFromTo&  s )          const noexcept;

  inline String         operator    []

    ( const SliceTo&      s )          const noexcept;

  inline String         operator    []

    ( const SliceFrom&    s )          const noexcept;

  inline String         operator    []

    ( const SliceAll&     s )           const noexcept;

  inline idx_t          size         () const noexcept;

  inline const char*    addr

    ( idx_t               i = 0 )       const noexcept;

  bool                  hasNull      () const noexcept;
  lint                  hashValue    () const noexcept;
  String                toUpper      () const noexcept;
  String                toLower      () const noexcept;
  String                stripWhite   () const noexcept;

  bool                  equals

    ( const String&       rhs )        const noexcept;

  bool                  equals

    ( const char*         rhs )        const noexcept;

  bool                  equalsIgnoreCase

    ( const String&       rhs )        const noexcept;

  int                   compareTo

    ( const String&       rhs )        const noexcept;

  int                   compareToIgnoreCase

    ( const String&       rhs )        const noexcept;

  bool                  startsWith

    ( const String&       word )       const noexcept;

  inline bool           startsWith

    ( char                ch )         const noexcept;

  bool                  endsWith

    ( const String&       word )       const noexcept;

  inline bool           endsWith

    ( char                ch )         const noexcept;

  idx_t                 find

    ( const String&       needle,
      idx_t               index = 0 )  const noexcept;

  idx_t                 find

    ( char                needle,
      idx_t               index = 0 )  const noexcept;

  idx_t                 rfind

    ( const String&       needle )     const noexcept;

  idx_t                 rfind

    ( const String&       needle,
      idx_t               fromIndex )  const noexcept;

  idx_t                 rfind

    ( char                needle )     const noexcept;

  idx_t                 rfind

    ( char                needle,
      idx_t               fromIndex )  const noexcept;

  static idx_t          sizeOf

    ( const char*         str )              noexcept;

  static inline String  format

    ( const char*         fmt )              noexcept;

  template <typename... Args>
    static String       format

    ( const char*         fmt,
      const Args&...      args )             noexcept;


 private:

  class                 Argument_;


                        String

    ( const Nil&          nil,
      idx_t               len )              noexcept;

                        String

    ( const String&       rhs,
      idx_t               first,
      idx_t               last )             noexcept;

                        String

    ( const String&       rhs,
      idx_t               first,
      idx_t               last,
      idx_t               stride )           noexcept;

  char*                 init_

    ( idx_t               size )             noexcept;

  void                  free_       ()       noexcept;

  void                  realloc_

    ( idx_t               len )              noexcept;

  static String         format_

    ( const char*         fmt,
      const Argument_*    args,
      int                 argc )             noexcept;


 private:

  class                 Buffer_;
  class                 Chunk_;
  class                 Utils_;

  friend class          Chunk_;
  friend class          Utils_;

  static const int      BUFSIZE_ = 128;

  idx_t                 size_;

  union
  {
    char*               str_;
    const char*         cstr_;
  };

  union
  {
    Chunk_*             chunk_;
    char                buffer_[TinyString::MAXLEN];
  };

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


String                  clone

  ( const char*           str ) noexcept;

inline void             swap

  ( String&               lhs,
    String&               rhs ) noexcept;

template <class Input>

  inline void           decode

  ( Input&                in,
    String&               s );

template <class Output>

  inline void           encode

  ( Output&               out,
    const String&         s );

template <class Input>

  inline void           read

  ( Input&                in,
    String&               s );

template <class Output>

  inline void           print

  ( Output&               out,
    const String&         s );




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class TinyString
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline TinyString::TinyString ( void* str ) noexcept :

  data ( str )

{}


inline TinyString::TinyString ( const char str[MAXLEN] ) noexcept
{
  for ( int i = 0; i < MAXLEN; i++ )
  {
    chars[i] = str[i];
  }
}


//=======================================================================
//   class String::Argument_
//=======================================================================


class String::Argument_
{
 public:

  enum                  Type
  {
                          INT,
                          FLOAT,
                          CHARS,
                          STRING,
                          POINTER
  };

  inline                Argument_

    ( bool                arg )        noexcept;

  inline                Argument_

    ( char                arg )        noexcept;

  inline                Argument_

    ( byte                arg )        noexcept;

  inline                Argument_

    ( short               arg )        noexcept;

  inline                Argument_

    ( int                 arg )        noexcept;

  inline                Argument_

    ( long                arg )        noexcept;

  inline                Argument_

    ( lint                arg )        noexcept;

  inline                Argument_

    ( idx_t               arg )        noexcept;

  inline                Argument_

    ( float               arg )        noexcept;

  inline                Argument_

    ( double              arg )        noexcept;

  inline                Argument_

    ( const char*         arg )        noexcept;

  inline                Argument_

    ( const String&       arg )        noexcept;

  inline                Argument_

    ( const void*         arg )        noexcept;


 public:

  Type                  type;

  union
  {
    lint                intValue;
    double              fltValue;
    const void*         ptrValue;
  };

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline String::Argument_::Argument_  ( bool arg ) noexcept
{
  type     = INT;
  intValue = (lint) arg;
}


inline String::Argument_::Argument_ ( char arg ) noexcept
{
  type     = INT;
  intValue = arg;
}


inline String::Argument_::Argument_ ( byte arg ) noexcept
{
  type     = INT;
  intValue = arg;
}


inline String::Argument_::Argument_ ( short arg ) noexcept
{
  type     = INT;
  intValue = arg;
}


inline String::Argument_::Argument_ ( int arg ) noexcept
{
  type     = INT;
  intValue = arg;
}


inline String::Argument_::Argument_ ( long arg ) noexcept
{
  type     = INT;
  intValue = arg;
}


inline String::Argument_::Argument_ ( lint arg ) noexcept
{
  type     = INT;
  intValue = arg;
}


inline String::Argument_::Argument_ ( idx_t arg ) noexcept
{
  type     = INT;
  intValue = (lint) arg;
}


inline String::Argument_::Argument_ ( float arg ) noexcept
{
  type     = FLOAT;
  fltValue = arg;
}


inline String::Argument_::Argument_ ( double arg ) noexcept
{
  type     = FLOAT;
  fltValue = arg;
}


inline String::Argument_::Argument_ ( const char* arg ) noexcept
{
  type     = CHARS;
  ptrValue = arg;
}


inline String::Argument_::Argument_ ( const String& arg ) noexcept
{
  type     = STRING;
  ptrValue = &arg;
}


inline String::Argument_::Argument_ ( const void* arg ) noexcept
{
  type     = POINTER;
  ptrValue = arg;
}


//=======================================================================
//   class String
//=======================================================================

//-----------------------------------------------------------------------
//   iterator constructor
//-----------------------------------------------------------------------


template <class InputIterator>

  String::String

  ( InputIterator  first,
    InputIterator  last )

{
  InputIterator  i = first;
  char*          dest;
  idx_t          n;

  for ( n = 0; i != last; ++i )
  {
    n++;
  }

  dest = init_ ( n );

  try
  {
    for ( ; first != last; ++first, dest++ )
    {
      *dest = *first;
    }
  }
  catch ( ... )
  {
    free_ ();
    throw;
  }
}


//-----------------------------------------------------------------------
//   move operator
//-----------------------------------------------------------------------


inline String& String::operator = ( String&& rhs ) noexcept
{
  swap ( rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   begin & end
//-----------------------------------------------------------------------


inline String::Iterator String::begin () const noexcept
{
  return cstr_;
}


inline String::Iterator String::end () const noexcept
{
  return (cstr_ + size_);
}


//-----------------------------------------------------------------------
//   front & back
//-----------------------------------------------------------------------


inline char String::front () const noexcept
{
  JEM_ASSERT_NOTHROW ( size() > 0 );

  return str_[0];
}


inline char String::back () const noexcept
{
  JEM_ASSERT_NOTHROW ( size() > 0 );

  return str_[size_ - 1];
}


//-----------------------------------------------------------------------
//   subscript operator
//-----------------------------------------------------------------------


template <class I>

  inline char String::operator [] ( I i ) const noexcept

{
  JEM_ASSERT_NOTHROW ( checkIndex( i, size() ) );

  return cstr_[i];
}


//-----------------------------------------------------------------------
//   slice operators
//-----------------------------------------------------------------------


inline String String::operator [] ( const SliceFromTo& s ) const noexcept
{
  JEM_ASSERT_NOTHROW ( s.first() <= s.last() && s.last() <= size() );

  return String ( *this, s.first(), s.last() );
}


inline String String::operator [] ( const SliceTo& s ) const noexcept
{
  return String ( *this, 0, s.last() );
}


inline String String::operator [] ( const SliceFrom& s ) const noexcept
{
  JEM_ASSERT_NOTHROW ( s.first() <= size() );

  return String ( *this, s.first(), size_ );
}


inline String String::operator [] ( const SliceAll& ) const noexcept
{
  return *this;
}


//-----------------------------------------------------------------------
//   startsWith
//-----------------------------------------------------------------------


inline bool String::startsWith ( char ch ) const noexcept
{
  return (size_ > 0 && cstr_[0] == ch);
}


//-----------------------------------------------------------------------
//   endsWith
//-----------------------------------------------------------------------


inline bool String::endsWith ( char ch ) const noexcept
{
  return (size_ > 0 && cstr_[size_ - 1] == ch);
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t String::size () const noexcept
{
  return size_;
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


inline const char* String::addr ( idx_t i ) const noexcept
{
  return (cstr_ + i);
}


//-----------------------------------------------------------------------
//   format
//-----------------------------------------------------------------------


inline String String::format ( const char* fmt ) noexcept
{
  return format_ ( fmt, 0, 0 );
}

template <typename... Args>

  String String::format

  ( const char*     fmt,
    const Args&...  args ) noexcept

{
  Argument_  arr[sizeof...(args)] = { args... };

  return format_ ( fmt, arr, sizeof...(args) );
}


//=======================================================================
//   Related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void       swap

  ( String&         lhs,
    String&         rhs ) noexcept

{
  lhs.swap ( rhs );
}


//-----------------------------------------------------------------------
//   operator +
//-----------------------------------------------------------------------


inline String     operator +

  ( const String&   lhs,
    const String&   rhs ) noexcept

{
  return String ( lhs, rhs );
}


inline String     operator +

  ( const String&   lhs,
    const char*     rhs ) noexcept

{
  return String ( lhs, String( rhs ) );
}


inline String     operator +

  ( const char*     lhs,
    const String&   rhs ) noexcept

{
  return String ( String( lhs ), rhs );
}


inline String     operator +

  ( const String&   lhs,
    char            rhs ) noexcept

{
  return String ( lhs, String( rhs ) );
}


inline String     operator +

  ( char            lhs,
    const String&   rhs ) noexcept

{
  return String ( String( lhs ), rhs );
}


//-----------------------------------------------------------------------
//   operator ==
//-----------------------------------------------------------------------


inline bool       operator ==

  ( const String&   lhs,
    const String&   rhs ) noexcept

{
  return (lhs.size() == rhs.size() && lhs.equals( rhs ));
}


inline bool       operator ==

  ( const String&   lhs,
    const char*     rhs ) noexcept

{
  return lhs.equals ( rhs );
}


inline bool       operator ==

  ( const char*     lhs,
    const String&   rhs ) noexcept

{
  return rhs.equals ( lhs );
}


//-----------------------------------------------------------------------
//   operator !=
//-----------------------------------------------------------------------


inline bool       operator !=

  ( const String&   lhs,
    const String&   rhs ) noexcept

{
  return ! lhs.equals ( rhs );
}


inline bool       operator !=

  ( const String&   lhs,
    const char*     rhs ) noexcept

{
  return ! lhs.equals ( rhs );
}


inline bool       operator !=

  ( const char*     lhs,
    const String&   rhs ) noexcept

{
  return ! rhs.equals ( lhs );
}


//-----------------------------------------------------------------------
//   operator <
//-----------------------------------------------------------------------


inline bool       operator <

  ( const String&   lhs,
    const String&   rhs ) noexcept

{
  return (lhs.compareTo( rhs ) < 0);
}


inline bool       operator <

  ( const String&   lhs,
    const char*     rhs ) noexcept

{
  return (lhs.compareTo( rhs ) < 0);
}


inline bool       operator <

  ( const char*     lhs,
    const String&   rhs ) noexcept

{
  return (rhs.compareTo( lhs ) > 0);
}


//-----------------------------------------------------------------------
//   operator <=
//-----------------------------------------------------------------------


inline bool       operator <=

  ( const String&   lhs,
    const String&   rhs ) noexcept

{
  return (lhs.compareTo( rhs ) <= 0);
}


inline bool       operator <=

  ( const String&   lhs,
    const char*     rhs ) noexcept

{
  return (lhs.compareTo( rhs ) <= 0);
}


inline bool       operator <=

  ( const char*     lhs,
    const String&   rhs ) noexcept

{
  return (rhs.compareTo( lhs ) >= 0);
}


//-----------------------------------------------------------------------
//   operator >
//-----------------------------------------------------------------------


inline bool       operator >

  ( const String&   lhs,
    const String&   rhs ) noexcept

{
  return (lhs.compareTo( rhs ) > 0);
}


inline bool       operator >

  ( const String&   lhs,
    const char*     rhs ) noexcept

{
  return (lhs.compareTo( rhs ) > 0);
}


inline bool       operator >

  ( const char*     lhs,
    const String&   rhs ) noexcept

{
  return (rhs.compareTo( lhs ) < 0);
}


//-----------------------------------------------------------------------
//   operator >=
//-----------------------------------------------------------------------


inline bool       operator >=

  ( const String&   lhs,
    const String&   rhs ) noexcept

{
  return (lhs.compareTo( rhs ) >= 0);
}


inline bool       operator >=

  ( const String&   lhs,
    const char*     rhs ) noexcept

{
  return (lhs.compareTo( rhs ) >= 0);
}


inline bool       operator >=

  ( const char*     lhs,
    const String&   rhs ) noexcept

{
  return (rhs.compareTo( lhs ) <= 0);
}


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input>

  inline void     decode

  ( Input&          in,
    String&         s )

{
  s.readFrom ( in );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void     encode

  ( Output&         out,
    const String&   s )

{
  s.writeTo ( out );
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


template <class Input>

  inline void     read

  ( Input&          in,
    String&         s )

{
  s.readFrom ( in );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output>

  inline void     print

  ( Output&         out,
    const String&   s )

{
  out.write ( s.addr(), s.size() );
}


JEM_END_PACKAGE_BASE

#endif
