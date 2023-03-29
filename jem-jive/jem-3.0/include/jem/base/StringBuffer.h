
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

#ifndef JEM_BASE_STRINGBUFFER_H
#define JEM_BASE_STRINGBUFFER_H

#include <jem/io/TextOutput.h>
#include <jem/base/String.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class StringBuffer
//-----------------------------------------------------------------------


class StringBuffer : public io::TextOutput
{
 public:

  typedef StringBuffer    Self;

  typedef char            ValueType;
  typedef char*           Iterator;
  typedef const char*     ConstIterator;


                          StringBuffer  () noexcept;

  explicit                StringBuffer

    ( idx_t                 size );

  explicit                StringBuffer

    ( const String&         str );

                          StringBuffer

    ( const Self&           rhs );

                          StringBuffer

    ( Self&&                rhs )          noexcept;

  virtual void            write

    ( int                   c )            override;

  virtual void            write

    ( int                   c,
      idx_t                 n )            override;

  virtual void            write

    ( const char*           buf,
      idx_t                 len )          override;

  virtual void            printByte

    ( byte                  b )            override;

  virtual void            printBool

    ( bool                  b )            override;

  virtual void            printInt

    ( lint                  i )            override;

  virtual void            printFloat

    ( double                d )            override;

  Self&                   operator       =

    ( const Self&           rhs );

  Self&                   operator       =

    ( Self&&                rhs )          noexcept;

  Self&                   operator       =

    ( const String&         rhs );

  template <class T>
    inline Self&          operator      +=

    ( const T&              rhs );

  Self&                   operator      +=

    ( const String&         rhs );

  Self&                   operator      +=

    ( char                  rhs );

  inline char&            operator      []

    ( idx_t                 i );

  inline char             operator      []

    ( idx_t                 i )            const;

  String                  operator      []

    ( const Slice&          s )            const;

  inline String           operator      []

    ( const SliceFromTo&    s )            const;

  inline String           operator      []

    ( const SliceTo&        s )            const;

  inline String           operator      []

    ( const SliceFrom&      s )            const;

  inline String           operator      []

    ( const SliceAll&       s )            const;

  inline idx_t            size          () const noexcept;
  inline idx_t            capacity      () const noexcept;

  inline Iterator         begin         ();
  inline ConstIterator    begin         () const noexcept;
  inline Iterator         end           ();
  inline ConstIterator    end           () const noexcept;

  inline char&            front         ();
  inline char             front         () const;
  inline char&            back          ();
  inline char             back          () const;

  inline char*            addr          ();
  inline const char*      addr          () const noexcept;

  void                    swap

    ( Self&                 rhs )                noexcept;

  void                    clear         ();

  void                    resize

    ( idx_t                 size );

  inline char*            expand

    ( idx_t                 len );

  inline char*            xalloc

    ( idx_t                 len );

  inline void             commit

    ( idx_t                 len );

  void                    reserve

    ( idx_t                 cap );

  String                  toString      ();


 private:

  void                    clone_        ();

  char*                   xalloc_

    ( idx_t                 len );

  void                    realloc_

    ( idx_t                 cap );


 private:

  String                  str_;
  idx_t                   size_;
  bool                    owner_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline void               swap

  ( StringBuffer&           lhs,
    StringBuffer&           rhs ) noexcept;

void                      decode

  ( io::DataInput&          in,
    StringBuffer&           sb );

void                      decode

  ( io::DataOutput&         out,
    const StringBuffer&     sb );

void                      read

  ( io::TextInput&          in,
    StringBuffer&           sb );

inline void               print

  ( io::TextOutput&         out,
    const StringBuffer&     sb );




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class StringBuffer
//=======================================================================

//-----------------------------------------------------------------------
//   operator +=
//-----------------------------------------------------------------------


template <class T>

  inline StringBuffer& StringBuffer::operator += ( const T& rhs )

{
  print ( *this, rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   subscript operators
//-----------------------------------------------------------------------


inline char& StringBuffer::operator [] ( idx_t i )
{
  JEM_ASSERT2 ( checkIndex( i, size() ),
                "invalid StringBuffer index" );

  return addr()[i];
}


inline char  StringBuffer::operator [] ( idx_t i ) const
{
  JEM_ASSERT2 ( checkIndex( i, size() ),
                "invalid StringBuffer index" );

  return str_.cstr_[i];
}


//-----------------------------------------------------------------------
//   slice operators
//-----------------------------------------------------------------------


inline String StringBuffer::operator []

  ( const SliceFromTo&  s ) const

{
  JEM_ASSERT2 ( s.first() <= s.last() && s.last() <= size(),
                "invalid StringBuffer slice" );

  idx_t  len = s.last() - s.first();

  return String ( str_.cstr_ + s.first(), len );
}


inline String StringBuffer::operator []

  ( const SliceTo&  s ) const

{
  return String ( str_.cstr_, s.last( size_ ) );
}


inline String StringBuffer::operator []

  ( const SliceFrom&  s ) const

{
  JEM_ASSERT2 ( s.first() <= size(),
                "invalid StringBuffer slice" );

  idx_t  len = size_ - s.first();

  return String ( str_.cstr_ + s.first(), len );
}


inline String StringBuffer::operator []

  ( const SliceAll&  s ) const

{
  return String ( str_.cstr_, size_ );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t StringBuffer::size () const noexcept
{
  return size_;
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


inline idx_t StringBuffer::capacity () const noexcept
{
  return str_.size_;
}


//-----------------------------------------------------------------------
//   begin
//-----------------------------------------------------------------------


inline char* StringBuffer::begin ()
{
  return addr ();
}


inline const char* StringBuffer::begin () const noexcept
{
  return addr ();
}


//-----------------------------------------------------------------------
//   end
//-----------------------------------------------------------------------


inline char* StringBuffer::end ()
{
  return addr() +  size_;
}


inline const char* StringBuffer::end () const noexcept
{
  return addr() + size_;
}


//-----------------------------------------------------------------------
//   front
//-----------------------------------------------------------------------


inline char& StringBuffer::front ()
{
  JEM_ASSERT2 ( size() > 0, "empty StringBuffer" );

  return addr()[0];
}


inline char StringBuffer::front () const
{
  JEM_ASSERT2 ( size() > 0, "empty StringBuffer" );

  return addr()[0];
}


//-----------------------------------------------------------------------
//   back
//-----------------------------------------------------------------------


inline char& StringBuffer::back ()
{
  JEM_ASSERT2 ( size() > 0, "empty StringBuffer" );

  return addr()[size_ - 1_idx];
}


inline char StringBuffer::back () const
{
  JEM_ASSERT2 ( size() > 0, "empty StringBuffer" );

  return addr()[size_ - 1_idx];
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


inline char* StringBuffer::addr ()
{
  if ( ! owner_ )
  {
    clone_ ();
  }

  return str_.str_;
}


inline const char* StringBuffer::addr () const noexcept
{
  return str_.cstr_;
}


//-----------------------------------------------------------------------
//   expand
//-----------------------------------------------------------------------


inline char* StringBuffer::expand ( idx_t len )
{
  char*  ptr = xalloc ( len );

  size_ += len;

  return ptr;
}


//-----------------------------------------------------------------------
//   xalloc
//-----------------------------------------------------------------------


inline char* StringBuffer::xalloc ( idx_t len )
{
  JEM_ASSERT2 ( len >= 0, "negative allocation size" );

  if ( size_ + len > str_.size_ )
  {
    return xalloc_ ( len );
  }
  else
  {
    return addr() + size_;
  }
}


//-----------------------------------------------------------------------
//   commit
//-----------------------------------------------------------------------


inline void StringBuffer::commit ( idx_t len )
{
  JEM_ASSERT2 ( len >= 0 && size() + len <= capacity(),
                "invalid commit length" );

  size_ += len;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void               swap

  ( StringBuffer&           lhs,
    StringBuffer&           rhs ) noexcept

{
  lhs.swap ( rhs );
}


//-----------------------------------------------------------------------
//   print operator
//-----------------------------------------------------------------------


inline void               print

  ( io::TextOutput&         out,
    const StringBuffer&     sb )

{
  out.write ( sb.addr(), sb.size() );
}


JEM_END_PACKAGE_BASE

#endif
