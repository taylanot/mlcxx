
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

#ifndef JEM_IO_XDR_H
#define JEM_IO_XDR_H

#include <climits>
#include <jem/defines.h>


#undef LITTLE_ENDIAN
#undef BIG_ENDIAN

JEM_BEGIN_PACKAGE   ( io )
JEM_BEGIN_NAMESPACE ( xdr )


//-----------------------------------------------------------------------
//   public constants
//-----------------------------------------------------------------------


enum  { XDR_BOOL_SIZE   = 1 };
enum  { XDR_BYTE_SIZE   = 1 };
enum  { XDR_CHAR_SIZE   = 1 };
enum  { XDR_SHORT_SIZE  = 2 };
enum  { XDR_INT_SIZE    = 4 };
enum  { XDR_LONG_SIZE   = 8 };
enum  { XDR_FLOAT_SIZE  = 4 };
enum  { XDR_DOUBLE_SIZE = 8 };

enum  { LITTLE_ENDIAN   = 0 };
enum  { BIG_ENDIAN      = 1 };


//-----------------------------------------------------------------------
//   public functions
//-----------------------------------------------------------------------


int               getByteOrder ();

template <class T>
  inline void     encode

  ( byte*           buf,
    const T&        value,
    int             sex )         noexcept;

template <class T>
  void            encode

  ( byte*           buf,
    const T*        values,
    idx_t           n,
    int             sex )         noexcept;

template <class T>
  inline void     decode

  ( T&              value,
    const byte*     buf,
    int             sex )         noexcept;

template <class T>
  void            decode

  ( T*              values,
    idx_t           n,
    const byte*     buf,
    int             sex )         noexcept;


//-----------------------------------------------------------------------
//   class TypeTraits & specializations
//-----------------------------------------------------------------------


template <class T>
  class TypeTraits {};


template <>
  class TypeTraits<bool>
{
  public:

    enum { XDR_SIZE    = XDR_BOOL_SIZE };
    enum { NATIVE_SIZE = sizeof(bool)  };

};


template <>
  class TypeTraits<byte>
{
  public:

    enum { XDR_SIZE    = XDR_BYTE_SIZE };
    enum { NATIVE_SIZE = sizeof(byte)  };

};


template <>
  class TypeTraits<char>
{
  public:

    enum { XDR_SIZE    = XDR_CHAR_SIZE };
    enum { NATIVE_SIZE = sizeof(char)  };

};


template <>
  class TypeTraits<short>
{
  public:

    enum { XDR_SIZE    = XDR_SHORT_SIZE };
    enum { NATIVE_SIZE = sizeof(short)  };

};


template <>
  class TypeTraits<int>
{
  public:

    enum { XDR_SIZE    = XDR_INT_SIZE };
    enum { NATIVE_SIZE = sizeof(int)  };

};


template <>
  class TypeTraits<long>
{
  public:

    enum { XDR_SIZE    = XDR_LONG_SIZE };
    enum { NATIVE_SIZE = sizeof(long)  };

};


template <>
  class TypeTraits<lint>
{
  public:

    enum { XDR_SIZE    = XDR_LONG_SIZE };
    enum { NATIVE_SIZE = sizeof(lint)  };

};


template <>
  class TypeTraits<float>
{
  public:

    enum { XDR_SIZE    = XDR_FLOAT_SIZE };
    enum { NATIVE_SIZE = sizeof(float)  };

};


template <>
  class TypeTraits<double>
{
  public:

    enum { XDR_SIZE    = XDR_DOUBLE_SIZE };
    enum { NATIVE_SIZE = sizeof(double)  };

};





//#######################################################################
//   Implementation
//#######################################################################


JEM_END_NAMESPACE ( xdr )
JEM_END_PACKAGE   ( io )

#include <jem/io/xdr.tcc>

#endif
