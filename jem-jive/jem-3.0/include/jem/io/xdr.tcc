
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

#ifndef JEM_IO_XDR_TCC
#define JEM_IO_XDR_TCC


JEM_BEGIN_PACKAGE   ( io )
JEM_BEGIN_NAMESPACE ( xdr )


//=======================================================================
//   private constants
//=======================================================================


enum  { BYTE_BITS  = CHAR_BIT      };
enum  { SIGN_SHIFT = BYTE_BITS - 1 };

enum  { SMALLER    = 0 };
enum  { EQUAL      = 1 };
enum  { GREATER    = 2 };


//=======================================================================
//   testSignBit
//=======================================================================


inline bool testSignBit ( byte b ) noexcept
{
  return ((b >> SIGN_SHIFT) != 0);
}


//=======================================================================
//   class ByteArray
//=======================================================================


template <int N>
  class ByteArray {};


//-----------------------------------------------------------------------
//   class ByteArray<0>
//-----------------------------------------------------------------------


template <>

  class ByteArray<0>

{
 public:

  static inline void copy  ( byte*, const byte* ) noexcept
  {}

  static inline void rcopy ( byte*, const byte* ) noexcept
  {}

  static inline void set   ( byte*, byte )        noexcept
  {}

};


//-----------------------------------------------------------------------
//   class ByteArray<2>
//-----------------------------------------------------------------------


template <>

  class ByteArray<2>

{
 public:

  static inline void copy ( byte* dest, const byte* src ) noexcept
  {
    dest[0] = src[0];
    dest[1] = src[1];
  }

  static inline void rcopy ( byte* dest, const byte* src ) noexcept
  {
    dest[0] = src[1];
    dest[1] = src[0];
  }

  static inline void set ( byte* buf, byte b ) noexcept
  {
    buf[0] = b;
    buf[1] = b;
  }

};


//-----------------------------------------------------------------------
//   class ByteArray<4>
//-----------------------------------------------------------------------


template <>

  class ByteArray<4>

{
 public:

  static inline void copy ( byte* dest, const byte* src ) noexcept
  {
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
  }

  static inline void rcopy ( byte* dest, const byte* src ) noexcept
  {
    dest[0] = src[3];
    dest[1] = src[2];
    dest[2] = src[1];
    dest[3] = src[0];
  }

  static inline void set ( byte* buf, byte b ) noexcept
  {
    buf[0] = b;
    buf[1] = b;
    buf[2] = b;
    buf[3] = b;
  }

};


//-----------------------------------------------------------------------
//   class ByteArray<6>
//-----------------------------------------------------------------------


template <>

  class ByteArray<6>

{
 public:

  static inline void copy ( byte* dest, const byte* src ) noexcept
  {
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
    dest[4] = src[4];
    dest[5] = src[5];
  }

  static inline void rcopy ( byte* dest, const byte* src ) noexcept
  {
    dest[0] = src[5];
    dest[1] = src[4];
    dest[2] = src[3];
    dest[3] = src[2];
    dest[4] = src[1];
    dest[5] = src[0];
  }

  static inline void set ( byte* buf, byte b ) noexcept
  {
    buf[0] = b;
    buf[1] = b;
    buf[2] = b;
    buf[3] = b;
    buf[4] = b;
    buf[5] = b;
  }

};


//-----------------------------------------------------------------------
//   class ByteArray<8>
//-----------------------------------------------------------------------


template <>

  class ByteArray<8>

{
 public:

  static inline void copy ( byte* dest, const byte* src ) noexcept
  {
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
    dest[4] = src[4];
    dest[5] = src[5];
    dest[6] = src[6];
    dest[7] = src[7];
  }

  static inline void rcopy ( byte* dest, const byte* src ) noexcept
  {
    dest[0] = src[7];
    dest[1] = src[6];
    dest[2] = src[5];
    dest[3] = src[4];
    dest[4] = src[3];
    dest[5] = src[2];
    dest[6] = src[1];
    dest[7] = src[0];
  }

  static inline void set ( byte* buf, byte b ) noexcept
  {
    buf[0] = b;
    buf[1] = b;
    buf[2] = b;
    buf[3] = b;
    buf[4] = b;
    buf[5] = b;
    buf[6] = b;
    buf[7] = b;
  }

};


//=======================================================================
//   class CompareBase & specializations
//=======================================================================


template <bool Greater>

  class CompareBase {};


template <>

  class CompareBase<true>

{
 public:

  enum { VALUE = GREATER };

};


template <>

  class CompareBase<false>

{
 public:

  enum { VALUE = SMALLER };

};


//=======================================================================
//   class Compare & specializations
//=======================================================================


template <int M, int N>

  class Compare : public CompareBase<(M > N)>

{};


template <int N>

  class Compare<N,N>

{
 public:

  enum { VALUE = EQUAL };

};


//=======================================================================
//   class BigEndianIntCodec
//=======================================================================


template <class T, int Diff>

  class BigEndianIntCodec {};


//-----------------------------------------------------------------------
//   class BigEndianIntCodec<T,GREATER>
//-----------------------------------------------------------------------


template <class T>

  class BigEndianIntCodec<T,GREATER>

{
 public:

  typedef TypeTraits<T>   Traits;

  enum                  { DELTA = Traits::XDR_SIZE -
                                  Traits::NATIVE_SIZE };


  static inline void encode ( byte* buf, const T& value ) noexcept
  {
    if ( value < 0 )
    {
      ByteArray<DELTA>::set ( buf, 255 );
    }
    else
    {
      ByteArray<DELTA>::set ( buf, 0x0 );
    }

    ByteArray<Traits::NATIVE_SIZE>::copy (
      buf + DELTA, (const byte*) &value
    );
  }

  static inline void decode ( T& value, const byte* buf ) noexcept
  {
    ByteArray<Traits::NATIVE_SIZE>::copy (
      (byte*) &value, buf + DELTA
    );
  }

};


//-----------------------------------------------------------------------
//   class BigEndianIntCodec<T,EQUAL>
//-----------------------------------------------------------------------


template <class T>

  class BigEndianIntCodec<T,EQUAL>

{
 public:

  typedef TypeTraits<T>   Traits;


  static inline void encode ( byte* buf, const T& value ) noexcept
  {
    ByteArray<Traits::XDR_SIZE>::copy ( buf, (const byte*) &value );
  }

  static inline void decode ( T& value, const byte* buf ) noexcept
  {
    ByteArray<Traits::XDR_SIZE>::copy ( (byte*) &value, buf );
  }

};


//-----------------------------------------------------------------------
//   class BigEndianIntCodec<T,SMALLER>
//-----------------------------------------------------------------------


template <class T>

  class BigEndianIntCodec<T,SMALLER>

{
 public:

  typedef TypeTraits<T>   Traits;

  enum                  { DELTA = Traits::NATIVE_SIZE -
                                  Traits::XDR_SIZE };


  static inline void encode ( byte* buf, const T& value ) noexcept
  {
    ByteArray<Traits::XDR_SIZE>::copy (
      buf, ((const byte*) &value) + DELTA
    );
  }

  static inline void decode ( T& value, const byte* buf ) noexcept
  {
    if ( testSignBit( buf[0] ) )
    {
      ByteArray<DELTA>::set ( (byte*) &value, 255 );
    }
    else
    {
      ByteArray<DELTA>::set ( (byte*) &value, 0x0 );
    }

    ByteArray<Traits::XDR_SIZE>::copy (
      ((byte*) &value) + DELTA, buf
    );
  }

};


//=======================================================================
//   class LittleEndianIntCodec
//=======================================================================


template <class T, int Diff>

  class LittleEndianIntCodec {};


//-----------------------------------------------------------------------
//   class LittleEndianIntCodec<T,GREATER>
//-----------------------------------------------------------------------


template <class T>

  class LittleEndianIntCodec<T,GREATER>

{
 public:

  typedef TypeTraits<T>   Traits;

  enum                  { DELTA = Traits::XDR_SIZE -
                                  Traits::NATIVE_SIZE };


  static inline void encode ( byte* buf, const T& value ) noexcept
  {
    if ( value < 0 )
    {
      ByteArray<DELTA>::set ( buf, 255 );
    }
    else
    {
      ByteArray<DELTA>::set ( buf, 0x0 );
    }

    ByteArray<Traits::NATIVE_SIZE>::rcopy (
      buf + DELTA, (const byte*) &value
    );
  }

  static inline void decode ( T& value, const byte* buf ) noexcept
  {
    ByteArray<Traits::NATIVE_SIZE>::rcopy (
      (byte*) &value, buf + DELTA
    );
  }

};


//-----------------------------------------------------------------------
//   class LittleEndianIntCodec<T,EQUAL>
//-----------------------------------------------------------------------


template <class T>

  class LittleEndianIntCodec<T,EQUAL>

{
 public:

  typedef TypeTraits<T>   Traits;


  static inline void encode ( byte* buf, const T& value ) noexcept
  {
    ByteArray<Traits::XDR_SIZE>::rcopy ( buf, (const byte*) &value );
  }


  static inline void decode ( T& value, const byte* buf ) noexcept
  {
    ByteArray<Traits::XDR_SIZE>::rcopy ( (byte*) &value, buf );
  }

};


//-----------------------------------------------------------------------
//   class LittleEndianIntCodec<T,SMALLER>
//-----------------------------------------------------------------------


template <class T>

  class LittleEndianIntCodec<T,SMALLER>

{
 public:

  typedef TypeTraits<T>   Traits;

  enum                  { DELTA = Traits::NATIVE_SIZE -
                                  Traits::XDR_SIZE };


  static inline void encode ( byte* buf, const T& value ) noexcept
  {
    ByteArray<Traits::XDR_SIZE>::rcopy ( buf, (const byte*) &value );
  }

  static inline void decode ( T& value, const byte* buf ) noexcept
  {
    if ( testSignBit( buf[0] ) )
    {
      ByteArray<DELTA>::set (
        ((byte*) &value) + Traits::XDR_SIZE, 255
      );
    }
    else
    {
      ByteArray<DELTA>::set (
        ((byte*) &value) + Traits::XDR_SIZE, 0x0
      );
    }

    ByteArray<Traits::XDR_SIZE>::rcopy ( (byte*) &value, buf );
  }

};


//=======================================================================
//   class BigEndianFloatCodec
//=======================================================================


template <class T>

  class BigEndianFloatCodec

{
  public:

  typedef TypeTraits<T>   Traits;


  static inline void encode ( byte* buf, const T& value ) noexcept
  {
    ByteArray<Traits::XDR_SIZE>::copy ( buf, (const byte*) &value );
  }

  static inline void decode ( T& value, const byte* buf ) noexcept
  {
    ByteArray<Traits::XDR_SIZE>::copy ( (byte*) &value, buf );
  }

};


//=======================================================================
//   class LittleEndianFloatCodec
//=======================================================================


template <class T>

  class LittleEndianFloatCodec : public TypeTraits<T>

{
 public:

  typedef TypeTraits<T>   Traits;


  static inline void encode ( byte* buf, const T& value ) noexcept
  {
    ByteArray<Traits::XDR_SIZE>::rcopy ( buf, (const byte*) &value );
  }

  static inline void decode ( T& value, const byte* buf ) noexcept
  {
    ByteArray<Traits::XDR_SIZE>::rcopy ( (byte*) &value, buf );
  }

};


//=======================================================================
//   class BigEndianCodec
//=======================================================================


template <class T>

  class BigEndianCodec {};


//-----------------------------------------------------------------------
//   class BigEndianCodec<bool>
//-----------------------------------------------------------------------


template <>

  class BigEndianCodec<bool>

{
 public:

  static inline void encode ( byte* buf, const bool& value ) noexcept
  {
    if ( value )
    {
      *buf = (byte) 255;
    }
    else
    {
      *buf = (byte) 0x0;
    }
  }

  static inline void decode ( bool& value, const byte* buf ) noexcept
  {
    if ( *buf )
    {
      value = true;
    }
    else
    {
      value = false;
    }
  }

};


//-----------------------------------------------------------------------
//   class BigEndianCodec<byte>
//-----------------------------------------------------------------------


template <>

  class BigEndianCodec<byte>

{
 public:

  static inline void encode ( byte* buf, const byte& value ) noexcept
  {
    *buf = value;
  }

  static inline void decode ( byte& value, const byte* buf ) noexcept
  {
    value = *buf;
  }

};


//-----------------------------------------------------------------------
//   BigEndianCodec<char>
//-----------------------------------------------------------------------


template <>

  class BigEndianCodec<char>

{
 public:

  static inline void encode ( byte* buf, const char& value ) noexcept
  {
    *buf = (byte) value;
  }

  static inline void decode ( char& value, const byte* buf ) noexcept
  {
    value = (char) *buf;
  }

};


//-----------------------------------------------------------------------
//   BigEndianCodec<short>
//-----------------------------------------------------------------------


template <>

  class BigEndianCodec<short> : public BigEndianIntCodec
    < short,
      Compare< XDR_SHORT_SIZE, sizeof(short) >::VALUE >

{};


//-----------------------------------------------------------------------
//   BigEndianCodec<int>
//-----------------------------------------------------------------------


template <>

  class BigEndianCodec<int> : public BigEndianIntCodec
    < int,
      Compare< XDR_INT_SIZE, sizeof(int) >::VALUE >

{};


//-----------------------------------------------------------------------
//   BigEndianCodec<long>
//-----------------------------------------------------------------------

template <>

  class BigEndianCodec<long> : public BigEndianIntCodec
    < long,
      Compare< XDR_LONG_SIZE, sizeof(long) >::VALUE >

{};


//-----------------------------------------------------------------------
//   BigEndianCodec<lint>
//-----------------------------------------------------------------------


template <>

  class BigEndianCodec<lint> : public BigEndianIntCodec
    < lint,
      Compare< XDR_LONG_SIZE, sizeof(lint) >::VALUE >

{};


//-----------------------------------------------------------------------
//   BigEndianCodec<float>
//-----------------------------------------------------------------------


template <>

  class BigEndianCodec<float> : public BigEndianFloatCodec<float>

{};


//-----------------------------------------------------------------------
//   BigEndianCodec<double>
//-----------------------------------------------------------------------


template <>

  class BigEndianCodec<double> : public BigEndianFloatCodec<double>

{};


//=======================================================================
//   class LittleEndianCodec
//=======================================================================


template <class T>

  class LittleEndianCodec {};


//-----------------------------------------------------------------------
//   class LittleEndianCodec<bool>
//-----------------------------------------------------------------------


template <>

  class LittleEndianCodec<bool> : public BigEndianCodec<bool>

{};


//-----------------------------------------------------------------------
//   class LittleEndianCodec<byte>
//-----------------------------------------------------------------------


template <>

  class LittleEndianCodec<byte> : public BigEndianCodec<byte>

{};


//-----------------------------------------------------------------------
//   LittleEndianCodec<char>
//-----------------------------------------------------------------------


template <>

  class LittleEndianCodec<char> : public BigEndianCodec<char>

{};


//-----------------------------------------------------------------------
//   LittleEndianCodec<short>
//-----------------------------------------------------------------------


template <>

  class LittleEndianCodec<short> : public LittleEndianIntCodec
    < short,
      Compare< XDR_SHORT_SIZE, sizeof(short) >::VALUE >

{};


//-----------------------------------------------------------------------
//   LittleEndianCodec<int>
//-----------------------------------------------------------------------


template <>

  class LittleEndianCodec<int> : public LittleEndianIntCodec
    < int,
      Compare< XDR_INT_SIZE, sizeof(int) >::VALUE >

{};


//-----------------------------------------------------------------------
//   LittleEndianCodec<long>
//-----------------------------------------------------------------------

template <>

  class LittleEndianCodec<long> : public LittleEndianIntCodec
    < long,
      Compare< XDR_LONG_SIZE, sizeof(long) >::VALUE >

{};


//-----------------------------------------------------------------------
//   LittleEndianCodec<lint>
//-----------------------------------------------------------------------


template <>

  class LittleEndianCodec<lint> : public LittleEndianIntCodec
    < lint,
      Compare< XDR_LONG_SIZE, sizeof(lint) >::VALUE >

{};


//-----------------------------------------------------------------------
//   LittleEndianCodec<float>
//-----------------------------------------------------------------------


template <>

  class LittleEndianCodec<float> :
    public LittleEndianFloatCodec<float>

{};


//-----------------------------------------------------------------------
//   LittleEndianCodec<double>
//-----------------------------------------------------------------------


template <>

  class LittleEndianCodec<double> :
    public LittleEndianFloatCodec<double>

{};


//=======================================================================
//   encode
//=======================================================================


template <class T>

  inline void encode ( byte* buf, const T& value, int sex ) noexcept

{
  if ( sex == BIG_ENDIAN )
  {
    BigEndianCodec<T>::encode ( buf, value );
  }
  else
  {
    LittleEndianCodec<T>::encode ( buf, value );
  }
}


template <class T>

  void encode ( byte* buf, const T* values,
                idx_t n, int sex ) noexcept

{
  if ( sex == BIG_ENDIAN )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      BigEndianCodec<T>::encode ( buf, values[i] );
      buf += TypeTraits<T>::XDR_SIZE;
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      LittleEndianCodec<T>::encode ( buf, values[i] );
      buf += TypeTraits<T>::XDR_SIZE;
    }
  }
}


//=======================================================================
//   decode
//=======================================================================


template <class T>

  inline void decode ( T& value, const byte* buf, int sex ) noexcept

{
  if ( sex == BIG_ENDIAN )
  {
    BigEndianCodec<T>::decode ( value, buf );
  }
  else
  {
    LittleEndianCodec<T>::decode ( value, buf );
  }
}


template <class T>

  void decode ( T* values, idx_t n,
                const byte* buf, int sex ) noexcept

{
  if ( sex == BIG_ENDIAN )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      BigEndianCodec<T>::decode ( values[i], buf );
      buf += TypeTraits<T>::XDR_SIZE;
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      LittleEndianCodec<T>::decode ( values[i], buf );
      buf += TypeTraits<T>::XDR_SIZE;
    }
  }
}


JEM_END_NAMESPACE ( xdr )
JEM_END_PACKAGE   ( io )

#endif
