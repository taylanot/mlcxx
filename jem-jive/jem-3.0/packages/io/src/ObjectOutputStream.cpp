
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


#include <jem/base/rtti.h>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/utilities.h>
#include <jem/base/Error.h>
#include <jem/base/MemCache.h>
#include <jem/base/TypeTraits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/WeakHashMap.h>
#include <jem/util/HashDictionary.h>
#include <jem/io/utilities.h>
#include <jem/io/StreamCodec.h>
#include <jem/io/Serializable.h>
#include <jem/io/ObjectOutputStream.h>
#include <jem/io/IOverflowException.h>
#include <jem/io/SerializationException.h>


JEM_DEFINE_CLASS( jem::io::ObjectOutputStream );


JEM_BEGIN_PACKAGE( io )


using jem::util::ObjectMap;
using jem::util::Dictionary;

typedef Ref<Object>  ObjRef;


//=======================================================================
//   class ObjectOutputStream::Data_
//=======================================================================


class ObjectOutputStream::Data_ : public Collectable
{
 public:

  explicit                  Data_

    ( Ref<OutputStream>       out );

  inline void               close         ();
  inline void               flush         ();
  inline void               reset         ();

  inline void               write

    ( const byte*             src,
      idx_t                   len,
      ThrowMode               mode = CAN_THROW );

  template <class T>
    inline void             encode

    ( T                       val );

  void                      encode

    ( idx_t                   val );

  void                      encode

    ( const String&           str );

  void                      startChunk

    ( byte                    tag,
      ThrowMode               mode = CAN_THROW );

  void                      closeChunk    ();

  bool                      addName

    ( idx_t&                  id,
      const String&           name );

  inline void               delName

    ( const String&           name );

  bool                      addObject

    ( idx_t&                  id,
      const ObjRef&           obj );

  inline void               delObject

    ( const ObjRef&           obj );

  void                      flushBuffer

    ( ThrowMode               mode = CAN_THROW );

  void                      reserveSpace

    ( int                     size );


 public:

  const int                 SEX;

  Ref<OutputStream>         output;
  Options                   options;


 protected:

  virtual                  ~Data_         ();


 private:

  void                      chunkError_   () const;


 private:

  byte                      tag_;
  int                       first_;
  int                       last_;
  byte*                     buffer_;

  Ref<Dictionary>           names_;
  Ref<ObjectMap>            objects_;

  idx_t                     nextObj_;
  idx_t                     nextName_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ObjectOutputStream::Data_::Data_ ( Ref<OutputStream> out ) :

  SEX    ( xdr::getByteOrder() ),
  output ( out )

{
  options   = DEFAULT_OPTIONS;
  tag_      = 0;
  first_    = 0;
  last_     = 0;
  buffer_   = (byte*) MemCache::alloc ( (size_t) MAX_CHUNK_SIZE + 8 );
  nextObj_  = 0;
  nextName_ = 0;
}


ObjectOutputStream::Data_::~Data_ ()
{
  if ( tag_ )
  {
    closeChunk ();
  }

  if ( first_ )
  {
    flushBuffer ( NO_THROW );
  }

  MemCache::dealloc ( buffer_, (size_t) MAX_CHUNK_SIZE + 8 );

  buffer_ = nullptr;
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


inline void ObjectOutputStream::Data_::close ()
{
  if ( tag_ )
  {
    closeChunk ();
  }

  if ( first_ )
  {
    flushBuffer ();
  }

  output->close ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


inline void ObjectOutputStream::Data_::flush ()
{
  if ( first_ )
  {
    flushBuffer ();
  }

  output->flush ();
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


inline void ObjectOutputStream::Data_::reset ()
{
  startChunk ( RESET_TAG );
  closeChunk ();

  objects_->clear ();

  nextObj_ = 0;
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


inline void ObjectOutputStream::Data_::write

  ( const byte*  src,
    idx_t        len,
    ThrowMode    mode )

{
  while ( len > 0 )
  {
    if ( last_ == MAX_CHUNK_SIZE )
    {
      if ( last_ - first_ > 1024 && tag_ )
      {
        closeChunk ();
      }
      else
      {
        flushBuffer ( mode );
      }
    }

    if ( tag_ != DATA_TAG )
    {
      startChunk ( DATA_TAG, mode );
    }

    int  k = (int) jem::min ( len, (idx_t) (MAX_CHUNK_SIZE - last_) );

    std::memcpy ( buffer_ + last_, src, (size_t) k );

    last_ += k;
    src   += k;
    len   -= k;
  }
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class T>

  inline void ObjectOutputStream::Data_::encode ( T val )

{
  const int  XDR_SIZE = xdr::TypeTraits<T>::XDR_SIZE;

  reserveSpace ( XDR_SIZE );
  xdr::encode  ( buffer_ + last_, val, SEX );

  last_ += XDR_SIZE;
}


void ObjectOutputStream::Data_::encode ( idx_t val )
{
  int  bytec = xdr::TypeTraits<JEM_IDX_T>::XDR_SIZE;

  if      ( options & IDX_SIZE_4BYTES )
  {
    bytec = 4;
  }
  else if ( options & IDX_SIZE_8BYTES )
  {
    bytec = 8;
  }

  if ( bytec == xdr::TypeTraits<JEM_IDX_T>::XDR_SIZE )
  {
    encode ( val.value );
  }
  else
  {
    if ( bytec == 4 )
    {
      if ( (val < (idx_t) minOf<int>()) ||
           (val > (idx_t) maxOf<int>()) )
      {
        throw IOverflowException (
          JEM_FUNC,
          "value of type jem::idx_t does not fit into four bytes"
        );
      }

      encode ( (int) val );
    }
    else
    {
      encode ( (lint) val );
    }
  }
}


void ObjectOutputStream::Data_::encode ( const String& str )
{
  const int  len = (int) str.size ();

  if ( str.size() > maxOf( len ) )
  {
    throw SerializationException (
      JEM_FUNC,
      "string too long"
    );
  }

  encode       ( len );
  reserveSpace ( len );
  std::memcpy  ( buffer_ + last_, str.addr(), (size_t) len );

  last_ += len;
}


//-----------------------------------------------------------------------
//   startChunk
//-----------------------------------------------------------------------


void ObjectOutputStream::Data_::startChunk

  ( byte       tag,
    ThrowMode  mode )

{
  if ( tag_ )
  {
    closeChunk ();
  }

  if ( last_ + 5 > MAX_CHUNK_SIZE )
  {
    flushBuffer ( mode );
  }

  tag_               = tag;
  buffer_[last_ + 4] = tag;
  last_             += 5;
}


//-----------------------------------------------------------------------
//   closeChunk
//-----------------------------------------------------------------------


void ObjectOutputStream::Data_::closeChunk ()
{
  JEM_ASSERT ( tag_ && last_ <= MAX_CHUNK_SIZE );

  ulint  len = (ulint) (last_ - first_ - 4);
  ulint  crc;


  encodeUL32 ( buffer_ + first_, len );

  first_ += 4;

  initCRC    ( crc );
  updateCRC  ( crc, buffer_ + first_, last_ - first_ );
  finishCRC  ( crc );
  encodeUL32 ( buffer_ + last_, crc );

  last_ += 4;
  first_ = last_;
  tag_   = 0;
}


//-----------------------------------------------------------------------
//   addName
//-----------------------------------------------------------------------


bool ObjectOutputStream::Data_::addName

  ( idx_t&         id,
    const String&  name )

{
  using jem::util::HashDict;

  ObjRef  itmp;

  if ( ! names_ )
  {
    names_ = newInstance<HashDict> ();
  }

  if ( names_->find( itmp, name ) )
  {
    id = toValue<idx_t> ( itmp );

    return false;
  }
  else
  {
    if ( nextName_ == maxOf( nextName_ ) )
    {
      throw SerializationException (
        JEM_FUNC,
        "name index overflow"
      );
    }

    id = nextName_++;

    names_->insert ( name, toObject( id ) );

    return true;
  }
}


//-----------------------------------------------------------------------
//   delName
//-----------------------------------------------------------------------


inline void ObjectOutputStream::Data_::delName

  ( const String&  name )

{
  if ( names_ )
  {
    names_->erase ( name );
  }
}


//-----------------------------------------------------------------------
//   addObject
//-----------------------------------------------------------------------


bool ObjectOutputStream::Data_::addObject

  ( idx_t&         id,
    const ObjRef&  obj )

{
  using jem::util::WeakHashMap;

  ObjRef  itmp;

  if ( ! objects_ )
  {
    objects_ = newInstance<WeakHashMap> ();
  }

  if ( objects_->find( itmp, obj ) )
  {
    id = toValue<idx_t> ( itmp );

    return false;
  }
  else
  {
    if ( nextObj_ == maxOf( nextObj_ ) )
    {
      throw SerializationException (
        JEM_FUNC,
        "object index overflow"
      );
    }

    id = nextObj_++;

    objects_->insert ( obj, toObject( id ) );

    return true;
  }
}


//-----------------------------------------------------------------------
//   delObject
//-----------------------------------------------------------------------


inline void ObjectOutputStream::Data_::delObject

  ( const ObjRef&  obj )

{
  if ( objects_ )
  {
    objects_->erase ( obj );
  }
}

//-----------------------------------------------------------------------
//   flushBuffer
//-----------------------------------------------------------------------


void ObjectOutputStream::Data_::flushBuffer ( ThrowMode mode )
{
  const int  i = first_;
  const int  j = last_;


  last_ -= first_;
  first_ = 0;

  if ( mode == CAN_THROW )
  {
    output->write        ( buffer_, i );
  }
  else
  {
    output->writeNoThrow ( buffer_, i );
  }

  if ( j > i )
  {
    ::memmove ( buffer_, buffer_ + i, (size_t) (j - i) );
  }
}


//-----------------------------------------------------------------------
//   reserveSpace
//-----------------------------------------------------------------------


void ObjectOutputStream::Data_::reserveSpace ( int size )
{
  if ( last_ + size > MAX_CHUNK_SIZE )
  {
    flushBuffer ();

    if ( last_ + size > MAX_CHUNK_SIZE )
    {
      chunkError_ ();
    }
  }
}


//-----------------------------------------------------------------------
//   chunkError_
//-----------------------------------------------------------------------


void ObjectOutputStream::Data_::chunkError_ () const
{
  throw Error (
    CLASS_NAME,
    String::format (
      "oops, %s chunk is too large", getChunkName ( tag_ )
    )
  );
}


//=======================================================================
//   class ObjectOutputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ObjectOutputStream::ObjectOutputStream ( Ref<OutputStream> out )
{
  JEM_PRECHECK ( out );

  data_ = newInstance<Data_> ( out );

  StreamCodec::encode ( *out, MAGIC, data_->SEX );

  data_->startChunk   ( HEADER_TAG );
  data_->encode       ( VERSION );
  data_->closeChunk   ();
}


ObjectOutputStream::~ObjectOutputStream ()
{}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void ObjectOutputStream::close ()
{
  data_->close ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void ObjectOutputStream::flush ()
{
  data_->flush ();
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void ObjectOutputStream::write ( const void* buf, idx_t n )
{
  data_->write ( (const byte*) buf, n );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void ObjectOutputStream::writeNoThrow

  ( const void*  buf,
    idx_t        n ) noexcept

{
  data_->write ( (const byte*) buf, n, NO_THROW );
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void ObjectOutputStream::reset ()
{
  data_->reset ();
}


//-----------------------------------------------------------------------
//   encode ( bool )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( bool b )
{
  StreamCodec::encode ( *this, b, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( byte )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( byte b )
{
  write ( & b, 1 );
}


//-----------------------------------------------------------------------
//   encode ( char )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( char c )
{
  StreamCodec::encode ( *this, c, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( short )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( short s )
{
  StreamCodec::encode ( *this, s, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( int )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( int i )
{
  StreamCodec::encode ( *this, i, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( long )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( long l )
{
  StreamCodec::encode ( *this, l, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( lint )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( lint l )
{
  StreamCodec::encode ( *this, l, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( idx_t )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( idx_t i )
{
  StreamCodec::encode ( *this, i, data_->SEX, data_->options );
}


//-----------------------------------------------------------------------
//   encode ( float )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( float f )
{
  StreamCodec::encode ( *this, f, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( double )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( double d )
{
  StreamCodec::encode ( *this, d, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( const bool* )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( const bool* buf, idx_t n )
{
  StreamCodec::encode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( const byte* )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( const byte* buf, idx_t n )
{
  write ( (void*) buf, n );
}


//-----------------------------------------------------------------------
//   encode ( const char* )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( const char* buf, idx_t n )
{
  StreamCodec::encode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( const short* )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( const short* buf, idx_t n )
{
  StreamCodec::encode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( const int* )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( const int* buf, idx_t n )
{
  StreamCodec::encode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( const long* )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( const long* buf, idx_t n )
{
  StreamCodec::encode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( const lint* )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( const lint* buf, idx_t n )
{
  StreamCodec::encode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( const idx_t* )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( const idx_t* buf, idx_t n )
{
  StreamCodec::encode ( *this, buf, n, data_->SEX, data_->options );
}


//-----------------------------------------------------------------------
//   encode ( const float* )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( const float* buf, idx_t n )
{
  StreamCodec::encode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   encode ( const double* )
//-----------------------------------------------------------------------


void ObjectOutputStream::encode ( const double* buf, idx_t n )
{
  StreamCodec::encode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   encodeObject
//-----------------------------------------------------------------------


void ObjectOutputStream::encodeObject ( const ObjRef& obj )
{
  Data_&  d = * data_;

  switch ( obj.getType() )
  {
  case ObjRef::NIL_TYPE:

    d.startChunk   ( NIL_TAG );
    d.closeChunk   ();

    return;

  case ObjRef::BOOL_TYPE:

    d.startChunk ( BOOL_TAG );
    d.encode     ( obj.getBool() );
    d.closeChunk ();

    return;

  case ObjRef::CHAR_TYPE:

    d.startChunk ( CHAR_TAG );
    d.encode     ( obj.getChar() );
    d.closeChunk ();

    return;

  case ObjRef::STRING_TYPE:

    d.startChunk ( STRING_TAG );
    d.encode     ( obj.getString() );
    d.closeChunk ();

    return;

  case ObjRef::INT_TYPE:

    d.startChunk ( INT_TAG );
    d.encode     ( obj.getInt() );
    d.closeChunk ();

    return;

  case ObjRef::FLOAT_TYPE:

    d.startChunk ( FLOAT_TAG );
    d.encode     ( obj.getFloat() );
    d.closeChunk ();

    return;

  default:

    break;
  }

  idx_t  objID = 0;
  bool   isNew = d.addObject ( objID, obj );

  if ( ! isNew )
  {
    d.startChunk ( REF_TAG );
    d.encode     ( objID );
    d.closeChunk ();
  }
  else
  {
    Class*         type = obj ->getClass ();
    String         name = type->getName  ();
    Serializable*  s    = dynamicCast<Serializable*> ( obj );

    if ( ! s )
    {
      throw SerializationException (
        JEM_FUNC,
        String::format (
          "class `%s\' does not implement the Serializable interface",
          name
        )
      );
    }

    if ( ! type->equalType( obj ) )
    {
      throw SerializationException(
        JEM_FUNC,
        String::format (
          "object of type `%s\' does not match its dynamic type",
          name
        )
      );
    }

    idx_t  typeID = 0;

    isNew = d.addName ( typeID, name );

    if ( isNew )
    {
      if ( name.size() == 0 ||
           name.size() >  MAX_CHUNK_SIZE - 16 )
      {
        throw SerializationException (
          JEM_FUNC,
          String::format ( "invalid class name: `%s\'", name )
        );
      }

      try
      {
        d.startChunk ( CLASS_TAG );
        d.encode     ( typeID );
        d.encode     ( name );
        d.closeChunk ();
      }
      catch ( ... )
      {
        d.delName ( name );
        throw;
      }
    }

    try
    {
      d.startChunk ( OBJECT_TAG );
      d.encode     ( typeID );
      d.encode     ( objID );
      d.closeChunk ();

      s->writeTo   ( *this );
    }
    catch ( ... )
    {
      d.delObject ( obj );
      throw;
    }
  }
}


//-----------------------------------------------------------------------
//   encodeReference
//-----------------------------------------------------------------------


void ObjectOutputStream::encodeReference ( const ObjRef& obj )
{
  Data_&  d  = * data_;

  idx_t   id = 0;

  d.addObject  ( id,  obj );
  d.startChunk ( XREF_TAG );
  d.encode     ( id );
  d.closeChunk ();
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


ObjectOutputStream::Options ObjectOutputStream::getOptions () const
{
  return data_->options;
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void ObjectOutputStream::setOption

  ( Option  option,
    bool    yesno )

{
  data_->options.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void ObjectOutputStream::setOptions ( Options options )
{
  data_->options = options;
}


JEM_END_PACKAGE( io )
