
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


#include <jem/base/assert.h>
#include <jem/base/utilities.h>
#include <jem/base/Error.h>
#include <jem/base/MemCache.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/InstantiationException.h>
#include <jem/util/Flex.h>
#include <jem/util/SparseArray.h>
#include <jem/io/utilities.h>
#include <jem/io/TextOutput.h>
#include <jem/io/StreamCodec.h>
#include <jem/io/Serializable.h>
#include <jem/io/SerializationException.h>
#include <jem/io/ObjectInputStream.h>


JEM_DEFINE_CLASS( jem::io::ObjectInputStream );


JEM_BEGIN_PACKAGE( io )


using jem::util::Flex;
using jem::util::SparseArray;

typedef Ref<Object>  ObjRef;


//=======================================================================
//   ObjectInputStream::Data_
//=======================================================================


class ObjectInputStream::Data_ : public Collectable
{
 public:

  explicit                  Data_

    ( Ref<InputStream>        in );

  inline idx_t              poll

    ( const Time&             timeout );

  inline void               reset         ();

  inline idx_t              read

    ( void*                   buf,
      idx_t                   len );

  template <class T>
    inline T                decode        ();

  template <class T>
    void                    decode

    ( T&                      val );

  void                      decode

    ( idx_t&                  val );

  void                      decode

    ( String&                 str );

  byte                      readChunk     ();

  void                      addClass

    ( idx_t                   id,
      Class*                  type );

  Class*                    getClass

    ( idx_t                   id )           const;

  void                      addObject

    ( idx_t                   id,
      const ObjRef&           obj );

  ObjRef                    getObject

    ( idx_t                   id )           const;

  inline void               delObject

    ( idx_t                   id );

  inline void               printTrace

    ( TextOutput&             out )          const;


 public:

  const int                 SEX;

  Ref<InputStream>          input;
  Flex<idx_t>               idStack;
  Options                   options;


 protected:

  virtual                  ~Data_         ();


 private:

  void                      chunkError_   () const;

  void                      fillBuffer_

    ( int                     count );


 private:

  byte                      tag_;
  int                       first_;
  int                       end_;
  int                       last_;
  byte*                     buffer_;

  SparseArray<ObjRef>       objects_;
  SparseArray<Class*>       classes_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ObjectInputStream::Data_::Data_ ( Ref<InputStream> in ) :

  SEX   ( xdr::getByteOrder() ),
  input ( in )

{
  options =  DEFAULT_OPTIONS;
  tag_    =  0;
  first_  =  0;
  end_    =  0;
  last_   =  0;
  buffer_ = (byte*) MemCache::alloc ( (size_t) MAX_CHUNK_SIZE + 8 );
}


ObjectInputStream::Data_::~Data_ ()
{
  MemCache::dealloc ( buffer_, (size_t) MAX_CHUNK_SIZE + 8 );

  buffer_ = nullptr;
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


inline idx_t ObjectInputStream::Data_::poll ( const Time& timeout )
{
  int  n = end_ - first_;

  if ( n > 0 )
  {
    return (idx_t) n;
  }
  else
  {
    return input->poll ( timeout );
  }
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


inline void ObjectInputStream::Data_::reset ()
{
  objects_.clear ();
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


inline idx_t ObjectInputStream::Data_::read

  ( void*  buf,
    idx_t  len )

{
  if ( len <= 0 )
  {
    return 0;
  }

  while ( tag_ != DATA_TAG || first_ == end_ )
  {
    readChunk ();

    if      ( tag_ == RESET_TAG )
    {
      reset ();
    }
    else if ( tag_ != DATA_TAG )
    {
      throw SerializationException (
        CLASS_NAME,
        String::format (
          "invalid input data: "
          "expected a data chunk but got a %s chunk",
          getChunkName ( tag_ )
        )
      );
    }
  }

  int  k = (int) jem::min ( len, (idx_t) (end_ - first_) );

  std::memcpy ( buf, buffer_ + first_, (size_t) k );

  first_ += k;

  return k;
}


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class T>

  inline T ObjectInputStream::Data_::decode ()

{
  T  tmp;

  decode ( tmp );

  return tmp;
}


template <class T>

  void ObjectInputStream::Data_::decode ( T& val )

{
  const int  XDR_SIZE = xdr::TypeTraits<T>::XDR_SIZE;

  if ( first_ + XDR_SIZE > end_ )
  {
    chunkError_ ();
  }

  xdr::decode ( val, buffer_ + first_, SEX );

  first_ += XDR_SIZE;
}


void ObjectInputStream::Data_::decode ( idx_t& val )
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

  if ( bytec == 4 )
  {
    val = (idx_t) decode<int>  ();
  }
  else
  {
    val = (idx_t) decode<lint> ();
  }
}


void ObjectInputStream::Data_::decode ( String& str )
{
  const int  len = decode<int> ();

  if ( first_ + len > end_ )
  {
    chunkError_ ();
  }

  str     = String ( (char*) (buffer_ + first_), (idx_t) len );
  first_ += len;
}


//-----------------------------------------------------------------------
//   readChunk
//-----------------------------------------------------------------------


byte ObjectInputStream::Data_::readChunk ()
{
  ulint  n, crc;
  int    k;

  if ( first_ < end_ )
  {
    throw SerializationException (
      CLASS_NAME,
      String::format (
        "new chunk requested within a %s chunk",
        getChunkName ( tag_ )
      )
    );
  }

  if ( tag_ )
  {
    // Skip the CRC of the current chunk

    if ( end_ + 4 > last_ )
    {
      throw Error ( JEM_FUNC, "oops, lost the checksum somewhere" );
    }

    end_  += 4;
    first_ = end_;
    tag_   = 0;
  }

  // Read the size of the next chunk

  if ( end_ + 4 > last_ )
  {
    fillBuffer_ ( end_ + 4 - last_ );
  }

  decodeUL32 ( n, buffer_ + end_ );

  if ( n < 1_ulint || n > (ulint) MAX_CHUNK_SIZE )
  {
    throw SerializationException (
      CLASS_NAME,
      String::format ( "invalid chunk size: %d", (lint) n )
    );
  }

  end_  += 4;
  first_ = end_;
  k      = end_ + (int) n;

  if ( k + 4 > last_ )
  {
    fillBuffer_ ( k + 4 - last_ );

    k = end_ + (int) n;
  }

  // Read and verify the checksum

  decodeUL32 ( n,   buffer_ + k );
  initCRC    ( crc );
  updateCRC  ( crc, buffer_ + first_, k - first_ );
  finishCRC  ( crc );

  if ( crc != n )
  {
    throw SerializationException (
      CLASS_NAME,
      "corrupt input data: checksum error"
    );
  }

  end_ = k;
  tag_ = buffer_[first_++];

  return tag_;
}


//-----------------------------------------------------------------------
//   addClass
//-----------------------------------------------------------------------


void ObjectInputStream::Data_::addClass

  ( idx_t   id,
    Class*  type )

{
  if ( classes_.find( id ) )
  {
    String  name1 = type        ->getName ();
    String  name2 = classes_[id]->getName ();

    throw SerializationException (
      CLASS_NAME,
      String::format (
        "invalid ID (%d) for class `%s\' "
        "(ID already used for class `%s\')",
        id,
        name1,
        name2
      )
    );
  }

  classes_[id] = type;
}


//-----------------------------------------------------------------------
//   getClass
//-----------------------------------------------------------------------


Class* ObjectInputStream::Data_::getClass ( idx_t id ) const
{
  Class*  type = classes_[id];

  if ( ! type )
  {
    throw SerializationException (
      CLASS_NAME,
      "invalid class ID: " + String ( id )
    );
  }

  return type;
}


//-----------------------------------------------------------------------
//   addObject
//-----------------------------------------------------------------------


void ObjectInputStream::Data_::addObject

  ( idx_t          id,
    const ObjRef&  obj )

{
  if ( objects_.find( id ) )
  {
    String  name1 = obj         ->toString ();
    String  name2 = objects_[id]->toString ();

    throw SerializationException (
      CLASS_NAME,
      String::format (
        "invalid ID (%d) for object `%s\' "
        "(ID already used for object `%s\')",
        id,
        name1,
        name2
      )
    );
  }

  objects_[id] = obj;
}


//-----------------------------------------------------------------------
//   getObject
//-----------------------------------------------------------------------


ObjRef ObjectInputStream::Data_::getObject ( idx_t id ) const
{
  const ObjRef*  obj = objects_.find ( id );

  if ( ! obj )
  {
    throw SerializationException (
      CLASS_NAME,
      "invalid object ID: " + String ( id )
    );
  }

  return *obj;
}


//-----------------------------------------------------------------------
//   delObject
//-----------------------------------------------------------------------


inline void ObjectInputStream::Data_::delObject ( idx_t id )
{
  objects_.erase ( id );
}


//-----------------------------------------------------------------------
//   printTrace
//-----------------------------------------------------------------------


inline void ObjectInputStream::Data_::printTrace

  ( TextOutput&  out ) const

{
  idx_t  n = idStack.size ();

  for ( idx_t i = 0; i < n; i += 2 )
  {
    idx_t   typeID = idStack [n - i - 1];
    idx_t   objID  = idStack [n - i - 2];
    Class*  type   = classes_[typeID];
    int     indent = 2;

    if      ( i < 10 )
    {
      indent += 2;
    }
    else if ( i < 100 )
    {
      indent += 1;
    }

    out.write ( ' ', indent );

    print ( out, i, " : " );

    if ( type )
    {
      print ( out, type->getName() );
    }
    else
    {
      print ( out, "<unknown type>" );
    }

    print ( out, " [", objID, "]\n" );
  }
}


//-----------------------------------------------------------------------
//   chunkError_
//-----------------------------------------------------------------------


void ObjectInputStream::Data_::chunkError_ () const
{
  throw SerializationException (
    CLASS_NAME,
    String::format (
      "unexpected end of %s chunk", getChunkName ( tag_ )
    )
  );
}


//-----------------------------------------------------------------------
//   fillBuffer_
//-----------------------------------------------------------------------


void ObjectInputStream::Data_::fillBuffer_ ( int n )
{
  int  k;

  if ( last_ + n > MAX_CHUNK_SIZE )
  {
    if ( last_ > first_ )
    {
      std::memmove ( buffer_, buffer_ + first_,
                     (size_t) (last_ - first_) );
    }

    last_ -= first_;
    end_  -= first_;
    first_ = 0;
  }

  while ( n > 0 )
  {
    k = (int) input->read ( buffer_ + last_, MAX_CHUNK_SIZE - last_ );

    if ( k == 0 )
    {
      throw SerializationException (
        CLASS_NAME,
        "unexpected end of input"
      );
    }

    last_ += k;
    n     -= k;
  }
}


//=======================================================================
//   class ObjectInputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ObjectInputStream::ObjectInputStream ( Ref<InputStream> in )
{
  JEM_PRECHECK ( in );

  int  magic;
  int  version;


  data_ = newInstance<Data_> ( in );

  StreamCodec::decode ( *in, magic, data_->SEX );

  if ( magic != MAGIC )
  {
    throw SerializationException (
      CLASS_NAME,
      "invalid input data: invalid magic number"
    );
  }

  if ( data_->readChunk() != HEADER_TAG )
  {
    throw SerializationException (
      CLASS_NAME,
      "invalid input data: missing header chunk"
    );
  }

  version = data_->decode<int> ();

  if ( version > VERSION )
  {
    throw SerializationException (
      CLASS_NAME,
      "object stream version mismatch (upgrade the jem library?)"
    );
  }
}


ObjectInputStream::~ObjectInputStream ()
{}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void ObjectInputStream::close ()
{
  data_->input->close ();
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t ObjectInputStream::poll ( const Time& timeout )
{
  return data_->poll ( timeout );
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t ObjectInputStream::read ( void* buf, idx_t n )
{
  return data_->read ( buf, n );
}


//-----------------------------------------------------------------------
//   decodeBool
//-----------------------------------------------------------------------


bool ObjectInputStream::decodeBool ()
{
  bool  b;

  StreamCodec::decode ( *this, b, data_->SEX );

  return b;
}


//-----------------------------------------------------------------------
//   decodeByte
//-----------------------------------------------------------------------


byte ObjectInputStream::decodeByte ()
{
  byte  b;

  if ( read( &b, 1 ) != 1 )
  {
    throw SerializationException (
      JEM_FUNC,
      "unexpected end of input"
    );
  }

  return b;
}


//-----------------------------------------------------------------------
//   decodeChar
//-----------------------------------------------------------------------


char ObjectInputStream::decodeChar ()
{
  char  c;

  StreamCodec::decode ( *this, c, data_->SEX );

  return c;
}


//-----------------------------------------------------------------------
//   decodeShort
//-----------------------------------------------------------------------


short ObjectInputStream::decodeShort ()
{
  short  s;

  StreamCodec::decode ( *this, s, data_->SEX );

  return s;
}


//-----------------------------------------------------------------------
//   decodeInt
//-----------------------------------------------------------------------


int ObjectInputStream::decodeInt ()
{
  int  i;

  StreamCodec::decode ( *this, i, data_->SEX );

  return i;
}


//-----------------------------------------------------------------------
//   decodeLong
//-----------------------------------------------------------------------


long ObjectInputStream::decodeLong ()
{
  long  l;

  StreamCodec::decode ( *this, l, data_->SEX );

  return l;
}


//-----------------------------------------------------------------------
//   decodeLint
//-----------------------------------------------------------------------


lint ObjectInputStream::decodeLint ()
{
  lint  l;

  StreamCodec::decode ( *this, l, data_->SEX );

  return l;
}


//-----------------------------------------------------------------------
//   decodeIdx
//-----------------------------------------------------------------------


idx_t ObjectInputStream::decodeIdx ()
{
  idx_t  i;

  StreamCodec::decode ( *this, i, data_->SEX, data_->options );

  return i;
}


//-----------------------------------------------------------------------
//   decodeFloat
//-----------------------------------------------------------------------


float ObjectInputStream::decodeFloat ()
{
  float  f;

  StreamCodec::decode ( *this, f, data_->SEX );

  return f;
}


//-----------------------------------------------------------------------
//   decodeDouble
//-----------------------------------------------------------------------


double ObjectInputStream::decodeDouble ()
{
  double  d;

  StreamCodec::decode ( *this, d, data_->SEX );

  return d;
}


//-----------------------------------------------------------------------
//   decode ( bool* )
//-----------------------------------------------------------------------


void ObjectInputStream::decode ( bool* buf, idx_t n )
{
  StreamCodec::decode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   decode ( byte* )
//-----------------------------------------------------------------------


void ObjectInputStream::decode ( byte* buf, idx_t n )
{
  readAll ( buf, n );
}


//-----------------------------------------------------------------------
//   decode ( char* )
//-----------------------------------------------------------------------


void ObjectInputStream::decode ( char* buf, idx_t n )
{
  StreamCodec::decode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   decode ( short* )
//-----------------------------------------------------------------------


void ObjectInputStream::decode ( short* buf, idx_t n )
{
  StreamCodec::decode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   decode ( int* )
//-----------------------------------------------------------------------


void ObjectInputStream::decode ( int* buf, idx_t n )
{
  StreamCodec::decode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   decode ( long* )
//-----------------------------------------------------------------------


void ObjectInputStream::decode ( long* buf, idx_t n )
{
  StreamCodec::decode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   decode ( lint* )
//-----------------------------------------------------------------------


void ObjectInputStream::decode ( lint* buf, idx_t n )
{
  StreamCodec::decode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   decode ( idx_t* )
//-----------------------------------------------------------------------


void ObjectInputStream::decode ( idx_t* buf, idx_t n )
{
  StreamCodec::decode ( *this, buf, n, data_->SEX, data_->options );
}


//-----------------------------------------------------------------------
//   decode ( float* )
//-----------------------------------------------------------------------


void ObjectInputStream::decode ( float* buf, idx_t n )
{
  StreamCodec::decode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   decode ( double* )
//-----------------------------------------------------------------------


void ObjectInputStream::decode ( double* buf, idx_t n )
{
  StreamCodec::decode ( *this, buf, n, data_->SEX );
}


//-----------------------------------------------------------------------
//   decodeObject
//-----------------------------------------------------------------------


Ref<Object> ObjectInputStream::decodeObject ()
{
  Data_&  d = * data_;

  idx_t   objID;
  byte    tag;

  do
  {
    tag = d.readChunk ();

    if      ( tag == RESET_TAG )
    {
      d.reset ();
    }
    else if ( tag == CLASS_TAG )
    {
      decodeClass_ ();
    }
    else
    {
      break;
    }
  }
  while ( true );

  switch ( (int) tag )
  {
  case NIL_TAG:

    return nullptr;

  case BOOL_TAG:

    return ObjRef ( d.decode<bool>(),   RAW_REF );

  case CHAR_TAG:

    return ObjRef ( d.decode<char>(),   RAW_REF );

  case STRING_TAG:

    return ObjRef ( d.decode<String>(), RAW_REF );

  case INT_TAG:

    return ObjRef ( d.decode<lint>(),   RAW_REF );

  case FLOAT_TAG:

    return ObjRef ( d.decode<double>(), RAW_REF );

  case REF_TAG:

    objID = d.decode<idx_t> ();

    return d.getObject ( objID );

  case OBJECT_TAG:

    return decodeObject_ ();

  default:

    throw SerializationException (
      JEM_FUNC,
      String::format (
        "error decoding object: unexpected %s chunk encountered",
        getChunkName ( tag )
      )
    );
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   decodeReference
//-----------------------------------------------------------------------


void ObjectInputStream::decodeReference ( const Ref<Object>& obj )
{
  Data_&  d = * data_;

  byte  tag = d.readChunk ();

  while ( tag == RESET_TAG )
  {
    d.reset ();

    tag = d.readChunk ();
  }

  if ( tag != XREF_TAG )
  {
    throw SerializationException (
      JEM_FUNC,
      String::format (
        "error decoding reference: unexpected %s chunk encountered",
        getChunkName ( tag )
      )
    );
  }

  idx_t  id = d.decode<idx_t> ();

  d.addObject ( id, obj );
}


//-----------------------------------------------------------------------
//   printTrace
//-----------------------------------------------------------------------


void ObjectInputStream::printTrace ( TextOutput& out ) const
{
  data_->printTrace ( out );
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


ObjectInputStream::Options ObjectInputStream::getOptions () const
{
  return data_->options;
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void ObjectInputStream::setOption

  ( Option  option,
    bool    yesno )

{
  data_->options.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void ObjectInputStream::setOptions ( Options options )
{
  data_->options = options;
}


//-----------------------------------------------------------------------
//   decodeClass_
//-----------------------------------------------------------------------


void ObjectInputStream::decodeClass_ ()
{
  idx_t   id   = data_->decode<idx_t>   ();
  String  name = data_->decode<String>  ();
  Class*  type = Class::findClass ( name );

  if ( ! type )
  {
    throw SerializationException (
      CLASS_NAME,
      String::format ( "undefined class: `%s\'", name )
    );
  }

  data_->addClass ( id, type );
}


//-----------------------------------------------------------------------
//   decodeObject_
//-----------------------------------------------------------------------


Ref<Object> ObjectInputStream::decodeObject_ ()
{
  Data_&  d      = * data_;

  idx_t   typeID = d.decode<idx_t> ();
  idx_t   objID  = d.decode<idx_t> ();

  Class*  type   = d.getClass ( typeID );

  ObjRef  obj;


  try
  {
    obj = type->newInstance ();
  }
  catch ( const InstantiationException& e )
  {
    throw SerializationException ( CLASS_NAME, e.what () );
  }

  Serializable*  s = dynamicCast<Serializable*> ( obj );

  if ( ! s )
  {
    String  name = type->getName ();

    throw SerializationException (
      CLASS_NAME,
      String::format (
        "class `%s\' is not serializable", name
      )
    );
  }

  d.addObject ( objID, obj );

  d.idStack.pushBack ( objID  );
  d.idStack.pushBack ( typeID );

  try
  {
    s->readFrom ( *this );
  }
  catch ( ... )
  {
    d.delObject ( objID );
    throw;
  }

  d.idStack.popBack ();
  d.idStack.popBack ();

  return obj;
}


JEM_END_PACKAGE( io )
