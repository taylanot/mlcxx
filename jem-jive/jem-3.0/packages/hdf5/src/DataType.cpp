
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


#include <jem/base/Once.h>
#include <jem/base/Error.h>
#include <jem/base/MemCache.h>
#include <jem/base/ReadWriteMutex.h>
#include <jem/hdf5/Library.h>
#include <jem/hdf5/DataType.h>
#include <jem/hdf5/DataSpace.h>
#include <jem/hdf5/H5Exception.h>
#include "Handle.h"


JEM_BEGIN_PACKAGE( hdf5 )


//=======================================================================
//   class DataType::Data_
//=======================================================================


class DataType::Data_
{
 public:

  enum                      Index
  {
                              BYTE,
                              CHAR,
                              SHORT,
                              INT,
                              LONG,
                              LINT,
                              IDX,
                              FLOAT,
                              DOUBLE
  };

  enum                    { TYPE_COUNT = DOUBLE };


  static DataType           getType

    ( Index                   index );

  static DataType           getVLType

    ( Index                   index );


 private:

  static void               newInstance_  ();

  static hid_t              getBaseType_

    ( Index                   index );


 private:

  static Once               once_;
  static Data_*             instance_;

  ReadWriteMutex            mutex_;
  DataType                  types_  [TYPE_COUNT];
  DataType                  vlTypes_[TYPE_COUNT];

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Once              DataType::Data_::once_     = JEM_ONCE_INITIALIZER;
DataType::Data_*  DataType::Data_::instance_ = nullptr;


//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


DataType DataType::Data_::getType ( Index index )
{
  runOnce ( once_, newInstance_ );

  Data_&     data = *instance_;
  DataType&  type = data.types_[index];

  bool       valid;

  {
    ReadLock  lock ( data.mutex_ );

    valid = type.isValid ();
  }

  if ( ! valid )
  {
    WriteLock  ( data.mutex_ );

    if ( ! type.isValid() )
    {
      type = DataType (
        newInstance<Handle> ( getBaseType_( index ) )
      );
    }
  }

  return type;
}


//-----------------------------------------------------------------------
//   getVLType
//-----------------------------------------------------------------------


DataType DataType::Data_::getVLType ( Index index )
{
  runOnce ( once_, newInstance_ );

  Data_&     data = *instance_;
  DataType&  type = data.vlTypes_[index];

  bool       valid;

  {
    ReadLock  lock ( data.mutex_ );

    valid = type.isValid ();
  }

  if ( ! valid )
  {
    WriteLock  ( data.mutex_ );

    if ( ! type.isValid() )
    {
      hid_t  tid = H5Tvlen_create ( getBaseType_( index ) );

      if ( tid < 0 )
      {
        throw H5Exception (
          JEM_FUNC,
          "failed to create a variable length data type",
          Library::strerror ()
        );
      }

      type = DataType ( newInstance<Handle>( tid ) );
    }
  }

  return type;
}


//-----------------------------------------------------------------------
//   newInstance_
//-----------------------------------------------------------------------


void DataType::Data_::newInstance_ ()
{
  Library::init ();

  instance_ = new Data_;
}


//-----------------------------------------------------------------------
//   getBaseType_
//-----------------------------------------------------------------------


hid_t DataType::Data_::getBaseType_ ( Index index )
{
  hid_t  base = 0;

  switch ( index )
  {
  case BYTE:

    base = H5T_NATIVE_UCHAR;
    break;

  case CHAR:

    base = H5T_NATIVE_CHAR;
    break;

  case SHORT:

    base = H5T_NATIVE_SHORT;
    break;

  case INT:

    base = H5T_NATIVE_INT;
    break;

  case LONG:

    base = H5T_NATIVE_LONG;
    break;

  case LINT:

    base = H5T_NATIVE_LLONG;
    break;

  case IDX:

#ifdef JEM_INDEX_IS_INT
    base = H5T_NATIVE_INT;
#else
    base = H5T_NATIVE_LLONG;
#endif

    break;

  case FLOAT:

    base = H5T_NATIVE_FLOAT;
    break;

  case DOUBLE:

    base = H5T_NATIVE_DOUBLE;
    break;

  default:

    throw Error (
      JEM_FUNC,
      String::format (
        "invalid type index: %d",
        (int) index
      )
    );
  }

  return base;
}


//=======================================================================
//   class DataType
//=======================================================================

//-----------------------------------------------------------------------
//   operator ==
//-----------------------------------------------------------------------


bool DataType::operator == ( const DataType& rhs ) const
{
  if      ( tid_ == rhs.tid_ )
  {
    return true;
  }
  else if ( (tid_ != nullptr) && (rhs.tid_ != nullptr) )
  {
    return (H5Tequal( tid_    ->getID(),
                      rhs.tid_->getID() ) > 0);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   hasVarSize()
//-----------------------------------------------------------------------


bool DataType::hasVarSize () const
{
  if ( tid_ )
  {
    hid_t        id = tid_->getID  ();
    H5T_class_t  c  = H5Tget_class ( id );

    if ( c == H5T_VLEN )
    {
      return true;
    }

    if ( c == H5T_STRING )
    {
      if ( H5Tis_variable_str( id ) > 0 )
      {
        return true;
      }
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   getType()
//-----------------------------------------------------------------------


DataType DataType::getType ( const byte* ptr )
{
  return Data_::getType ( Data_::BYTE );
}


DataType DataType::getType ( const char* ptr )
{
  return Data_::getType ( Data_::CHAR );
}


DataType DataType::getType ( const short* ptr )
{
  return Data_::getType ( Data_::SHORT );
}


DataType DataType::getType ( const int* ptr )
{
  return Data_::getType ( Data_::INT );
}


DataType DataType::getType ( const long* ptr )
{
  return Data_::getType ( Data_::LONG );
}


DataType DataType::getType ( const lint* ptr )
{
  return Data_::getType ( Data_::LINT );
}


DataType DataType::getType ( const idx_t* ptr )
{
  return Data_::getType ( Data_::IDX );
}


DataType DataType::getType ( const float* ptr )
{
  return Data_::getType ( Data_::FLOAT );
}


DataType DataType::getType ( const double* ptr )
{
  return Data_::getType ( Data_::DOUBLE );
}


DataType DataType::getType ( const String* ptr )
{
  return Data_::getVLType ( Data_::CHAR );
}


DataType DataType::getType ( const ByteFlex* ptr )
{
  return Data_::getVLType ( Data_::BYTE );
}


DataType DataType::getType ( const ShortFlex* ptr )
{
  return Data_::getVLType ( Data_::SHORT );
}


DataType DataType::getType ( const IntFlex* ptr )
{
  return Data_::getVLType ( Data_::INT );
}


DataType DataType::getType ( const LongFlex* ptr )
{
  return Data_::getVLType ( Data_::LONG );
}


DataType DataType::getType ( const LintFlex* ptr )
{
  return Data_::getVLType ( Data_::LINT );
}


DataType DataType::getType ( const IdxFlex* ptr )
{
  return Data_::getVLType ( Data_::IDX );
}


DataType DataType::getType ( const FloatFlex* ptr )
{
  return Data_::getVLType ( Data_::FLOAT );
}


DataType DataType::getType ( const DoubleFlex* ptr )
{
  return Data_::getVLType ( Data_::DOUBLE );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   vlAlloc()
//-----------------------------------------------------------------------


void* vlAlloc ( idx_t count )
{
  JEM_PRECHECK ( count >= 0 );

  hvl_t*  data =

    (hvl_t*) MemCache::alloc ( (size_t) count * sizeof(hvl_t) );

  for ( idx_t i = 0; i < count; i++ )
  {
    data[i].len = 0;
    data[i].p   = nullptr;
  }

  return data;
}


//-----------------------------------------------------------------------
//   vlDealloc()
//-----------------------------------------------------------------------


void      vlDealloc

  ( void*   addr,
    idx_t   count,
    size_t  size )

{
  JEM_PRECHECK ( count >= 0 );

  hvl_t*  data = (hvl_t*) addr;

  for ( idx_t i = 0; i < count; i++ )
  {
    if ( data[i].p )
    {
      MemCache::dealloc ( data[i].p, data[i].len * size );

      data[i].len = 0;
      data[i].p   = nullptr;
    }
  }

  MemCache::dealloc ( addr, (size_t) count * sizeof(hvl_t) );
}


//-----------------------------------------------------------------------
//   vlReclaim()
//-----------------------------------------------------------------------


void                vlReclaim

  ( void*             addr,
    const DataType&   type,
    const DataSpace&  space )

{
  JEM_PRECHECK ( type.isValid() && space.isValid() );

  H5Dvlen_reclaim   ( type .getHandle()->getID (),
                      space.getHandle()->getID (),
                      H5P_DEFAULT, addr );
  MemCache::dealloc ( addr, (size_t) space.itemCount() *
                            sizeof(hvl_t) );
}


JEM_END_PACKAGE( hdf5 )

