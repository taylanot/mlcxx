
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

#ifndef JEM_HDF5_DATATYPE_H
#define JEM_HDF5_DATATYPE_H

#include <jem/base/Ref.h>
#include <jem/base/String.h>
#include <jem/base/array/Array.h>
#include <jem/util/Flex.h>


JEM_BEGIN_PACKAGE( hdf5 )


class Handle;
class DataSpace;


//-----------------------------------------------------------------------
//   typedefs
//-----------------------------------------------------------------------


typedef util::Flex<byte>    ByteFlex;
typedef util::Flex<short>   ShortFlex;
typedef util::Flex<int>     IntFlex;
typedef util::Flex<long>    LongFlex;
typedef util::Flex<lint>    LintFlex;
typedef util::Flex<idx_t>   IdxFlex;
typedef util::Flex<float>   FloatFlex;
typedef util::Flex<double>  DoubleFlex;


//-----------------------------------------------------------------------
//   class DataType
//-----------------------------------------------------------------------


class DataType
{
 public:

  inline                    DataType     ()       noexcept;

  explicit inline           DataType

    ( const Ref<Handle>&      tid );

  bool                      operator ==

    ( const DataType&         rhs )         const;

  inline bool               operator !=

    ( const DataType&         rhs )         const;

  inline Handle*            getHandle    () const noexcept;
  inline bool               isValid      () const noexcept;
  bool                      hasVarSize   () const;

  static DataType           getType

    ( const byte*             ptr );

  static DataType           getType

    ( const char*             ptr );

  static DataType           getType

    ( const short*            ptr );

  static DataType           getType

    ( const int*              ptr );

  static DataType           getType

    ( const long*             ptr );

  static DataType           getType

    ( const lint*             ptr );

  static DataType           getType

    ( const idx_t*            ptr );

  static DataType           getType

    ( const float*            ptr );

  static DataType           getType

    ( const double*           ptr );

  static DataType           getType

    ( const String*           ptr );

  static DataType           getType

    ( const ByteFlex*         ptr );

  static DataType           getType

    ( const ShortFlex*        ptr );

  static DataType           getType

    ( const IntFlex*          ptr );

  static DataType           getType

    ( const LongFlex*         ptr );

  static DataType           getType

    ( const LintFlex*         ptr );

  static DataType           getType

    ( const IdxFlex*          ptr );

  static DataType           getType

    ( const FloatFlex*        ptr );

  static DataType           getType

    ( const DoubleFlex*       ptr );


 private:

  class                     Data_;

  Ref<Handle>               tid_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class T>

  inline DataType           typeOf  ();


template <class T>

  inline DataType           typeOf

  ( const T&                  obj );

template <class T>

  inline DataType           getType

  ( const T*                  ptr );

template <class T, int N>

  inline DataType           getType

  ( const Array<T,N>*         ptr );

void*                       vlAlloc

  ( idx_t                     count );

void                        vlDealloc

  ( void*                     addr,
    idx_t                     count,
    size_t                    size );

void                        vlReclaim

  ( void*                     addr,
    const DataType&           type,
    const DataSpace&          space );

template <class T>

  inline size_t             vlElemSize  ();

template <class T>

  inline size_t             vlElemSize

  ( const util::Flex<T>*      ptr );

inline size_t               vlElemSize

  ( const String*             ptr );



//#######################################################################
//  Implementation
//#######################################################################

//=======================================================================
//   class DataType
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline DataType::DataType () noexcept
{}


inline DataType::DataType ( const Ref<Handle>& tid ) :

  tid_ ( tid )

{}


//-----------------------------------------------------------------------
//   inequality operator
//-----------------------------------------------------------------------


inline bool DataType::operator != ( const DataType& rhs ) const
{
  return ! operator == ( rhs );
}


//-----------------------------------------------------------------------
//   getHandle
//-----------------------------------------------------------------------


inline Handle* DataType::getHandle () const noexcept
{
  return tid_.get ();
}


//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


inline bool DataType::isValid () const noexcept
{
  return (tid_ != nullptr);
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   typeOf
//-----------------------------------------------------------------------


template <class T>

  inline DataType           typeOf  ()

{
  return getType ( (const T*) nullptr );
}


template <class T>

  inline DataType           typeOf

  ( const T&                  obj )

{
  return getType ( &obj );
}


//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


template <class T>

  inline DataType           getType

  ( const T*                  ptr )

{
  return DataType::getType ( ptr );
}


template <class T, int N>

  inline DataType           getType

  ( const Array<T,N>*         ptr )

{
  return DataType::getType ( (const T*) nullptr );
}


//-----------------------------------------------------------------------
//   vlElemSize
//-----------------------------------------------------------------------


template <class T>

  inline size_t             vlElemSize  ()

{
  return vlElemSize ( (const T*) nullptr );
}


template <class T>

  inline size_t             vlElemSize

  ( const util::Flex<T>*      ptr )

{
  return sizeof(T);
}


inline size_t               vlElemSize

  ( const String*             ptr )

{
  return sizeof(char);
}


JEM_END_PACKAGE( hdf5 )

#endif
