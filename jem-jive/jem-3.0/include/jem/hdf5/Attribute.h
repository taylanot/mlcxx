
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

#ifndef JEM_HDF5_ATTRIBUTE_H
#define JEM_HDF5_ATTRIBUTE_H

#include <jem/hdf5/Sink.h>
#include <jem/hdf5/Source.h>
#include <jem/hdf5/DataType.h>


JEM_BEGIN_PACKAGE( hdf5 )


class Handle;


//-----------------------------------------------------------------------
//   class Attribute
//-----------------------------------------------------------------------


class Attribute
{
 public:

  explicit inline           Attribute

    ( const Ref<Handle>&      aid = nullptr );

  inline Handle*            getHandle     () const noexcept;
  inline bool               isValid       () const noexcept;
  idx_t                     itemCount     () const;
  DataType                  getDataType   () const;
  DataSpace                 getDataSpace  () const;
  String                    getName       () const;
  String                    getContext    () const;

  template <class T>
    inline void             getValue

    ( T&                      value )        const;

  template <class T, int N>
    inline void             getValue

    ( const Array<T,N>&       value )        const;

  template <class T>
    inline void             setValue

    ( const T&                value )        const;

  void                      readData

    ( void*                   buffer,
      const DataType&         type )         const;

  void                      writeData

    ( const void*             buffer,
      const DataType&         type )         const;

  void                      countError

    ( const String&           where,
      idx_t                   count )        const;


 private:

  template <class S>
    inline void             getValue_

    ( const S&                sink )         const;

  template <class S>
    inline void             setValue_

    ( const S&                source )       const;


 private:

  Ref<Handle>               aid_;

};




//#######################################################################
//  Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Attribute::Attribute ( const Ref<Handle>& aid ) :

  aid_ ( aid )

{}


//-----------------------------------------------------------------------
//   getHandle
//-----------------------------------------------------------------------


inline Handle* Attribute::getHandle () const noexcept
{
  return aid_.get ();
}


//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


inline bool Attribute::isValid () const noexcept
{
  return (aid_ != nullptr);
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


template <class T>

  inline void Attribute::getValue ( T& value ) const

{
  getValue_ ( Sink<T>( value ) );
}


template <class T, int N>

  inline void Attribute::getValue

  ( const Array<T,N>&  value ) const

{
  getValue_ ( Sink< Array<T,N> >( value ) );
}


//-----------------------------------------------------------------------
//   setValue
//-----------------------------------------------------------------------


template <class T>

  inline void Attribute::setValue ( const T& value ) const

{
  setValue_ ( Source<T>( value ) );
}


//-----------------------------------------------------------------------
//   getValue_
//-----------------------------------------------------------------------


template <class S>

  inline void Attribute::getValue_ ( const S& sink ) const

{
  if ( itemCount() != sink.count() )
  {
    countError ( JEM_FUNC, sink.count() );
  }

  readData   ( sink.addr(), sink.type() );
  sink.flush ();
}


//-----------------------------------------------------------------------
//   setValue_
//-----------------------------------------------------------------------


template <class S>

  inline void Attribute::setValue_ ( const S& source ) const

{
  if ( itemCount() != source.count() )
  {
    countError ( JEM_FUNC, source.count() );
  }

  writeData ( source.addr(), source.type() );
}


JEM_END_PACKAGE( hdf5 )

#endif
