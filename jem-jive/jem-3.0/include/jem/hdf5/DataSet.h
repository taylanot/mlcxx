
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

#ifndef JEM_HDF5_DATASET_H
#define JEM_HDF5_DATASET_H

#include <jem/hdf5/Sink.h>
#include <jem/hdf5/Source.h>
#include <jem/hdf5/DataType.h>
#include <jem/hdf5/DataSpace.h>
#include <jem/hdf5/H5Object.h>


JEM_BEGIN_PACKAGE( hdf5 )


//-----------------------------------------------------------------------
//   class DataSet
//-----------------------------------------------------------------------


class DataSet : public H5Object
{
 public:

  JEM_DECLARE_CLASS       ( DataSet, H5Object );

  static const int          COMPRESS;


  explicit                  DataSet

    ( const Ref<Handle>&      oid );

  idx_t                     itemCount     () const;
  DataType                  getDataType   () const;
  DataSpace                 getDataSpace  () const;

  template <class T>
    inline void             getData

    ( T&                      data )         const;

  template <class T, int N>
    inline void             getData

    ( const Array<T,N>&       data )         const;

  template <class T>
    inline void             getData

    ( T&                      data,
      const DataSpace&        dspace )       const;

  template <class T, int N>
    inline void             getData

    ( const Array<T,N>&       data,
      const DataSpace&        dspace )       const;

  template <class T>
    inline void             setData

    ( const T&                data )         const;

  template <class T>
    inline void             setData

    ( const T&                data,
      const DataSpace&        dspace )       const;

  void                      readData

    ( void*                   buffer,
      const DataType&         type )         const;

  void                      readData

    ( void*                   buffer,
      const DataType&         type,
      const DataSpace&        mspace,
      const DataSpace&        dspace )       const;

  void                      writeData

    ( const void*             buffer,
      const DataType&         type )         const;

  void                      writeData

    ( const void*             buffer,
      const DataType&         type,
      const DataSpace&        mspace,
      const DataSpace&        dspace )       const;

  void                      countError

    ( const String&           where,
      idx_t                   mcount )       const;

  void                      countError

    ( const String&           where,
      idx_t                   mcount,
      idx_t                   dcount )       const;


 protected:

  virtual                  ~DataSet       ();


 private:

  template <class S>
    inline void             getData_

    ( const S&                sink )         const;

  template <class S>
    inline void             getData_

    ( const S&                sink,
      const DataSpace&        dspace )       const;

  template <class S>
    inline void             setData_

    ( const S&                source )       const;

  template <class S>
    inline void             setData_

    ( const S&                source,
      const DataSpace&        dspace )       const;

};




//#######################################################################
//  Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


template <class T>

  inline void DataSet::getData ( T& data )  const

{
  getData_ ( Sink<T>( data ) );
}


template <class T, int N>

  inline void DataSet::getData

  ( const Array<T,N>&  data ) const

{
  getData_ ( Sink< Array<T,N> >( data ) );
}


template <class T>

  inline void DataSet::getData

  ( T&                data,
    const DataSpace&  dspace ) const

{
  getData_ ( Sink<T>( data ), dspace );
}


template <class T, int N>

  inline void DataSet::getData

  ( const Array<T,N>&  data,
    const DataSpace&   dspace ) const

{
  getData_ ( Sink< Array<T,N> >( data ), dspace );
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


template <class T>

  inline void DataSet::setData ( const T& data ) const

{
  setData_ ( Source<T>( data ) );
}


template <class T>

  inline void DataSet::setData

  ( const T&          data,
    const DataSpace&  dspace ) const

{
  setData_ ( Source<T>( data ), dspace );
}


//-----------------------------------------------------------------------
//   getData_
//-----------------------------------------------------------------------


template <class S>

  inline void DataSet::getData_ ( const S& sink ) const

{
  if ( itemCount() != sink.count() )
  {
    countError ( JEM_FUNC, sink.count() );
  }

  readData   ( sink.addr(), sink.type() );
  sink.flush ();
}


template <class S>

  inline void DataSet::getData_

  ( const S&          sink,
    const DataSpace&  dspace ) const

{
  if ( dspace.itemCount() != sink.count() )
  {
    countError ( JEM_FUNC, sink.count(), dspace.itemCount() );
  }

  readData   ( sink.addr(), sink.type(), sink.space(), dspace );
  sink.flush ();
}


//-----------------------------------------------------------------------
//   setData_
//-----------------------------------------------------------------------


template <class S>

  inline void DataSet::setData_ ( const S& source ) const

{
  if ( itemCount() != source.count() )
  {
    countError ( JEM_FUNC, source.count() );
  }

  writeData ( source.addr(), source.type() );
}


template <class S>

  inline void DataSet::setData_

  ( const S&          source,
    const DataSpace&  dspace ) const

{
  if ( dspace.itemCount() != source.count() )
  {
    countError ( JEM_FUNC, source.count(), dspace.itemCount() );
  }

  writeData ( source.addr(), source.type(), source.space(), dspace );
}


JEM_END_PACKAGE( hdf5 )

#endif
