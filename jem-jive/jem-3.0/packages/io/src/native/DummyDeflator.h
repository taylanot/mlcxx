
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

#ifndef JEM_IO_NATIVE_DUMMYDEFLATOR_H
#define JEM_IO_NATIVE_DUMMYDEFLATOR_H

#include <jem/base/utilities.h>
#include <jem/io/params.h>
#include <jem/io/ZipMode.h>
#include <jem/io/OutputStream.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class Deflator
//-----------------------------------------------------------------------


class Deflator : public Collectable
{
 public:

                          Deflator

    ( Ref<OutputStream>     output,
      ZipMode               zmode,
      int                   level,
      idx_t                 bufsize );

  inline ulint            getBytesIn    () const;
  inline ulint            getBytesOut   () const;
  inline ulint            getChecksum   () const;

  inline void             sync

    ( ThrowMode             tm = CAN_THROW );

  inline void             finish

    ( ThrowMode             tm = CAN_THROW );

  inline lint             flush

    ( ThrowMode             tm = CAN_THROW );

  inline void             deflate

    ( const void*           buf,
      idx_t                 n,
      ThrowMode             tm = CAN_THROW );

  inline void             write

    ( byte                  b,
      ThrowMode             tm = CAN_THROW );


 protected:

  virtual                ~Deflator ();

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef Deflator          DummyDeflator;




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getBytesIn
//-----------------------------------------------------------------------


inline ulint Deflator::getBytesIn () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   getBytesOut
//-----------------------------------------------------------------------


inline ulint Deflator::getBytesOut () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   getChecksum
//-----------------------------------------------------------------------


inline ulint Deflator::getChecksum () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   sync
//-----------------------------------------------------------------------


inline void Deflator::sync ( ThrowMode tm )
{}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


inline void Deflator::finish ( ThrowMode tm )
{}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


inline lint Deflator::flush ( ThrowMode tm )
{
  return 0;
}


//-----------------------------------------------------------------------
//   deflate
//-----------------------------------------------------------------------


inline void Deflator::deflate

  ( const void*  buf,
    idx_t        n,
    ThrowMode    tm )

{}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


inline void Deflator::write ( byte b, ThrowMode tm )
{}


JEM_END_PACKAGE( io )

#endif
