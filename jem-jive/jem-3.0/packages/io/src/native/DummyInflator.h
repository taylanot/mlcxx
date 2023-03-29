
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

#ifndef JEM_IO_INFLATOR_H
#define JEM_IO_INFLATOR_H

#include <jem/base/utilities.h>
#include <jem/io/params.h>
#include <jem/io/ZipMode.h>
#include <jem/io/InputStream.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class Inflator
//-----------------------------------------------------------------------


class Inflator : public Collectable
{
 public:

                          Inflator

    ( Ref<InputStream>      input,
      ZipMode               zmode,
      idx_t                 bufsize );

  inline bool             atEnd         () const;
  inline ulint            getBytesIn    () const;
  inline ulint            getBytesOut   () const;
  inline ulint            getChecksum   () const;

  inline idx_t            poll

    ( const Time&           timeout );

  inline void             finish

    ( ThrowMode             tm = CAN_THROW );

  inline lint             inflate

    ( void*                 buf,
      idx_t                 n );

  inline bool             read

    ( byte&                 b );


 protected:

  virtual                ~Inflator      ();

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef Inflator          DummyInflator;





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   atEnd
//-----------------------------------------------------------------------


inline bool Inflator::atEnd () const
{
  return true;
}


//-----------------------------------------------------------------------
//   getBytesIn
//-----------------------------------------------------------------------


inline ulint Inflator::getBytesIn () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   getBytesOut
//-----------------------------------------------------------------------


inline ulint Inflator::getBytesOut () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   getChecksum
//-----------------------------------------------------------------------


inline ulint Inflator::getChecksum () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


inline idx_t Inflator::poll ( const Time& timeout )
{
  return 0;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


inline void Inflator::finish ( ThrowMode tm )
{}


//-----------------------------------------------------------------------
//   inflate
//-----------------------------------------------------------------------


inline lint Inflator::inflate ( void* buf, idx_t n )
{
  return 0;
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


inline bool Inflator::read ( byte& b )
{
  return false;
}


JEM_END_PACKAGE( io )

#endif
