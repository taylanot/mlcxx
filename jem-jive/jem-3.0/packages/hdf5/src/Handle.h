
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

#ifndef JEM_HDF5_HANDLE_H
#define JEM_HDF5_HANDLE_H

#include <jem/base/Object.h>
#include <jem/hdf5/Library.h>
#include "jemhdf5.h"


JEM_BEGIN_PACKAGE( hdf5 )


//-----------------------------------------------------------------------
//   class Handle
//-----------------------------------------------------------------------


class Handle : public Object
{
 public:

  typedef Object            Super;
  typedef Handle            Self;

  static const int          ALLOCATED;


  inline explicit           Handle

    ( hid_t                   id,
      int                     flags = 0 );

  virtual String            toString   () const override;
  virtual lint              hashValue  () const override;

  virtual bool              equals

    ( const Ref<Object>&      rhs )       const override;

  inline hid_t              getID      () const noexcept;
  inline hid_t              release    ()       noexcept;

 protected:

  virtual                  ~Handle     ();

 private:

  hid_t                     id_;
  int                       flags_;

};



//#######################################################################
//  Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Handle::Handle

  ( hid_t  id,
    int    flags )

{
  Library::init ();

  id_    = id;
  flags_ = flags;
}


//-----------------------------------------------------------------------
//   getID
//-----------------------------------------------------------------------


inline hid_t Handle::getID () const noexcept
{
  return id_;
}


//-----------------------------------------------------------------------
//   release()
//-----------------------------------------------------------------------


inline hid_t Handle::release () noexcept
{
  hid_t  id = id_;

  id_    = 0;
  flags_ = 0;

  return id;
}


JEM_END_PACKAGE( hdf5 )

#endif
