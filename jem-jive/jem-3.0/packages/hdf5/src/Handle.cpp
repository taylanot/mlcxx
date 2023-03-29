
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


#include <jem/hdf5/H5Exception.h>
#include "Handle.h"


JEM_BEGIN_PACKAGE( hdf5 )


//=======================================================================
//   class Handle
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Handle::ALLOCATED = 1 << 0;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Handle::~Handle ()
{
  if ( flags_ & ALLOCATED )
  {
    H5Idec_ref ( id_ );
  }
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Handle::toString () const
{
  return String::format ( "HDF5 handle %d", (lint) id_ );
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint Handle::hashValue () const
{
  return (lint) id_;
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool Handle::equals ( const Ref<Object>& rhs ) const
{
  Self*  handle = dynamic_cast<Self*> ( rhs.get() );

  if ( ! handle )
  {
    return false;
  }
  else
  {
    return (id_ == handle->id_);
  }
}


JEM_END_PACKAGE( hdf5 )

