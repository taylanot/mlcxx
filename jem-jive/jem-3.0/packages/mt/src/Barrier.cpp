
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
#include <jem/mt/Barrier.h>


JEM_BEGIN_PACKAGE( mt )


//=======================================================================
//   class Barrier
//=======================================================================

//-----------------------------------------------------------------------
//   static data members
//-----------------------------------------------------------------------


const int  Barrier::LAST_THREAD = 0;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Barrier::Barrier ( int n ) : size_( n )
{
  JEM_PRECHECK( n > 0 );

  count_ = 0;
  cycle_ = true;
}


Barrier::~Barrier()
{}


//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


int Barrier::wait ()
{
  Lock<Monitor>  lock  ( monitor_ );

  bool           cycle = cycle_;


  count_++;

  if ( count_ == size_ )
  {
    count_ = 0;
    cycle_ = ! cycle_;

    // Wake up all waiting threads

    monitor_.notifyAll ();

    return LAST_THREAD;
  }
  else
  {
    while ( cycle == cycle_ )
    {
      monitor_.wait ();
    }

    return -1;
  }
}


JEM_END_PACKAGE( mt )
