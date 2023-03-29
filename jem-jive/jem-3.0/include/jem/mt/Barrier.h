
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

#ifndef JEM_MT_BARRIER_H
#define JEM_MT_BARRIER_H

#include <jem/base/Monitor.h>


JEM_BEGIN_PACKAGE( mt )


//-----------------------------------------------------------------------
//   class Barrier
//-----------------------------------------------------------------------


class Barrier
{
 public:

  static const int    LAST_THREAD;


  explicit            Barrier

    ( int               n );

                     ~Barrier   ();

  int                 wait      ();
  inline int          size      () const noexcept;


 private:

  const int           size_;
  int                 count_;
  bool                cycle_;
  Monitor             monitor_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline int Barrier::size () const noexcept
{
  return size_;
}


JEM_END_PACKAGE( mt )

#endif
