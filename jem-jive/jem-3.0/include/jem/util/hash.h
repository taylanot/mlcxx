
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

#ifndef JEM_UTIL_HASH_H
#define JEM_UTIL_HASH_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   hash utility functions
//-----------------------------------------------------------------------


lint                hash

  ( const void*       key,
    idx_t             len,
    lint              ini = 0_lint ) noexcept;

int                 hashBits

  ( idx_t             count ) noexcept;

inline lint         hashMask

  ( int               nbits ) noexcept;

inline idx_t        hashSize

  ( int               nbits ) noexcept;




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   hashMask
//-----------------------------------------------------------------------


inline lint   hashMask ( int nbits ) noexcept
{
  return ((1_lint << nbits) - 1_lint);
}


//-----------------------------------------------------------------------
//   hashSize
//-----------------------------------------------------------------------


inline idx_t  hashSize ( int nbits ) noexcept
{
  return (1_idx << nbits);
}


JEM_END_PACKAGE( util )

#endif
