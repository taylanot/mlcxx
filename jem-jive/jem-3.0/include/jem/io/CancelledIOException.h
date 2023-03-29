
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

#ifndef JEM_IO_CANCELLEDIOEXCEPTION_H
#define JEM_IO_CANCELLEDIOEXCEPTION_H

#include <jem/base/CancelledException.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class CancelledIOException
//-----------------------------------------------------------------------


class CancelledIOException : public CancelledException
{
 public:

  typedef CancelledIOException  Self;
  typedef CancelledException    Super;


  explicit                  CancelledIOException

    ( const String&           where = "",
      const String&           what  = "I/O operation cancelled" );

  explicit                  CancelledIOException

    ( idx_t                   n,
      const String&           where = "",
      const String&           what  = "" );

                            CancelledIOException

    ( const Self&             rhs );

  virtual String            name   () const override;


 public:

  const idx_t               bytesTransferred;

};


JEM_END_PACKAGE( io )

#endif
