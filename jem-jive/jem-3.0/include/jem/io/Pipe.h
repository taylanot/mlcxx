
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

#ifndef JEM_IO_PIPE_H
#define JEM_IO_PIPE_H

#include <jem/base/Mutex.h>
#include <jem/base/FastCondition.h>
#include <jem/base/Object.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class Pipe
//-----------------------------------------------------------------------


class Pipe : public Object
{
 public:

  typedef Pipe        Self;
  typedef Object      Super;


  explicit            Pipe

    ( idx_t              bufsize = 1024 );

  void                addReader     ();
  void                addWriter     ();
  void                deleteReader  ();
  void                deleteWriter  ();

  idx_t               poll

    ( const Time&       timeout );

  idx_t               read

    ( void*             buf,
      idx_t             n );

  void                write

    ( const void*       buf,
      idx_t             n );

  void                writeNoThrow

    ( const void*       buf,
      idx_t             n )            noexcept;


 protected:

  virtual            ~Pipe          ();


 private:

  static const byte   CANARY_VALUE_;

  Mutex               mutex_;
  FastCondition       readers_;
  FastCondition       writers_;

  int                 readerCount_;
  int                 writerCount_;

  byte*               buffer_;
  idx_t               bufsize_;
  idx_t               first_;
  idx_t               last_;

  bool                cycle_;

};


JEM_END_PACKAGE( io )

#endif
