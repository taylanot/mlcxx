
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

#ifndef JEM_IO_FILESTREAM_H
#define JEM_IO_FILESTREAM_H

#include <jem/base/Object.h>


namespace jem
{
  class Time;
}


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class FileStream
//-----------------------------------------------------------------------


class FileStream : public Object
{
 public:

  JEM_DECLARE_CLASS     ( FileStream, Object );

  static const int        POLL_READ;
  static const int        POLL_WRITE;


  virtual Ref<Self>       dup           ()          = 0;
  virtual void            sync          ()          = 0;
  virtual void            close         ()          = 0;
  virtual void            closeNoThrow  () noexcept = 0;

  virtual int             poll

    ( const Time&           timeout )               = 0;

  virtual idx_t           read

    ( void*                 buf,
      idx_t                 n )                     = 0;

  virtual void            write

    ( const void*           buf,
      idx_t                 n )                     = 0;

  virtual void            writeNoThrow

    ( const void*           buf,
      idx_t                 n )            noexcept = 0;

  virtual lint            getPosition   () const    = 0;

  virtual void            setPosition

    ( lint                  pos )                   = 0;

  virtual lint            getLength     () const    = 0;


 protected:

  virtual                ~FileStream   ();

};


JEM_END_PACKAGE( io )

#endif
