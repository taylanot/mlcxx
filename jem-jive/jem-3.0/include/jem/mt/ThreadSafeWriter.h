
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

#ifndef JEM_MT_THREADSAFEWRITER_H
#define JEM_MT_THREADSAFEWRITER_H

#include <jem/io/Writer.h>
#include <jem/mt/import.h>


JEM_BEGIN_PACKAGE( mt )


class IOMutex;


//-----------------------------------------------------------------------
//   class ThreadSafeWriter
//-----------------------------------------------------------------------


class ThreadSafeWriter : public Writer
{
 public:

  typedef
    ThreadSafeWriter      Self;
  typedef Writer          Super;


  explicit                ThreadSafeWriter

    ( const Ref<Writer>&    out );

                          ThreadSafeWriter

    ( const Ref<Writer>&    out,
      const Ref<IOMutex>&   mutex );

                          ThreadSafeWriter

    ( const Self&           rhs );

  virtual Ref<Writer>     dup    () override;
  void                    lock   ();
  void                    unlock ();
  virtual void            close  () override;
  virtual void            flush  () override;

  virtual void            write

    ( int                   c )     override;

  virtual void            write

    ( int                   c,
      idx_t                 n )     override;

  virtual void            write

    ( const char*           buf,
      idx_t                 n )     override;

  virtual void            writeNoThrow

    ( const char*           buf,
      idx_t                 n )     noexcept override;


 protected:

                          ThreadSafeWriter ();
  virtual                ~ThreadSafeWriter ();


 protected:

  Ref<Writer>             output_;
  Ref<IOMutex>            mutex_;

};


JEM_END_PACKAGE( mt )

#endif
