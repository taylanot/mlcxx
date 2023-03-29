
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

#ifndef JEM_MP_BCASTSTREAM_H
#define JEM_MP_BCASTSTREAM_H

#include <jem/io/FileFlags.h>
#include <jem/io/InputStream.h>


JEM_BEGIN_PACKAGE( mp )


class Context;


//-----------------------------------------------------------------------
//   class BcastStream
//-----------------------------------------------------------------------


class BcastStream : public io::InputStream,
                    public io::FileFlags
{
 public:

  JEM_DECLARE_CLASS       ( BcastStream, io::InputStream );

  typedef io::InputStream   Input;


                            BcastStream

    ( int                     root,
      const Ref<Context>&     ctx,
      const Ref<Input>&       in,
      idx_t                   bufsize = -1 );

  static Ref<Self>          open

    ( int                     root,
      const Ref<Context>&     ctx,
      const String&           fname,
      OpenFlags               flags   =  0,
      idx_t                   bufsize = -1 );

  virtual void              close         () override;

  virtual idx_t             poll

    ( const Time&             timeout )      override;

  virtual idx_t             read

    ( void*                   buf,
      idx_t                   n )            override;


 protected:

  virtual                  ~BcastStream   ();


 private:

  void                      allocBuffer_

    ( Context&                ctx );

  void                      fillBuffer_   ();


 private:

  int                       root_;
  int                       myRank_;
  int                       procCount_;
  Ref<Context>              context_;
  Ref<Input>                input_;
  byte*                     buffer_;
  idx_t                     capacity_;
  idx_t                     first_;
  idx_t                     last_;
  bool                      isOpen_;

};


JEM_END_PACKAGE( mp )

#endif
