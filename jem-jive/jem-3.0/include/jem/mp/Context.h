
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

#ifndef JEM_MP_CONTEXT_H
#define JEM_MP_CONTEXT_H

#include <jem/base/Object.h>
#include <jem/mp/params.h>
#include <jem/mp/Type.h>
#include <jem/mp/Opcode.h>


JEM_BEGIN_PACKAGE( mp )


class Status;
class RecvBuffer;
class SendBuffer;
class Request;
class RequestList;


//-----------------------------------------------------------------------
//   class Context
//-----------------------------------------------------------------------


class Context : public Object
{
 public:

  JEM_DECLARE_CLASS       ( Context, Object );

  typedef const void*       ExtensionID;


  virtual int               size           () const = 0;
  virtual int               myRank         () const = 0;
  virtual bool              isShared       () const;

  virtual void              abort

    ( int                     err )                 = 0;

  virtual Ref<Context>      clone          ()       = 0;

  virtual String            getErrorString

    ( int                     err )           const = 0;

  virtual Ref<RequestList>  newRequestList ()       = 0;

  virtual void              send

    ( const SendBuffer&       buf,
      int                     dest,
      int                     tag = DEFAULT_TAG )   = 0;

  virtual void              recv

    ( const RecvBuffer&       buf,
      int                     src,
      int                     tag  = DEFAULT_TAG,
      Status*                 stat = 0 )            = 0;

  virtual Ref<Request>      initSend

    ( const SendBuffer&       buf,
      int                     dest,
      int                     tag = DEFAULT_TAG )   = 0;

  virtual Ref<Request>      initRecv

    ( const RecvBuffer&       buf,
      int                     src,
      int                     tag = DEFAULT_TAG )   = 0;

  virtual void              barrier        ()       = 0;

  virtual void              broadcast

    ( const SendBuffer&       buf  )                = 0;

  virtual void              broadcast

    ( const RecvBuffer&       buf,
      int                     root )                = 0;

  virtual void              reduce

    ( const RecvBuffer&       rbuf,
      const SendBuffer&       sbuf,
      int                     root,
      Opcode                  op )                  = 0;

  virtual void              allreduce

    ( const RecvBuffer&       rbuf,
      const SendBuffer&       sbuf,
      Opcode                  op )                  = 0;


 protected:

  virtual                  ~Context        ();

};


JEM_END_PACKAGE( mp )

#endif
