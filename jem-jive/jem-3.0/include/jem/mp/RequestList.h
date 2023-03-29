
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

#ifndef JEM_MP_REQUESTLIST_H
#define JEM_MP_REQUESTLIST_H

#include <jem/base/Object.h>
#include <jem/mp/params.h>
#include <jem/mp/Status.h>


JEM_BEGIN_PACKAGE( mp )


class RecvBuffer;
class SendBuffer;


//-----------------------------------------------------------------------
//   class RequestList
//-----------------------------------------------------------------------


class RequestList : public Object
{
 public:

  typedef RequestList   Self;
  typedef Object        Super;


  virtual idx_t         size         () const = 0;

  virtual void          clear

    ( Status*             stats = 0 )         = 0;

  virtual idx_t         addRequest

    ( const SendBuffer&   buf,
      int                 dest,
      int                 tag = DEFAULT_TAG ) = 0;

  virtual idx_t         addRequest

    ( const RecvBuffer&   buf,
      int                 src,
      int                 tag = DEFAULT_TAG ) = 0;

  virtual void          startOne

    ( idx_t               ireq )              = 0;

  virtual void          startAll     ()       = 0;

  virtual bool          testOne

    ( idx_t               ireq,
      Status*             stat = 0 )          = 0;

  virtual void          testSome

    ( idx_t&              count,
      idx_t*              ireqs,
      Status*             stats = 0 )         = 0;

  virtual bool          testAll

    ( Status*             stats = 0 )         = 0;

  virtual void          waitOne

    ( idx_t               ireq,
      Status*             stat = 0 )          = 0;

  virtual void          waitSome

    ( idx_t&              count,
      idx_t*              ireqs,
      Status*             stats = 0 )         = 0;

  virtual void          waitAll

    ( Status*             stats = 0 )         = 0;

  virtual void          cancelOne

    ( idx_t               ireq )              = 0;

  virtual void          cancelAll

    ( Status*             stats = 0 )         = 0;


 protected:

  virtual              ~RequestList ();

};


JEM_END_PACKAGE( mp )

#endif
