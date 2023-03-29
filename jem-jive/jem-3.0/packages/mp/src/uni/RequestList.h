
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

#ifndef JEM_MP_UNI_REQUESTLIST_H
#define JEM_MP_UNI_REQUESTLIST_H

#include <jem/util/Flex.h>
#include <jem/mp/RequestList.h>


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( uni )


class Request;
class MessagePool;


//-----------------------------------------------------------------------
//   class RequestList
//-----------------------------------------------------------------------


class RequestList : public mp::RequestList
{
 public:

  typedef RequestList       Self;
  typedef mp::RequestList   Super;
  typedef MessagePool       MPool;


  explicit              RequestList

    ( const Ref<MPool>&   mpool );

  virtual idx_t         size    () const override;

  virtual void          clear

    ( Status*             stats )        override;

  virtual idx_t         addRequest

    ( const SendBuffer&   buf,
      int                 dest,
      int                 tag )          override;

  virtual idx_t         addRequest

    ( const RecvBuffer&   buf,
      int                 src,
      int                 tag )          override;

  virtual void          startOne

    ( idx_t               ireq )         override;

  virtual void          startAll ()      override;

  virtual bool          testOne

    ( idx_t               ireq,
      Status*             stat )         override;

  virtual void          testSome

    ( idx_t&              count,
      idx_t*              ireqs,
      Status*             stats )        override;

  virtual bool          testAll

    ( Status*             stats )        override;

  virtual void          waitOne

    ( idx_t               ireq,
      Status*             stat )         override;

  virtual void          waitSome

    ( idx_t&              count,
      idx_t*              ireqs,
      Status*             stats )        override;

  virtual void          waitAll

    ( Status*             stats )        override;

  virtual void          cancelOne

    ( idx_t               ireq )         override;

  virtual void          cancelAll

    ( Status*             stats )        override;


 protected:

  virtual              ~RequestList ();


 private:

  void                  setStats_

    ( Status*             stats ) const;

  int                   getError_

    ( const idx_t*        ireqs,
      idx_t               count ) const;

  void                  indexError_

    ( const String&       where,
      idx_t               ireq  ) const;


 private:

  Ref<MPool>            mpool_;
  util::Flex
    < Ref<Request> >    reqs_;

};


JEM_END_NAMESPACE ( uni )
JEM_END_PACKAGE   ( mp )

#endif
