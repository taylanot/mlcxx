
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

#ifndef JEM_UTIL_OBJECTSEQUENCE_H
#define JEM_UTIL_OBJECTSEQUENCE_H

#include <jem/util/ObjectCollection.h>
#include <jem/util/SequenceEnumerator.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class ObjectSequence
//-----------------------------------------------------------------------


class ObjectSequence : public ObjectCollection
{
 public:

  JEM_DECLARE_CLASS       ( ObjectSequence, ObjectCollection );


  Ref<SeqEnum>              enumerate       () const;
  virtual Ref<Enumerator>   getEnumerator   () const override;
  virtual Ref<SeqEnum>      getSeqEnum      () const = 0;

  virtual void              insert

    ( idx_t                   idx,
      const Ref<Object>&      obj )                  = 0;

  virtual void              insertSpace

    ( idx_t                   idx,
      idx_t                   count );

  virtual void              erase

    ( idx_t                   idx )                  = 0;

  virtual void              eraseRange

    ( idx_t                   first,
      idx_t                   last );

  virtual Ref<Object>       get

    ( idx_t                   idx )            const = 0;

  virtual void              set

    ( idx_t                   idx,
      const Ref<Object>&      obj )                  = 0;

  virtual bool              add

    ( const Ref<Object>&      obj )                  override;

  virtual Ref<Object>       front           () const;
  virtual Ref<Object>       back            () const;

  virtual void              pushBack

    ( const Ref<Object>&      obj );

  virtual void              pushBack

    ( Ref<Object>&&           obj );

  virtual void              pushFront

    ( const Ref<Object>&      obj );

  virtual void              pushFront

    ( Ref<Object>&&           obj );

  virtual void              popBack         ();
  virtual void              popFront        ();
  virtual void              clear           ()       override;


 protected:

  virtual                  ~ObjectSequence  ();


 protected:

  void                      emptyError_     () const;

  void                      indexError_

    ( idx_t                   idx )            const;

  void                      indexError_

    ( idx_t                   idx,
      idx_t                   ubound )         const;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef ObjectSequence      ObjSequence;



JEM_END_PACKAGE( util )

#endif
