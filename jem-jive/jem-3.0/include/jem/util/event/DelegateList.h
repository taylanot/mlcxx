
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

#ifndef JEM_UTIL_EVENT_DELEGATELIST_H
#define JEM_UTIL_EVENT_DELEGATELIST_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( util )


class DelegateIterator;
class AbstractDelegate;


//-----------------------------------------------------------------------
//   class DelegateList
//-----------------------------------------------------------------------


class DelegateList
{
 public:

  friend class            DelegateIterator;

  typedef DelegateList    Self;


                          DelegateList    ();

  void                    pushBack

    ( AbstractDelegate*     d );

  void                    erase

    ( AbstractDelegate*     d );

  bool                    enable

    ( bool                  flag );

  void                    clear           ();
  inline bool             isEmpty         () const noexcept;

  static void             addRef

    ( Self*                 list );

  static void             delRef

    ( Self*                 list );


 private:

                          DelegateList    ( const Self& );
  DelegateList&           operator =      ( const Self& );

  virtual                ~DelegateList    ();


  void                    erase_

    ( AbstractDelegate*     d    );

  static void             clear_

    ( AbstractDelegate*&    list );


 private:

  static const byte       MAX_GC_COUNT_;

  static const byte       IS_ENABLED_;
  static const byte       CLEAR_PENDING_;


  AbstractDelegate*       first_;
  AbstractDelegate*       eraseList_;

  short                   refcount_;
  short                   iterCount_;
  byte                    gcCount_;
  byte                    status_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   isEmpty
//-----------------------------------------------------------------------


inline bool DelegateList::isEmpty () const noexcept
{
  return (first_ == nullptr);
}


JEM_END_PACKAGE( util )

#endif
