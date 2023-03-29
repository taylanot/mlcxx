
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

#ifndef JEM_UTIL_EVENT_ABSTRACTDELEGATE_H
#define JEM_UTIL_EVENT_ABSTRACTDELEGATE_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( util )


class DelegateList;
class DelegateIterator;


//-----------------------------------------------------------------------
//   class AbstractDelegate
//-----------------------------------------------------------------------


class AbstractDelegate
{
 public:

  typedef
    AbstractDelegate    Self;

  friend class          DelegateList;
  friend class          DelegateIterator;


  virtual bool          isConnected      () const = 0;

  virtual bool          equals

    ( Self*               rhs  )            const = 0;

  static inline void    free

    ( Self*               self );


 protected:

  inline                AbstractDelegate () noexcept;
  virtual              ~AbstractDelegate ();


 private:

                        AbstractDelegate ( const Self& );
  Self&                 operator =       ( const Self& );


 private:

  Self*                 next_;

};





//#######################################################################
//   Implementation
//#######################################################################


inline AbstractDelegate::AbstractDelegate () noexcept :

  next_ ( nullptr )

{}


inline void AbstractDelegate::free ( Self* self )
{
  delete self;
}


JEM_END_PACKAGE( util )

#endif
