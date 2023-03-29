
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

#ifndef JEM_UTIL_OBJECTCOLLECTION_H
#define JEM_UTIL_OBJECTCOLLECTION_H

#include <jem/util/Enumerable.h>


namespace jem
{
  namespace io
  {
    class ObjectInput;
    class ObjectOutput;
  }
}


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class ObjectCollection
//-----------------------------------------------------------------------


class ObjectCollection : public Object,
                         public Enumerable
{
 public:

  JEM_DECLARE_CLASS       ( ObjectCollection, Object );


  Ref<Enumerator>           enumerate         () const;
  virtual Ref<Enumerator>   getEnumerator     () const override = 0;

  virtual bool              add

    ( const Ref<Object>&      obj )                             = 0;

  virtual void              clear             ()                = 0;

  virtual void              reserve

    ( idx_t                   count );

  virtual void              trimToSize        ();
  virtual idx_t             capacity          () const;
  virtual idx_t             size              () const          = 0;


 protected:

  virtual                  ~ObjectCollection  ();

  void                      clone_

    ( const Self&             rhs );

  void                      readFrom_

    ( io::ObjectInput&        in );

  void                      writeTo_

    ( io::ObjectOutput&       out )              const;

  static idx_t               expand_

    ( idx_t                   cap );

  void                      writeError_       () const;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef ObjectCollection    ObjCollection;


JEM_END_PACKAGE( util )

#endif
