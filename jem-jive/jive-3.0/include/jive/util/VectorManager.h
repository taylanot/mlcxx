
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_UTIL_VECTORMANAGER_H
#define JIVE_UTIL_VECTORMANAGER_H

#include <jem/base/Object.h>
#include <jem/base/Clonable.h>
#include <jem/util/Properties.h>
#include <jem/io/Serializable.h>
#include <jive/Array.h>
#include <jive/util/import.h>


JIVE_BEGIN_PACKAGE( util )


class DofSpace;
class Reordering;


//-----------------------------------------------------------------------
//   class VectorManager
//-----------------------------------------------------------------------


class VectorManager : public Object,
                      public Clonable,
                      public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( VectorManager, Object );


                          VectorManager   ();

  explicit                VectorManager

    ( const String&         name,
      const Ref<DofSpace>&  dofs );

  virtual void            readFrom

    ( ObjectInput&          in )                   override;

  virtual void            writeTo

    ( ObjectOutput&         out )            const override;

  virtual Ref<Object>     clone           () const override;
  inline StringVector     listVectors     () const;

  void                    getVector

    ( Vector&               vec,
      const String&         name );

  static void             getVector

    ( Vector&               vec,
      const String&         name,
      const Ref<DofSpace>&  dofs,
      const Properties&     globdat );

  bool                    findVector

    ( Vector&               vec,
      const String&         name );

  static bool             findVector

    ( Vector&               vec,
      const String&         name,
      const Ref<DofSpace>&  dofs,
      const Properties&     globdat );

  void                    storeVector

    ( const String&         name,
      const Vector&         vec );

  static void             storeVector

    ( const String&         name,
      const Vector&         vec,
      const Ref<DofSpace>&  dofs,
      const Properties&     globdat );

  void                    store

    ( const Properties&     globdat )        const;

  static Ref<Self>        get

    ( const Ref<DofSpace>&  dofs,
      const Properties&     globdat );

  static Ref<Self>        find

    ( const Ref<DofSpace>&  dofs,
      const Properties&     globdat );


 protected:

  virtual                ~VectorManager   ();


 private:

  void                    resize_

    ( Vector&               vec,
      idx_t                 size,
      const String&         name );

  void                    connect_        ();

  void                    dofsResized_

    ( idx_t                 count );

  void                    dofsReordered_

    ( const Reordering&     reord );


 private:

  String                  myName_;

  Ref<DofSpace>           dofs_;
  Properties              data_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   listVectors
//-----------------------------------------------------------------------


inline StringVector VectorManager::listVectors () const
{
  return data_.listProps ();
}


JIVE_END_PACKAGE( util )

#endif
