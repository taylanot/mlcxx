
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

#ifndef JIVE_UTIL_STDDATABASE_H
#define JIVE_UTIL_STDDATABASE_H

#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jive/util/Database.h>


JIVE_BEGIN_PACKAGE( util )


class Reordering;


//-----------------------------------------------------------------------
//   class StdDatabase
//-----------------------------------------------------------------------


class StdDatabase : public Database,
                    public Clonable,
                    public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( StdDatabase, Database );


                            StdDatabase ();

                            StdDatabase

    ( const String&           name,
      const Ref<ItemSet>&     items );

                            StdDatabase

    ( const Self&             rhs );

  virtual void              resetEvents     ()       override;

  virtual void              readFrom

    ( ObjectInput&            in )                   override;

  virtual void              writeTo

    ( ObjectOutput&           out )            const override;

  virtual Ref<Object>       clone           () const override;
  virtual ItemSet*          getRowItems     () const override;
  virtual void              clearAll        ()       override;

  virtual IntColumn*        addIntColumn

    ( const String&           name )                 override;

  virtual IntColumn*        findIntColumn

    ( const String&           name )           const override;

  virtual IntColArray       getIntColumns   () const override;

  virtual FloatColumn*      addFloatColumn

    ( const String&           name )                 override;

  virtual FloatColumn*      findFloatColumn

    ( const String&           name )           const override;

  virtual FloatColArray     getFloatColumns () const override;


 protected:

  virtual                  ~StdDatabase     ();

  virtual void              emitEvents_

    ( int                     events )               override;


 private:

  void                      connect_        ();

  void                      itemsResized_

    ( idx_t                   size );

  void                      itemsReordered_

    ( const Reordering&       reord );


 private:

  class                     Data_;

  Ref<Data_>                data_;

};


JIVE_END_PACKAGE( util )

#endif
