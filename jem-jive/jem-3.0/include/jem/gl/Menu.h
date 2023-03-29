
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

#ifndef JEM_GL_MENU_H
#define JEM_GL_MENU_H

#include <jem/base/Object.h>
#include <jem/base/Clonable.h>
#include <jem/base/array/Array.h>
#include <jem/util/List.h>


JEM_BEGIN_PACKAGE( gl )


class MenuNode;


//-----------------------------------------------------------------------
//   class Menu
//-----------------------------------------------------------------------


class Menu : public Object,
             public Clonable
{
 public:

  JEM_DECLARE_CLASS       ( Menu, Object );

  friend class              MenuNode;


                            Menu          ();

  explicit                  Menu

  ( const Array<String>&      list );

                            Menu

    ( const Menu&             rhs );

  virtual Ref<Object>       clone         () const override;
  void                      clear         ();
  void                      addLine       ();

  void                      addEntry

    ( const String&           entry );

  void                      addOption

    ( const String&           first,
      const String&           second,
      bool                    state );

  void                      insertEntry

    ( const String&           entry,
      const String&           next );

  void                      eraseEntry

    ( const String&           entry );

  Array<String>             listEntries   () const;


 protected:

  virtual                  ~Menu          ();


 private:

  util::List<String>        entries_;

};


JEM_END_PACKAGE( gl )

#endif
