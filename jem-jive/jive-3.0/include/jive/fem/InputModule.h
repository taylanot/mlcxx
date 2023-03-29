
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

#ifndef JIVE_FEM_INPUTMODULE_H
#define JIVE_FEM_INPUTMODULE_H

#include <jive/util/StorageMode.h>
#include <jive/app/Module.h>
#include <jive/fem/import.h>


JIVE_BEGIN_PACKAGE( fem )


class DataParser;


//-----------------------------------------------------------------------
//   class InputModule
//-----------------------------------------------------------------------


class InputModule : public Module
{
 public:

  JEM_DECLARE_CLASS       ( InputModule, Module );

  static const char*        TYPE_NAME;

  static const int          READ_NODES;
  static const int          READ_ELEMS;
  static const int          READ_MESH;
  static const int          READ_BOUNDS;
  static const int          READ_CONS;
  static const int          READ_GROUPS;
  static const int          READ_TABLES;
  static const int          READ_DBASES;
  static const int          READ_FUNCS;
  static const int          READ_ALL;


  explicit                  InputModule

    ( const String&           name     = "input",
      int                     mask     = READ_ALL );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat )      override;

  static void               initData

    ( const Properties&       globdat,
      const String&           baseName = "",
      int                     mask     = READ_ALL,
      util::StorageMode       smode    = util::DEFAULT_STORAGE );

  static void               initParser

    ( DataParser&             parser,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat,
      int                     mask     = READ_ALL );

  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare       ();


 protected:

  virtual                  ~InputModule   ();

  virtual void              initData_

    ( const Properties&       globdat,
      const String&           baseName,
      int                     mask,
      util::StorageMode       smode );

  virtual void              initParser_

    ( DataParser&             parser,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat,
      int                     mask  );


 private:

  int                       readMask_;

};


JIVE_END_PACKAGE( fem )

#endif
