
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

#ifndef JIVE_APP_INITMODULE_H
#define JIVE_APP_INITMODULE_H

#include <jive/Array.h>
#include <jive/app/Module.h>


JIVE_BEGIN_PACKAGE( app )


//-----------------------------------------------------------------------
//   class InitModule
//-----------------------------------------------------------------------


class InitModule : public Module
{
 public:

  JEM_DECLARE_CLASS       ( InitModule, Module );


  explicit                  InitModule

    ( const String&           name = "init" );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat )          override;

  virtual void              shutdown

    ( const Properties&       globdat )          override;

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat )          override;

  virtual void              getConfig

    ( const Properties&       conf,
      const Properties&       globdat )    const override;

  static void               makeModel

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat,
      const Ref<Writer>&      logger = nullptr );

  static void               syncDofs

    ( const Properties&       globdat );

  static void               initConstraints

    ( const Properties&       globdat,
      const String&           context );

  static void               initVectors

    ( const StringVector&     specs,
      const Properties&       globdat,
      const String&           context,
      const Ref<Writer>&      logger = nullptr );

  static void               checkVecSpecs

    ( const StringVector&     specs );

  static void               initModel

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );


  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare     ();


 protected:

  virtual                  ~InitModule  ();

  virtual void              makeModel_

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              reorderDofs_

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              syncDofs_

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              initConstraints_

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              initVectors_

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              initModel_

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );


 private:

  static void               parseVecSpec_

    ( String&                 itemPath,
      String&                 vecName,
      String&                 tabName,
      const String&           vecSpec );

};


JIVE_END_PACKAGE( app )

#endif
