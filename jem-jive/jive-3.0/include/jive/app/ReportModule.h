
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

#ifndef JIVE_APP_REPORTMODULE_H
#define JIVE_APP_REPORTMODULE_H

#include <jive/Array.h>
#include <jive/app/Module.h>


JIVE_BEGIN_PACKAGE( app )


//-----------------------------------------------------------------------
//   class ReportModule
//-----------------------------------------------------------------------


class ReportModule : public Module
{
 public:

  JEM_DECLARE_CLASS       ( ReportModule, Module );

  static const char*        TYPE_NAME;


                            ReportModule

    ( const String&           name,
      const Ref<Module>&      child );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat )            override;

  virtual Status            run

    ( const Properties&       globdat )            override;

  virtual void              shutdown

    ( const Properties&       globdat )            override;

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat )            override;

  virtual void              getConfig

    ( const Properties&       conf,
      const Properties&       globdat )      const override;

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params,
      const Properties&       globdat )            override;

  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare       ();


 protected:

  virtual                  ~ReportModule  ();


 private:

  Status                    snoopInit_

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  void                      printConfig_

    ( const Properties&       conf );


 private:

  class                     RunData_;
  class                     Writer_;
  friend class              Writer_;

  Ref<Module>               child_;
  Ref<RunData_>             rundat_;

  StringVector              usedProps_;

};


JIVE_END_PACKAGE( app )

#endif
