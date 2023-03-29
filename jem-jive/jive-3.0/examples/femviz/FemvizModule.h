
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

#ifndef FEMVIZ_MODULE_H
#define FEMVIZ_MODULE_H

#include <jive/app/Module.h>


using namespace jem;

using jem::io::PrintWriter;
using jem::util::Properties;
using jive::app::Module;


//-----------------------------------------------------------------------
//   class FemvizModule
//-----------------------------------------------------------------------


class FemvizModule : public Module
{
 public:

  typedef FemvizModule      Self;
  typedef Module            Super;


  explicit                  FemvizModule

    ( const String&           name = "" );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual Status            run

    ( const Properties&       globdat );

  virtual void              shutdown

    ( const Properties&       globdat );

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat );

  virtual void              getConfig

    ( const Properties&       props,
      const Properties&       globdat )        const;

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params,
      const Properties&       globdat );

  void                      printHelp

    ( PrintWriter&            out )            const;


 protected:

  virtual                  ~FemvizModule    ();


 private:

  bool                      exeCommand_

    ( const String&           cmd,
      const String&           arg,
      const Properties&       globdat );

  void                      animate_

    ( const String&           arg,
      const Properties&       globdat );

  bool                      sleep_

    ( double                  time );


 private:

  enum                      State_
  {
                              INACTIVE_,
                              STARTING_,
                              RUNNING_
  };


  Ref<Module>               child_;
  int                       state_;

};


#endif
