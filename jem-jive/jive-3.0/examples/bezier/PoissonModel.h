
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

#ifndef JIVE_FEM_MISC_TRANSPORTMODEL_H
#define JIVE_FEM_MISC_TRANSPORTMODEL_H

#include <jive/model/Model.h>

#include "import.h"


//-----------------------------------------------------------------------
//   class TransportModel
//-----------------------------------------------------------------------


class PoissonModel : public Model
{
 public:

  static const char*        TYPE_NAME;

                            PoissonModel  ();

                            PoissonModel

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params,
      const Properties&       globdat );

  static Ref<Model>         makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare       ();

 protected:

  virtual                  ~PoissonModel  ();


 private:

  void                      calcMatrix_

    ( MBuilder&               mb,
      const Vector&           fint,
      const Vector&           state )        const;

  bool                      getTable_

    ( const Properties&       params,
      const Properties&       globdat )      const;

  void                      calcFlux_

    ( XTable&                 table,
      const Vector&           state )        const;


 private:

  Ref<XDofSpace>            dofs_;
  Ref<BezierElement>        element_;
  IdxVector                 dofTypes_;

};


#endif
