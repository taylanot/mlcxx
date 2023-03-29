
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

#ifndef JIVE_IMPLICT_SOLVERBOUNDS_H
#define JIVE_IMPLICT_SOLVERBOUNDS_H

#include <jem/base/Collectable.h>
#include <jem/util/Flex.h>
#include <jive/Array.h>
#include <jive/implict/import.h>


JIVE_BEGIN_PACKAGE( implict )


using jem::Collectable;


//-----------------------------------------------------------------------
//   class SolverBounds
//-----------------------------------------------------------------------


class SolverBounds : public Collectable
{
 public:

  typedef SolverBounds    Self;
  typedef Collectable     Super;


  explicit                SolverBounds

    ( const String&         context );

  void                    configure

    ( const Properties&     props );

  void                    getConfig

    ( const Properties&     conf )         const;

  bool                    hasBounds     () const noexcept;

  bool                    getBounds

    ( const Vector&         lb,
      const Vector&         ub,
      const DofSpace&       dofs )         const;


 protected:

  virtual                ~SolverBounds  ();


 private:

  static const int        LOWER_BOUND_;
  static const int        UPPER_BOUND_;

  struct                  Bnd_
  {
    String                  name;
    String                  dofType;
    double                  lowerBound;
    double                  upperBound;
    int                     status;
  };

  String                  context_;
  jem::util::Flex<Bnd_>   bounds_;
  int                     status_;

};


JIVE_END_PACKAGE( implict )

#endif
