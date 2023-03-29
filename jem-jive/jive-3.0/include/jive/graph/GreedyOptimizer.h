
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

#ifndef JIVE_GRAPH_GREEDYOPTIMIZER_H
#define JIVE_GRAPH_GREEDYOPTIMIZER_H

#include <jive/graph/import.h>
#include <jive/graph/Optimizer.h>


JIVE_BEGIN_PACKAGE( graph )


class BisectionBorder;
class PartitionBorder;


//-----------------------------------------------------------------------
//   class Greedyoptimizer
//-----------------------------------------------------------------------


class GreedyOptimizer : public Optimizer
{
 public:

  JEM_DECLARE_CLASS       ( GreedyOptimizer, Optimizer );

  static const char*        TYPE_NAME;
  static const int          MAX_PASSES;


  explicit                  GreedyOptimizer

    ( const String&           name = "",
      const Ref<Random>&      rand = nullptr );

  virtual void              configure

    ( const Properties&       props )                override;

  virtual void              getConfig

    ( const Properties&       props )          const override;

  virtual void              optimize

    ( Partition&              part,
      const Control&          ctrl )                 override;

  static void               optimize

    ( BisectionBorder&        border,
      Random&                 rand,
      const Control&          ctrl,
      int                     maxPasses = MAX_PASSES );

  static void               optimize

    ( PartitionBorder&        border,
      Random&                 rand,
      const Control&          ctrl,
      int                     maxPasses = MAX_PASSES );

  static Ref<Optimizer>     makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare         ();


 protected:

  virtual                  ~GreedyOptimizer ();


 private:

  Ref<Random>               rand_;
  int                       maxPasses_;

};


JIVE_END_PACKAGE( graph )

#endif
