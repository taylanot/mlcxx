
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

#ifndef JIVE_GEOM_SIMPLEMASKMATRIX_H
#define JIVE_GEOM_SIMPLEMASKMATRIX_H

#include <jem/io/Serializable.h>
#include <jive/geom/import.h>
#include <jive/geom/MaskMatrix.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class SimpleMaskMatrix
//-----------------------------------------------------------------------


class SimpleMaskMatrix : public MaskMatrix,
                         public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( SimpleMaskMatrix, MaskMatrix );


                          SimpleMaskMatrix ();

  explicit                SimpleMaskMatrix

    ( const BoolMatrix&     mask );

  virtual void            readFrom

    ( ObjectInput&          in  )                   override;

  virtual void            writeTo

    ( ObjectOutput&         out )             const override;

  virtual bool            getValue

    ( idx_t                 irow,
      idx_t                 jcol )            const override;

  virtual void            getRowValues

    ( const BoolVector&     values,
      idx_t                 irow,
      const IdxVector&      jcols )           const override;


 protected:

  virtual                ~SimpleMaskMatrix ();


 private:

  BoolMatrix              mask_;

};


JIVE_END_PACKAGE( geom )

#endif
