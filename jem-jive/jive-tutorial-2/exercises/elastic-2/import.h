#ifndef IMPORT_H
#define IMPORT_H

#include <jem/defines.h>

//-----------------------------------------------------------------------
//   forward declarations
//-----------------------------------------------------------------------

namespace jem
{
  class String;
  class Object;

  template <class T> class Ref;

  namespace util
  {
    class Properties;
  }
}

//-----------------------------------------------------------------------
//   using declarations
//-----------------------------------------------------------------------

using jem::idx_t;
using jem::Ref;
using jem::String;
using jem::Object;
using jem::util::Properties;

#endif
