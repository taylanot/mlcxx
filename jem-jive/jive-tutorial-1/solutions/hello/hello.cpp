
#include <jem/base/String.h>
#include <jem/base/System.h>

using jem::System;
using jem::String;

int run ()
{
  String  text = "Hello World\n";

  for ( int i = 0; i < 10; i++ )
  {
    System::out() << text;
  }

  return 0;
}

int main ()
{
  return System::exec ( run );
}
