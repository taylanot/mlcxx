#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  // for fork()
#include <iostream>

int main() 
{
  std::cout << "Parent process: My PID is" << getpid() << std::endl;

  while(true)
  {
    sleep(3);
    std::cout << "A" << std::endl;
  }

  return 0;
}
