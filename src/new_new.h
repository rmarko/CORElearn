#if !defined(NEW_NEW_H)
#define NEW_NEW_H

#include <new>
#include <stddef.h>
#include <malloc.h>

   #define ALLOC  malloc
   #define FREE   free
   void  operator delete(void* ptr);
   void* operator new(size_t size);
#endif



