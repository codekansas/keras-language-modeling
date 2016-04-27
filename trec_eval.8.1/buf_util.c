#ifdef RCSID
static char rcsid[] = "$Header: /home/smart/release/src/libgeneral/buf_util.c,v 11.0 1992/07/21 18:21:04 chrisb Exp $";
#endif

/* Copyright (c) 1991, 1990, 1984 - Gerard Salton, Chris Buckley. 

   Permission is granted for use of this file in unmodified form for
   research purposes. Please contact the SMART project to obtain 
   permission for other uses.
*/

/********************   PROCEDURE DESCRIPTION   ************************
 *0 Utility procedure to add the memory contents of new.buf to result.buf
 *2 add_buf (new, result)
 *3  SM_BUF *new;
 *3  SM_BUF *result;
 *7 Both new and result are of type
 *7 typedef struct {
 *7     int size;                      * allocated space for buf *
 *7     int end;                       * end of valid data in buf *
 *7     char *buf;                     * buffer of arbitrary data *
 *7 } SM_BUF;
 *7     
 *7 Append the data in new to the end of the data in result.  The data can
 *7 be arbitrary data, eg, include '\0's.
 *7 Return UNDEF if can't allocate enough space for the result, 0 otherwise.
***********************************************************************/

#include "common.h"
#include "sysfunc.h"
#include "buf.h"

int 
add_buf (new, result)
SM_BUF *new, *result;
{
    if (result->size == 0) {
        if (NULL == (result->buf = malloc ((unsigned) new->end * 2 + 1)))
            return (UNDEF);
        result->size = 2 * new->end + 1;
        result->end = 0;
    }
    else if (new->end >= result->size - result->end) {
        if (NULL == (result->buf =
                     realloc (result->buf,
                              (unsigned) result->size * 2 + new->end)))
            return (UNDEF);
        result->size += result->size + new->end;
    }

    bcopy (new->buf, &result->buf[result->end], new->end);
    result->end += new->end;
    
    return (0);
}

/********************   PROCEDURE DESCRIPTION   ************************
 *0 Utility procedure to add the string new to result.buf
 *2 add_buf_string (new, result)
 *3  char *new;
 *3  SM_BUF *result;

 *7 Result is of type
 *7 typedef struct {
 *7     int size;                      * allocated space for buf *
 *7     int end;                       * end of valid data in buf *
 *7     char *buf;                     * buffer of arbitrary data *
 *7 } SM_BUF;
 *7     
 *7 Append the data in new to the end of the data in result.
 *7 Return UNDEF if can't allocate enough space for the result, 0 otherwise.
***********************************************************************/

int 
add_buf_string (new, result)
char *new;
SM_BUF *result;
{
    SM_BUF temp_buf;
    temp_buf.end = strlen (new);
    temp_buf.buf = new;

    return (add_buf (&temp_buf, result));

}

