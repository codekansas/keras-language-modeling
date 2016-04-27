#include "common.h"
#include "sysfunc.h"
#include "smart_error.h"
#include "tr_vec.h"
#include "trec_eval.h"
#include "buf.h"

static int comp_tr_tup_rank(), comp_tr_tup_did(), comp_tr_docno(), 
    comp_qrels_docno(), comp_sim_docno(), comp_negsim_docno();

/* Space reserved for output TR_TUP tuples */
static TR_TUP *start_tr_tup;
static long max_tr_tup = 0;

/* Takes the top docs and pool docs for a query, and returns a
   tr_vec object giving the relevance valuse for all top docs.
   Relevance value is
       value in trec_qrels if docno is in trec_qrels and was judged
       RELVALUE_NONPOOL (-1) if docno is not in trec_qrels
       RELVALUE_UNJUDGED (-2) if docno is in trec_qrels and was not judged
*/

int
form_trvec (epi, trec_top, trec_qrels, tr_vec, num_rel)
EVAL_PARAM_INFO *epi;
TREC_TOP *trec_top;
TREC_QRELS *trec_qrels;
TR_VEC *tr_vec;
long *num_rel;
{
    TR_TUP *tr_tup;
    TEXT_QRELS *qrels_ptr, *end_qrels;
    long i;

    /* Reserve space for output tr_tups, if needed */
    if (trec_top->num_text_tr > max_tr_tup) {
        if (max_tr_tup > 0) 
            (void) free ((char *) start_tr_tup);
        max_tr_tup += trec_top->num_text_tr;
        if (NULL == (start_tr_tup = Malloc (max_tr_tup, TR_TUP)))
            return (UNDEF);
    }

    /* Sort trec_top by sim, breaking ties lexicographically using docno */
    if (epi->time_flag) {
	qsort ((char *) trec_top->text_tr,
	       (int) trec_top->num_text_tr,
	       sizeof (TEXT_TR),
	       comp_negsim_docno);
    }
    else {
	qsort ((char *) trec_top->text_tr,
	       (int) trec_top->num_text_tr,
	       sizeof (TEXT_TR),
	       comp_sim_docno);
    }

    /* Add ranks to trec_top (starting at 1) */
    for (i = 0; i < trec_top->num_text_tr; i++) {
        trec_top->text_tr[i].rank = i+1;
    }

    /* Sort trec_top lexicographically */
    qsort ((char *) trec_top->text_tr,
           (int) trec_top->num_text_tr,
           sizeof (TEXT_TR),
           comp_tr_docno);

    for (i = 1; i < trec_top->num_text_tr; i++) {
	if (0 == strcmp (trec_top->text_tr[i].docno,
			 trec_top->text_tr[i-1].docno)) {
	    set_error (SM_ILLPA_ERR, "Duplicate top docs docno", "trec_eval");
	    return (UNDEF);
	}
    }

    /* Sort trec_qrels lexicographically */
    qsort ((char *) trec_qrels->text_qrels,
           (int) trec_qrels->num_text_qrels,
           sizeof (TEXT_QRELS),
           comp_qrels_docno);

    /* Find number of relevant docs, and check for duplicates */
    *num_rel = 0;
    for (i = 0; i < trec_qrels->num_text_qrels; i++) {
	if (i > 0 && (0 == strcmp (trec_qrels->text_qrels[i].docno,
				   trec_qrels->text_qrels[i-1].docno))) {
	    set_error (SM_ILLPA_ERR, "Duplicate qrels docno", "trec_eval");
	    return (UNDEF);
	}
	if (trec_qrels->text_qrels[i].rel >= epi->relevance_level)
	    (*num_rel)++;
    }

    /* Go through trec_top, trec_qrels in parallel to determine which
       docno's are in both (ie, which trec_top are relevant).  Once relevance
       is known, convert trec_top tuple into TR_TUP. */
    tr_tup = start_tr_tup;
    qrels_ptr = trec_qrels->text_qrels;
    end_qrels = &trec_qrels->text_qrels[trec_qrels->num_text_qrels];
    for (i = 0; i < trec_top->num_text_tr; i++) {
	if (trec_top->text_tr[i].rank > epi->max_num_docs_per_topic)
	    /* Skip if evaluation desired over fewer docs than this rank */
	    continue;
        while (qrels_ptr < end_qrels &&
               strcmp (qrels_ptr->docno, trec_top->text_tr[i].docno) < 0)
            qrels_ptr++;
        if (qrels_ptr >= end_qrels ||
            strcmp (qrels_ptr->docno, trec_top->text_tr[i].docno) > 0) {
            /* Doc is non-judged */
            tr_tup->rel = RELVALUE_NONPOOL;
	    /* Skip unjudged docs if desired */
	    if (epi->judged_docs_only_flag) 
		continue;
	}
        else {
            /* Doc is in pool, assign relevance */
	    if (qrels_ptr->rel == -1)
		/* In pool, but unjudged (eg, infAP uses a sample of pool) */
		tr_tup->rel = RELVALUE_UNJUDGED;
	    else
		tr_tup->rel = qrels_ptr->rel;
	    qrels_ptr++;
        }
        tr_tup->did = i;
        tr_tup->rank = trec_top->text_tr[i].rank;
        tr_tup->sim = trec_top->text_tr[i].sim;
        tr_tup->action = 0;
        tr_tup->iter = 0;
        tr_tup++;
    }

    /* Form the full TR_VEC object for this qid */
    tr_vec->qid = trec_top->qid;
    tr_vec->num_tr = tr_tup - start_tr_tup;
    tr_vec->tr = start_tr_tup;

    /* If judged_docs_only_flag, then must fix up ranks to reflect unjudged
       docs being thrown out. Note: done this way to preserve original
       tie-breaking based on text docno */
    if (epi->judged_docs_only_flag) {
	/* Sort tuples by increasing rank */
	qsort ((char *) tr_vec->tr,
	       (int) tr_vec->num_tr,
	       sizeof (TR_TUP),
	       comp_tr_tup_rank);
	for (i = 0; i < tr_vec->num_tr; i++) {
	    tr_vec->tr[i].rank = i+1;
	}
	qsort ((char *) tr_vec->tr,
	       (int) tr_vec->num_tr,
	       sizeof (TR_TUP),
	       comp_tr_tup_did);
    }

    return (1);
}

static int 
comp_sim_docno (ptr1, ptr2)
TEXT_TR *ptr1;
TEXT_TR *ptr2;
{
    if (ptr1->sim > ptr2->sim)
        return (-1);
    if (ptr1->sim < ptr2->sim)
        return (1);
    return (strcmp (ptr2->docno, ptr1->docno));
}

static int 
comp_negsim_docno (ptr1, ptr2)
TEXT_TR *ptr1;
TEXT_TR *ptr2;
{
    if (ptr1->sim < ptr2->sim)
        return (-1);
    if (ptr1->sim > ptr2->sim)
        return (1);
    return (strcmp (ptr2->docno, ptr1->docno));
}

static int 
comp_tr_docno (ptr1, ptr2)
TEXT_TR *ptr1;
TEXT_TR *ptr2;
{
    return (strcmp (ptr1->docno, ptr2->docno));
}

static int 
comp_qrels_docno (ptr1, ptr2)
TEXT_QRELS *ptr1;
TEXT_QRELS *ptr2;
{
    return (strcmp (ptr1->docno, ptr2->docno));
}

static int 
comp_tr_tup_rank (ptr1, ptr2)
TR_TUP *ptr1;
TR_TUP *ptr2;
{
    return (ptr1->rank - ptr2->rank);
}

static int 
comp_tr_tup_did (ptr1, ptr2)
TR_TUP *ptr1;
TR_TUP *ptr2;
{
    return (ptr1->did - ptr2->did);
}
