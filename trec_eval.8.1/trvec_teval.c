#ifdef RCSID
static char rcsid[] = "$Header: /home/smart/release/src/libevaluate/trvec_trec_eval.c,v 11.0 1992/07/21 18:20:35 chrisb Exp chrisb $";
#endif

/* Copyright (c) 2005
*/

#include "common.h"
#include "sysfunc.h"
#include "smart_error.h"
#include "tr_vec.h"
#include "trec_eval.h"

static int compare_iter_rank();
static void calc_cutoff_measures(EVAL_PARAM_INFO *epi, TR_VEC *tr_vec,
				 TREC_EVAL *eval, long num_rel,
				 long num_nonrel);
static void calc_bpref_measures(EVAL_PARAM_INFO *epi, TR_VEC *tr_vec,
				TREC_EVAL *eval, long num_rel,
				long num_nonrel);
static void calc_average_measures(EVAL_PARAM_INFO *epi, TR_VEC *tr_vec,
				  TREC_EVAL *eval, long num_rel,
				  long num_nonrel);
static void calc_exact_measures(EVAL_PARAM_INFO *epi, TR_VEC *tr_vec,
				TREC_EVAL *eval, long num_rel,
				long num_nonrel);
static void calc_time_measures(EVAL_PARAM_INFO *epi, TR_VEC *tr_vec,
			       TREC_EVAL *eval, long num_rel,
			       long num_nonrel);

int
trvec_trec_eval (epi, tr_vec, eval, num_rel, num_nonrel)
EVAL_PARAM_INFO *epi;
TR_VEC *tr_vec;
TREC_EVAL *eval;
long num_rel;               /* Number relevant judged */
long num_nonrel;            /* Number nonrelevant judged */
{
    long j;
    long max_iter;

    if (tr_vec == (TR_VEC *) NULL)
        return (UNDEF);

    /* Initialize everything to 0 */
    bzero ((char *) eval, sizeof (TREC_EVAL));

    eval->qid = tr_vec->qid;
    eval->num_queries = 1;

    /* If no retrieved docs, then just return */
    if (tr_vec->num_tr == 0) {
        return (0);
    }

    eval->num_rel = num_rel;

    /* Evaluate only the docs on the last iteration of new_tr_vec */
    /* Sort the tr tuples for this query by decreasing iter and 
       increasing rank */
    qsort ((char *) tr_vec->tr,
           (int) tr_vec->num_tr,
           sizeof (TR_TUP),
           compare_iter_rank);

    max_iter = tr_vec->tr[0].iter;
    for (j = 0; j < tr_vec->num_tr; j++) {
        if (tr_vec->tr[j].iter == max_iter) {
            eval->num_ret++;
            if (tr_vec->tr[j].rel >= epi->relevance_level)
                eval->num_rel_ret++;
        }
        else {
            if (tr_vec->tr[j].rel >= epi->relevance_level)
                eval->num_rel--;
        }
    }

    /* Calculate cutoff measures, and those measures dependant on them */
    /* Also includes recip_rank and rank_first_rel */
    calc_cutoff_measures (epi, tr_vec, eval, num_rel, num_nonrel);

    /* Calculate bpref and related measures (judged docs only) */
    calc_bpref_measures (epi, tr_vec, eval, num_rel, num_nonrel);

    /* Calculate measures that average over ret or rel docs */
    calc_average_measures (epi, tr_vec, eval, num_rel, num_nonrel);

    /* Calculate exact measures over entire retrieved sets */
    calc_exact_measures (epi, tr_vec, eval, num_rel, num_nonrel);

    /* Calculate time measures, if wanted */
    if (epi->time_flag)
	calc_time_measures (epi, tr_vec, eval, num_rel, num_nonrel);


    return (1);
}

static int
compare_iter_rank (tr1, tr2)
TR_TUP *tr1;
TR_TUP *tr2;
{
    if (tr1->iter > tr2->iter)
        return (-1);
    if (tr1->iter < tr2->iter)
        return (1);
    if (tr1->rank < tr2->rank)
        return (-1);
    if (tr1->rank > tr2->rank)
        return (1);
    return (0);
}


/* ********************************************************************* */
/* calculate cutoff measures */
 /* cutoff values for recall precision output */
static int cutoff[NUM_CUTOFF] = CUTOFF_VALUES;
static int three_pts[3] = THREE_PTS;


static void
calc_cutoff_measures(epi, tr_vec, eval, num_rel, num_nonrel)
EVAL_PARAM_INFO *epi;
TR_VEC *tr_vec;
TREC_EVAL *eval;
long num_rel;               /* Number relevant judged */
long num_nonrel;            /* Number nonrelevant judged */
{
    double recall, precis;     /* current recall, precision values */
    double rel_precis, rel_uap;/* relative precision, uap values */
    double int_precis;         /* current interpolated precision values */
    
    long i,j;

    long cut_rp[NUM_RP_PTS];   /* number of rel docs needed to be retrieved
                                  for each recall-prec cutoff */
    long cut_fr[NUM_FR_PTS];   /* number of non-rel docs needed to be
                                  retrieved for each fall-recall cutoff */
    long cut_rprec[NUM_PREC_PTS]; /* Number of docs needed to be retrieved
                                    for each R-based prec cutoff */
    long current_cutoff, current_cut_rp, current_cut_fr, current_cut_rprec;

    long rel_so_far = eval->num_rel_ret;

    /* Note for interpolated precision values (Prec(X) = MAX (PREC(Y)) for all
       Y >= X) */
    int_precis = (float) rel_so_far / (float) eval->num_ret;

    /* Discover cutoff values for this query */
    current_cutoff = NUM_CUTOFF - 1;
    while (current_cutoff > 0 && cutoff[current_cutoff] > eval->num_ret)
        current_cutoff--;
    for (i = 0; i < NUM_RP_PTS; i++)
        cut_rp[i] = ((eval->num_rel * i) + NUM_RP_PTS - 2) / (NUM_RP_PTS - 1);
    current_cut_rp = NUM_RP_PTS - 1;
    while (current_cut_rp > 0 && cut_rp[current_cut_rp] > eval->num_rel_ret)
        current_cut_rp--;
    for (i = 0; i < NUM_FR_PTS; i++)
        cut_fr[i] = ((MAX_FALL_RET * i) + NUM_FR_PTS - 2) / (NUM_FR_PTS - 1);
    current_cut_fr = NUM_FR_PTS - 1;
    while (current_cut_fr > 0 && cut_fr[current_cut_fr] > eval->num_ret - eval->num_rel_ret)
        current_cut_fr--;
    for (i = 1; i < NUM_PREC_PTS+1; i++)
        cut_rprec[i-1] = ((MAX_RPREC * eval->num_rel * i) + NUM_PREC_PTS - 2) 
                        / (NUM_PREC_PTS - 1);
    current_cut_rprec = NUM_PREC_PTS - 1;
    while (current_cut_rprec > 0 && cut_rprec[current_cut_rprec]>eval->num_ret)
        current_cut_rprec--;

    /* Loop over all retrieved docs in reverse order */
    for (j = eval->num_ret; j > 0; j--) {
	if (rel_so_far > 0) {
	    recall = (float) rel_so_far / (float) eval->num_rel;
	    precis = (float) rel_so_far / (float) j;
	    if (j > eval->num_rel) {
		rel_precis = (float) rel_so_far / (float) eval->num_rel;
	    }
	    else {
		rel_precis = (float) rel_so_far / (float) j;
	    }
	}
	else {
	    recall = 0.0;
	    precis = 0.0;
	    rel_precis = 0.0;
	}
	rel_uap = rel_precis * rel_precis;
        if (int_precis < precis)
            int_precis = precis;
        while (j == cutoff[current_cutoff]) {
            eval->recall_cut[current_cutoff] = recall;
            eval->precis_cut[current_cutoff] = precis;
            eval->rel_precis_cut[current_cutoff] = rel_precis;
            eval->uap_cut[current_cutoff] = precis * recall;
	    eval->rel_uap_cut[current_cutoff] = rel_uap;
            current_cutoff--;
        }

        while (j == cut_rprec[current_cut_rprec]) {
            eval->R_prec_cut[current_cut_rprec] = precis;
            eval->int_R_prec_cut[current_cut_rprec] = int_precis;
            current_cut_rprec--;
        }

        if (j == eval->num_rel) {
            eval->R_recall_precis = precis;
            eval->int_R_recall_precis = int_precis;
        }

        if (tr_vec->tr[j-1].rel >= epi->relevance_level) {
            while (rel_so_far == cut_rp[current_cut_rp]) {
                eval->int_recall_precis[current_cut_rp] = int_precis;
                current_cut_rp--;
            }
	    eval->recip_rank = 1.0 / (float) j;
	    eval->rank_first_rel = j;
            rel_so_far--;
        }
        else {
            /* Note: for fallout-recall, the recall at X non-rel docs
               is used for the recall 'after' (X-1) non-rel docs.
               Ie. recall_used(X-1 non-rel docs) = MAX (recall(Y)) for 
               Y retrieved docs where X-1 non-rel retrieved */
            while (current_cut_fr >= 0 &&
                   j - rel_so_far == cut_fr[current_cut_fr] + 1) {
                eval->fall_recall[current_cut_fr] = recall;
                current_cut_fr--;
            }
        }
    }

    /* Fill in the 0.0 value for recall-precision (== max precision
       at any point in the retrieval ranking) */
    eval->int_recall_precis[0] = int_precis;

    /* Fill in those cutoff values and averages that were not achieved
       because insufficient docs were retrieved. */
    for (i = 0; i < NUM_CUTOFF; i++) {
        if (eval->num_ret < cutoff[i]) {
	    if (eval->num_rel_ret > 0) {
		eval->recall_cut[i] = ((float) eval->num_rel_ret /
				       (float) eval->num_rel);
		eval->precis_cut[i] = ((float) eval->num_rel_ret / 
				       (float) cutoff[i]);
	    }
            eval->rel_precis_cut[i] = (cutoff[i] < eval->num_rel) ?
                                            eval->precis_cut[i] :
                                            eval->recall_cut[i];
            eval->uap_cut[i] = eval->precis_cut[i] *
		                         eval->recall_cut[i];
            eval->rel_uap_cut[i] = eval->precis_cut[i] *
		                         eval->precis_cut[i];
	}
    }
    for (i = 0; i < NUM_FR_PTS; i++) {
        if (eval->num_ret - eval->num_rel_ret < cut_fr[i]) {
	    if (eval->num_rel_ret > 0)
		eval->fall_recall[i] = (float) eval->num_rel_ret / 
		                       (float) eval->num_rel;
        }
    }
    for (i = 0; i < NUM_PREC_PTS; i++) {
        if (eval->num_ret < cut_rprec[i]) {
            eval->R_prec_cut[i] = (float) eval->num_rel_ret / 
                (float) cut_rprec[i];
            eval->int_R_prec_cut[i] = (float) eval->num_rel_ret / 
                (float) cut_rprec[i];
        }
    }

    if (eval->num_rel > eval->num_ret) {
        eval->R_recall_precis = (float) eval->num_rel_ret / 
                                (float)eval->num_rel;
        eval->int_R_recall_precis = (float) eval->num_rel_ret / 
                                    (float)eval->num_rel;
    }

    /* Calculate other indirect evaluation measure averages. */
    /* average recall-precis of 3 and 11 intermediate points */
    eval->int_av3_recall_precis =
        (eval->int_recall_precis[three_pts[0]] +
         eval->int_recall_precis[three_pts[1]] +
         eval->int_recall_precis[three_pts[2]]) / 3.0;
    for (i = 0; i < NUM_RP_PTS; i++) {
        eval->int_av11_recall_precis += eval->int_recall_precis[i];
    }
    eval->int_av11_recall_precis /= NUM_RP_PTS;

}

static void
calc_bpref_measures (epi, tr_vec, eval, num_rel, num_nonrel)
EVAL_PARAM_INFO *epi;
TR_VEC *tr_vec;
TREC_EVAL *eval;
long num_rel;               /* Number relevant judged */
long num_nonrel;            /* Number nonrelevant judged */
{
    long j;
    long nonrel_ret;
    long nonrel_so_far, rel_so_far, pool_unjudged_so_far;
    long bounded_5R_nonrel_so_far, bounded_10R_nonrel_so_far;
    long pref_top_nonrel_num = PREF_TOP_NONREL_NUM;
    long pref_top_50pRnonrel_num;
    long pref_top_25pRnonrel_num;
    long pref_top_25p2Rnonrel_num;
    long pref_top_10pRnonrel_num;
    long pref_top_Rnonrel_num;
    
    /* Calculate judgement based measures (dependent on only
       judged docs; no assumption of non-relevance if not judged) */
    /* Binary Preference measures; here expressed as all docs with a higher 
       value of rel are to be preferred.  Optimize by keeping track of nonrel
       seen so far */
    pref_top_nonrel_num = PREF_TOP_NONREL_NUM;
    pref_top_50pRnonrel_num = 50 + eval->num_rel;
    pref_top_25pRnonrel_num = 25 + eval->num_rel;
    pref_top_10pRnonrel_num = 10 + eval->num_rel;
    pref_top_Rnonrel_num = eval->num_rel;
    pref_top_25p2Rnonrel_num = 25 + (2 * eval->num_rel);
    nonrel_ret = 0;
    for (j = 0; j < tr_vec->num_tr; j++) {
	if (tr_vec->tr[j].rel >= 0 && tr_vec->tr[j].rel < epi->relevance_level)
	    nonrel_ret++;
    }
    nonrel_so_far = 0;
    rel_so_far = 0;
    pool_unjudged_so_far = 0;
    bounded_5R_nonrel_so_far = 0; 
    bounded_10R_nonrel_so_far = 0; 
    for (j = 0; j < tr_vec->num_tr; j++) {
	if (tr_vec->tr[j].rel == RELVALUE_NONPOOL)
	    /* document not in pool. Skip */
	    continue;
	if (tr_vec->tr[j].rel == RELVALUE_UNJUDGED) {
	    /* document in pool but unjudged. */
	    pool_unjudged_so_far++;
	    continue;
	}

	if (tr_vec->tr[j].rel >= 0 && tr_vec->tr[j].rel < epi->relevance_level) {
	    /* Judged Nonrel document */
	    if (nonrel_so_far < 5 * eval->num_rel) {
		bounded_5R_nonrel_so_far++;
		if (nonrel_so_far < 10 * eval->num_rel) {
		    bounded_10R_nonrel_so_far++;
		}
	    }
	    nonrel_so_far++;
	}
	else {
	    /* Judged Rel doc */
	    rel_so_far++;
	    /* Add fraction of correct preferences. */
	    /* Special case nonrel_so_far == 0 to avoid division by 0 */
	    if (nonrel_so_far > 0) {
		eval->bpref_allnonrel += 1.0 - (((float) nonrel_so_far) /
					       (float) num_nonrel);
		eval->bpref_retnonrel += 1.0 - (((float) nonrel_so_far) /
					       (float) nonrel_ret);
		eval->bpref_retall += 1.0 - (((float) nonrel_so_far) /
					    (float) nonrel_ret);
		eval->bpref_num_correct += 
		    MIN (num_nonrel, pref_top_Rnonrel_num) -
		    MIN (nonrel_so_far, pref_top_Rnonrel_num);
		eval->bpref += 1.0 - 
		    (((float) MIN (nonrel_so_far, pref_top_Rnonrel_num)) /
		     (float) MIN (num_nonrel, pref_top_Rnonrel_num));
		eval->old_bpref += 1.0 - 
		    (((float) MIN (nonrel_so_far, pref_top_Rnonrel_num)) /
		     (float) MIN (nonrel_ret, pref_top_Rnonrel_num));
		eval->bpref_topnonrel += 1.0 - 
		    (((float) MIN (nonrel_so_far, pref_top_nonrel_num)) /
		     (float) MIN (num_nonrel, pref_top_nonrel_num));
		eval->bpref_top50pRnonrel += 1.0 - 
		    (((float) MIN (nonrel_so_far, pref_top_50pRnonrel_num)) /
		     (float) MIN (num_nonrel, pref_top_50pRnonrel_num));
		eval->bpref_top25pRnonrel += 1.0 - 
		    (((float) MIN (nonrel_so_far, pref_top_25pRnonrel_num)) /
		     (float) MIN (num_nonrel, pref_top_25pRnonrel_num));
		eval->bpref_top10pRnonrel += 1.0 - 
		    (((float) MIN (nonrel_so_far, pref_top_10pRnonrel_num)) /
		     (float) MIN (num_nonrel, pref_top_10pRnonrel_num));
		eval->old_bpref_top10pRnonrel += 1.0 - 
		    (((float) MIN (nonrel_so_far, pref_top_10pRnonrel_num)) /
		     (float) MIN (nonrel_ret, pref_top_10pRnonrel_num));
		eval->bpref_top25p2Rnonrel += 1.0 - 
		    (((float) MIN (nonrel_so_far, pref_top_25p2Rnonrel_num)) /
		     (float) MIN (num_nonrel, pref_top_25p2Rnonrel_num));
		if (rel_so_far <= 5 && nonrel_so_far < 5)
		    eval->bpref_5 += 1.0 - (float) nonrel_so_far /
			(float) MIN (num_nonrel, 5);
		if (rel_so_far <= 10 && nonrel_so_far < 10)
		    eval->bpref_10 += 1.0 - (float) nonrel_so_far /
			(float) MIN (num_nonrel, 10);
	    }
	    else {
		eval->bpref += 1.0;
		eval->old_bpref += 1.0;
		eval->bpref_allnonrel += 1.0;
		eval->bpref_retnonrel += 1.0;
		eval->bpref_retall += 1.0;
		eval->bpref_topnonrel += 1.0;
		eval->bpref_top50pRnonrel += 1.0;
		eval->bpref_top25pRnonrel += 1.0;
		eval->bpref_top10pRnonrel += 1.0;
		eval->old_bpref_top10pRnonrel += 1.0;
		eval->bpref_top25p2Rnonrel += 1.0;
		if (rel_so_far <= 5)
		    eval->bpref_5 += 1.0;
		if (rel_so_far <= 10)
		    eval->bpref_10 += 1.0;

	    }
	    eval->bpref_top5Rnonrel += 1.0 -
		(((float) bounded_5R_nonrel_so_far) /
		 (float) MIN (num_nonrel, eval->num_rel * 5));
	    eval->bpref_top10Rnonrel += 1.0 -
		(((float) bounded_10R_nonrel_so_far) /
		 (float) MIN (num_nonrel, eval->num_rel * 10));
	    eval->bpref_num_all += num_nonrel - nonrel_so_far;
	    eval->bpref_num_ret += nonrel_ret - nonrel_so_far;
	    /* inf_ap */
	    if (0 == j)
		eval->inf_ap += 1.0;
	    else {
		float fj = (float) j;
		eval->inf_ap += 1.0 / (fj+1.0) +
		    (fj / (fj+1.0)) *
		    ((rel_so_far-1+nonrel_so_far+pool_unjudged_so_far) / fj)  *
		    ((rel_so_far-1 + INFAP_EPSILON) / 
		     (rel_so_far-1 + nonrel_so_far + 2 * INFAP_EPSILON));
	    }
	}
    }
    if (eval->num_rel) {
	eval->bpref /= eval->num_rel;
	eval->old_bpref /= eval->num_rel;
	eval->bpref_allnonrel /= eval->num_rel;
	eval->bpref_retnonrel /= eval->num_rel;
	eval->bpref_topnonrel /= eval->num_rel;
	eval->bpref_top5Rnonrel /= eval->num_rel;
	eval->bpref_top10Rnonrel /= eval->num_rel;
	eval->bpref_top50pRnonrel /= eval->num_rel;
	eval->bpref_top25pRnonrel /= eval->num_rel;
	eval->bpref_top10pRnonrel /= eval->num_rel;
	eval->old_bpref_top10pRnonrel /= eval->num_rel;
	eval->bpref_top25p2Rnonrel /= eval->num_rel;
	if (eval->num_rel_ret) {
	    eval->bpref_retall /= eval->num_rel_ret;
	    eval->bpref_5 /= MIN (rel_so_far, 5);
	    eval->bpref_10 /= MIN (rel_so_far, 10);
	}
	eval->bpref_num_possible = eval->num_rel *
	    MIN (num_nonrel, pref_top_Rnonrel_num);
	eval->inf_ap /= eval->num_rel;
    }
    eval->num_nonrel_judged_ret = nonrel_ret;

    /* For those bpref measure variants which use the geometric mean instead
       of straight averages, compute them here.  Original measure value
       is constrained to be greater than MIN_GEO_MEAN (for time being .00001,
       since trec_eval prints to four significant digits) */
    eval->gm_bpref = (float) log ((double)(MAX (eval->bpref,
						MIN_GEO_MEAN)));
}

static void
calc_average_measures (epi, tr_vec, eval, num_rel, num_nonrel)
EVAL_PARAM_INFO *epi;
TR_VEC *tr_vec;
TREC_EVAL *eval;
long num_rel;               /* Number relevant judged */
long num_nonrel;            /* Number nonrelevant judged */
{
    double recall, precis;     /* current recall, precision values */
    double rel_precis, rel_uap;/* relative precision, uap values */
    double int_precis;         /* current interpolated precision values */
    
    long i,j;
    long rel_so_far;

    /* Note for interpolated precision values (Prec(X) = MAX (PREC(Y)) for all
       Y >= X) */
    rel_so_far = eval->num_rel_ret;
    int_precis = (float) rel_so_far / (float) eval->num_ret;

    /* Loop over all retrieved docs in reverse order */
    for (j = eval->num_ret; j > 0; j--) {
	if (rel_so_far > 0) {
	    recall = (float) rel_so_far / (float) eval->num_rel;
	    precis = (float) rel_so_far / (float) j;
	    if (j > eval->num_rel) {
		rel_precis = (float) rel_so_far / (float) eval->num_rel;
	    }
	    else {
		rel_precis = (float) rel_so_far / (float) j;
	    }
	}
	else {
	    recall = 0.0;
	    precis = 0.0;
	    rel_precis = 0.0;
	}
	rel_uap = rel_precis * rel_precis;
        if (int_precis < precis)
            int_precis = precis;
	eval->av_rel_precis += rel_precis;
	eval->av_rel_uap += rel_uap;

        if (j < eval->num_rel) {
            eval->av_R_precis += precis;
            eval->int_av_R_precis += int_precis;
        }

        if (tr_vec->tr[j-1].rel >= epi->relevance_level) {
            eval->int_av_recall_precis += int_precis;
            eval->av_recall_precis += precis;
            eval->avg_doc_prec += precis;
            rel_so_far--;
        }
        else {
            /* Note: for fallout-recall, the recall at X non-rel docs
               is used for the recall 'after' (X-1) non-rel docs.
               Ie. recall_used(X-1 non-rel docs) = MAX (recall(Y)) for 
               Y retrieved docs where X-1 non-rel retrieved */
            if (j - rel_so_far < MAX_FALL_RET) {
                eval->av_fall_recall += recall;
            }
        }
    }

    if (eval->num_ret - eval->num_rel_ret < MAX_FALL_RET) {
	if (eval->num_rel_ret > 0)
	    eval->av_fall_recall += ((MAX_FALL_RET - 
				      (eval->num_ret - eval->num_rel_ret))
				     * ((float)eval->num_rel_ret / 
					(float)eval->num_rel));
    }
    if (eval->num_rel > eval->num_ret) {
        for (i = eval->num_ret; i < eval->num_rel; i++) {
            eval->av_R_precis += (float) eval->num_rel_ret / 
                                 (float) i;
            eval->int_av_R_precis += (float) eval->num_rel_ret / 
                                     (float) i;
        }
    }

    /* Calculate all the other averages */
    if (eval->num_rel_ret > 0) {
        eval->av_recall_precis /= eval->num_rel;
        eval->int_av_recall_precis /= eval->num_rel;
    }

    eval->av_fall_recall /= MAX_FALL_RET;

    eval->av_rel_precis /= eval->num_ret;
    eval->av_rel_uap /= eval->num_ret;

    if (eval->num_rel) {
        eval->av_R_precis /= eval->num_rel;
        eval->int_av_R_precis /= eval->num_rel;
    }

    /* For those measure variants which use the geometric mean instead
       of straight averages, compute them here.  Original measure value
       is constrained to be greater than MIN_GEO_MEAN (for time being .00001,
       since trec_eval prints to four significant digits) */
    eval->gm_ap = (float) log ((double)(MAX (eval->av_recall_precis,
					     MIN_GEO_MEAN)));
}

static void
calc_exact_measures (epi, tr_vec, eval, num_rel, num_nonrel)
EVAL_PARAM_INFO *epi;
TR_VEC *tr_vec;
TREC_EVAL *eval;
long num_rel;               /* Number relevant judged */
long num_nonrel;            /* Number nonrelevant judged */
{

    if (eval->num_rel) {
        eval->exact_recall = (double) eval->num_rel_ret / eval->num_rel;
        eval->exact_precis = (double) eval->num_rel_ret / eval->num_ret;
	eval->exact_uap = eval->exact_recall * eval->exact_precis;
        if (eval->num_rel > eval->num_ret) {
            eval->exact_rel_precis = eval->exact_precis;
	}
        else {
            eval->exact_rel_precis = eval->exact_recall;
	}
	eval->exact_rel_uap = eval->exact_precis * eval->exact_precis;
	eval->exact_utility =
		epi->utility_a * eval->num_rel_ret +
		epi->utility_b * (eval->num_ret - eval->num_rel_ret) +
		epi->utility_c * (eval->num_rel - eval->num_rel_ret) +
		epi->utility_d * (epi->num_docs_in_coll + eval->num_rel_ret
				 - eval->num_ret - eval->num_rel);
    }
}

static void
calc_time_measures (epi, tr_vec, eval, num_rel, num_nonrel)
EVAL_PARAM_INFO *epi;
TR_VEC *tr_vec;
TREC_EVAL *eval;
long num_rel;               /* Number relevant judged */
long num_nonrel;            /* Number nonrelevant judged */
{
    double recall, precis;     /* current recall, precision values */
    double rel_precis, rel_uap;/* relative precision, uap values */
    double int_precis = 0.0;   /* current interpolated precision values */
    
    long i,j;

    long bucket;
    long last_time_bucket = NUM_TIME_PTS;  /* Last time bucket filled in */

    long rel_so_far = eval->num_rel_ret;
    long min_ret_rel = MIN(eval->num_rel, eval->num_ret);

    /* Loop over all retrieved docs in reverse order */
    for (j = eval->num_ret; j > 0; j--) {
	if (rel_so_far > 0) {
	    recall = (float) rel_so_far / (float) eval->num_rel;
	    precis = (float) rel_so_far / (float) j;
	    if (j > eval->num_rel) {
		rel_precis = (float) rel_so_far / (float) eval->num_rel;
	    }
	    else {
		rel_precis = (float) rel_so_far / (float) j;
	    }
	}
	else {
	    recall = 0.0;
	    precis = 0.0;
	    rel_precis = 0.0;
	}
	rel_uap = rel_precis * rel_precis;
        if (int_precis < precis)
            int_precis = precis;

	bucket = tr_vec->tr[j-1].sim *
	    ((double) NUM_TIME_PTS / (double) MAX_TIME);
	if (bucket < 0) bucket = 0;
	if (bucket >= NUM_TIME_PTS) bucket = NUM_TIME_PTS-1;
	if (tr_vec->tr[j-1].rel >= epi->relevance_level)
	    eval->time_num_rel[bucket]++;
	else
	    eval->time_num_nrel[bucket]++;
	eval->time_precis[bucket] = (float)rel_so_far /
	    (float) eval->num_ret;
	eval->time_relprecis[bucket] = ((float)rel_so_far) / 
	    (float) min_ret_rel;
	eval->time_uap[bucket] = (float) rel_so_far * rel_so_far /
	    ((float) eval->num_ret * (float) min_ret_rel);
	eval->time_reluap[bucket] = (float) rel_so_far * rel_so_far /
	    ((float) min_ret_rel * (float) min_ret_rel);
	eval->time_utility[bucket] = 
	    epi->utility_a * rel_so_far +
	    epi->utility_b * (j - rel_so_far) +
	    epi->utility_c * (eval->num_rel - rel_so_far) +
	    epi->utility_d * (epi->num_docs_in_coll +
			      rel_so_far - j - eval->num_rel);
	
	/* Need to fill in buckets up to last bucket */
	/* note assumes buckets are decreasing */
	/* Must do here since utility can be negative and zero
	   cannot be used as flag later */
	for (i = bucket+1; i < last_time_bucket; i++) {
	    eval->time_precis[i] = eval->time_precis[bucket];
	    eval->time_relprecis[i] = eval->time_relprecis[bucket];
	    eval->time_uap[i] = eval->time_uap[bucket];
	    eval->time_reluap[i] = eval->time_reluap[bucket];
	    eval->time_utility[i] = eval->time_utility[bucket];
	}
	last_time_bucket = bucket;
    }

    eval->time_cum_rel[0] = eval->time_num_rel[0];
    eval->av_time_cum_rel = eval->time_num_rel[0];
    for (i=1; i< NUM_TIME_PTS; i++) {
	eval->time_cum_rel[i] = eval->time_cum_rel[i-1] + eval->time_num_rel[i];
	eval->av_time_cum_rel += eval->time_cum_rel[i];
	eval->av_time_precis += eval->time_precis[i];
	eval->av_time_relprecis += eval->time_relprecis[i];
	eval->av_time_uap += eval->time_uap[i];
	eval->av_time_reluap += eval->time_reluap[i];
	eval->av_time_utility += eval->time_utility[i];
    }
    eval->av_time_cum_rel /= NUM_TIME_PTS;
    eval->av_time_precis /= NUM_TIME_PTS;
    eval->av_time_relprecis /= NUM_TIME_PTS;
    eval->av_time_uap /= NUM_TIME_PTS;
    eval->av_time_reluap /= NUM_TIME_PTS;
    eval->av_time_utility /= NUM_TIME_PTS;
}
