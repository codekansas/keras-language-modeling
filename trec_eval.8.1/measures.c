#ifdef RCSID
static char rcsid[] = "$Header: /home/smart/release/src/libevaluate/tr_eval.c,v 11.0 1992/07/21 18:20:33 chrisb Exp chrisb $";
#endif

/* Copyright (c) 1991, 1990, 1984 - Gerard Salton, Chris Buckley. 

   Permission is granted for use of this file in unmodified form for
   research purposes. Please contact the SMART project to obtain 
   permission for other uses.
*/

#include "common.h"
#include "sysfunc.h"
#include "buf.h"
#include "trec_eval.h"

static long cutoff[] = CUTOFF_VALUES;
static char param_val[20];
static char *get_param_str_ircl_prn(epi, index)
EVAL_PARAM_INFO *epi;
long index;
{
    sprintf (param_val, "%4.2f", (float) index / (NUM_RP_PTS -1));
    return (param_val);
}
static char *get_param_str_cutoff(epi, index)
EVAL_PARAM_INFO *epi;
long index;
{
    sprintf (param_val, "%ld", cutoff[index]);
    return (param_val);
}
static char *get_param_str_Rcutoff(epi, index)
EVAL_PARAM_INFO *epi;
long index;
{
    sprintf (param_val, "%4.2f",
	     (float) MAX_RPREC * (index+1) /(float) (NUM_PREC_PTS - 1));
    return (param_val);
}
static char *get_param_str_utility(epi, index)
EVAL_PARAM_INFO *epi;
long index;
{
    sprintf (param_val, "%3.1f_%3.1f_%3.1f_%3.1f",
	     epi->utility_a, epi->utility_b, epi->utility_c, epi->utility_d);
    return (param_val);
}
static char *get_param_str_maxfallout(epi, index)
EVAL_PARAM_INFO *epi;
long index;
{
    sprintf (param_val, "%ld", (long) MAX_FALL_RET);
    return (param_val);
}
static char *get_param_str_fall_recall(epi, index)
EVAL_PARAM_INFO *epi;
long index;
{
    sprintf (param_val, "%ld",
	     (long) (MAX_FALL_RET * index) / (NUM_FR_PTS - 1));
    return (param_val);
}
static char *get_param_str_time_cutoff(epi, index)
EVAL_PARAM_INFO *epi;
long index;
{
    sprintf (param_val, "%ld",
	     (long) (index * MAX_TIME / NUM_TIME_PTS));
    return (param_val);
}
static char *get_param_str_time_utility_cutoff(epi, index)
EVAL_PARAM_INFO *epi;
long index;
{
    sprintf (param_val, "%3.1f_%3.1f_%3.1f_%3.1f-%ld",
	     epi->utility_a, epi->utility_b, epi->utility_c, epi->utility_d,
	     (long) (index * MAX_TIME / NUM_TIME_PTS));
    return (param_val);
}

SINGLE_MEASURE sing_meas[] = {
    {"num_ret", "Total number of documents retrieved over all queries",
     1, 1, 0, 0, 0, 0, 0, 0, offsetof(TREC_EVAL, num_ret)},
    {"num_rel", "Total number of relevant documents over all queries",
     1, 1, 0, 0, 0, 0, 0, 0, offsetof(TREC_EVAL, num_rel)},
    {"num_rel_ret", "Total number of relevant documents retrieved over all queries",
     1, 1, 0, 0, 0, 0, 0, 0, offsetof(TREC_EVAL, num_rel_ret)},
    {"map", "Mean Average Precision (MAP)",
     0, 1, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, av_recall_precis)},
    {"gm_ap","Average Precision. Geometric Mean, q_score=log(MAX(map,.00001))",
     0, 1, 0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, gm_ap)},
    {"R-prec", "R-Precision (Precision after R (= num-rel for topic) documents retrieved)",
     0, 1, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, R_recall_precis)},
    {"bpref", "Binary Preference, top R judged nonrel",
     0, 1, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref)},
    {"recip_rank", "Reciprical rank of top relevant document",
     0, 1, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, recip_rank)},
    /* end of short output measures (the major ones) */

    {"num_nonrel_judged_ret", "Total number of judged non-relevant documents retrieved over all queries",
     1, 0, 0, 0, 0, 0, 0, 0, offsetof(TREC_EVAL, num_nonrel_judged_ret)},
    {"exact_prec", "Exact Precision over retrieved set",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, exact_precis)},
    {"exact_recall", "Exact Recall over retrieved set",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, exact_recall)},
    {"11-pt_avg", "Average over all 11 points of recall-precision graph",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, int_av11_recall_precis)},
    {"3-pt_avg", "Average over 3 points of recall-precision graph",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, int_av3_recall_precis)},
    {"avg_doc_prec", "Rel doc precision averaged over all relevant docs (NOT over topics)",
     0, 0, 0, 0, 0, 0, 1, 0, offsetof(TREC_EVAL, avg_doc_prec)},
    {"exact_relative_prec", "Exact relative precision",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, exact_rel_precis)},
    {"avg_relative_prec", "Average relative precision",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, av_rel_precis)},
    {"exact_unranked_avg_prec", "Exact Unranked Average Precision",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, exact_uap)},
    {"exact_relative_unranked_avg_prec", "Exact Relative Unranked Average Precision",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, exact_rel_uap)},
    {"map_at_R", "Average Precision over first R docs retrieved",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, av_R_precis)},
    {"int_map", "Interpolated Mean Average Precision",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, int_av_recall_precis)},
    {"exact_int_R_rcl_prec", "Exact R-based-interpolated-Precision",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, int_R_recall_precis)},
    {"int_map_at_R", "Average Interpolated Precision for first R docs retrieved",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, int_av_R_precis)},
    {"time_integral_prec", "Time: Average Integral Precision",
     0, 0, 1, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, av_time_precis)},
    {"time_integral_relative_prec", "Time: Average Integral Relative Precision",
     0, 0, 1, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, av_time_relprecis)},
    {"time_integral_uap", "Time: Average Integral Unranked Precision",
     0, 0, 1, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, av_time_uap)},
    {"time_integral_relative_uap", "Time: Average Integral Unranked Relative Precision",
     0, 0, 1, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, av_time_reluap)},
    {"time_integral_cum_rel", "Time: Average (Integral) cumulative number relevant",
     0, 0, 1, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, av_time_cum_rel)},
    {"bpref_allnonrel", "Binary Preference, all judged nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_allnonrel)},
    {"bpref_retnonrel", "Binary Preference, all retrieved judged nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_retnonrel)},
    {"bpref_topnonrel", "Binary Preference, top 100 judged nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_topnonrel)},
    {"bpref_top5Rnonrel", "Binary Preference, top 5R judged nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_top5Rnonrel)},
    {"bpref_top10Rnonrel", "Binary Preference, top 10R judged nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_top10Rnonrel)},
    {"bpref_top10pRnonrel", "Binary Preference, top 10 + R judged nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_top10pRnonrel)},
    {"bpref_top25pRnonrel", "Binary Preference, top 25 + R judged nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_top25pRnonrel)},
    {"bpref_top50pRnonrel", "Binary Preference, top 50 + R judged nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_top50pRnonrel)},
    {"bpref_top25p2Rnonrel", "Binary Preference, top 25 + 2*R judged nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_top25p2Rnonrel)},
    {"bpref_retall", "Binary Preference, Only retrieved judged rel and nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_retall)},
    {"bpref_5", "Binary Preference, top 5 rel, top 5 nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_5)},
    {"bpref_10", "Binary Preference, top 10 rel, top 10 nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_10)},
    {"bpref_num_all", "Binary Preference, Number not retrieved before (all judged)",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_num_all)},
    {"bpref_num_ret", "Binary Preference, Number retrieved after",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, bpref_num_ret)},
    {"bpref_num_correct", "Binary Preference, Number correct preferences",
     1, 0, 0, 0, 0, 0, 0, 0, offsetof(TREC_EVAL, bpref_num_correct)},
    {"bpref_num_possible", "Binary Preference, Number possible correct_preferences",
     1, 0, 0, 0, 0, 0, 0, 0, offsetof(TREC_EVAL, bpref_num_possible)},
    {"old_bpref", "Buggy Version 7.3. Binary Preference, top R judged nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, old_bpref)},
    {"old_bpref_top10pRnonrel", "Buggy Version 7.3. Binary Preference,top 10+R judged nonrel",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, old_bpref_top10pRnonrel)},
    {"infAP", "Inferred AP. Calculate AP using only a judged random sample of the pool, averaging in unpooled documents as nonrel.",
     0, 0, 0, 0, 0, 1, 0, 0, offsetof(TREC_EVAL, inf_ap)},
    {"gm_bpref", "Binary Preference, top R judged nonrel, Geometric Mean, q_score=log(MAX(bpref,.00001))",
     0, 0, 0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, gm_bpref)},
    {"rank_first_rel", "Rank of top relevant document (0 if none)",
     1, 0, 0, 1, 0, 0, 0, 0, offsetof(TREC_EVAL, rank_first_rel)},
};

int num_sing_meas = sizeof (sing_meas) / sizeof (sing_meas[0]);

PARAMETERIZED_MEASURE param_meas[] = {
    {"Interpolated Recall - Precision Averages",
     0, 1, 0, 0, 0, 1, offsetof(TREC_EVAL, int_recall_precis[0]), NUM_RP_PTS, 
     "ircl_prn.%s", " at %s recall",
     get_param_str_ircl_prn},
    {"Precision",
     0, 1, 0, 0, 0, 1, offsetof(TREC_EVAL, precis_cut[0]), NUM_CUTOFF,
     "P%s", " after %s docs retrieved",
     get_param_str_cutoff},
    /* end of short output measures (the major ones) */

    {"Recall",
     0, 0, 0, 0, 0, 1, offsetof(TREC_EVAL, recall_cut[0]), NUM_CUTOFF,
     "recall%s", " after %s docs retrieved",
     get_param_str_cutoff},
    {"R-based precision",
     0, 0, 0, 0, 0, 1, offsetof(TREC_EVAL, R_prec_cut[0]), NUM_PREC_PTS-1,
     "%sR-prec", "- precision after %s * R docs retrieved",
     get_param_str_Rcutoff},
    {"Relative precision",
     0, 0, 0, 0, 0, 1, offsetof(TREC_EVAL, rel_precis_cut[0]), NUM_CUTOFF,
     "relative_prec%s", " after %s docs retrieved",
     get_param_str_cutoff},
    {"Unranked Average Precision",
     0, 0, 0, 0, 0, 1, offsetof(TREC_EVAL, uap_cut[0]), NUM_CUTOFF,
     "unranked_avg_prec%s", " after %s docs retrieved",
     get_param_str_cutoff},
    {"Relative Unranked Average Precision",
     0, 0, 0, 0, 0, 1, offsetof(TREC_EVAL, rel_uap_cut[0]), NUM_CUTOFF,
     "relative_unranked_avg_prec%s", " after %s docs retrieved",
     get_param_str_cutoff},
    {"Utility (a,b,c,d)",
     0, 0, 0, 0, 0, 1, offsetof(TREC_EVAL, exact_utility), 1,
     "utility_%s", " Coefficients %s ",
     get_param_str_utility},
    {"Recall averaged at X nonrel docs",
     0, 0, 0, 0, 0, 1, offsetof(TREC_EVAL, av_fall_recall), 1,
     "rcl_at_%s_nonrel", "    X= %s ",
     get_param_str_maxfallout},
    {"Fallout - Recall Averages",
     0, 0, 0, 0, 0, 1, offsetof(TREC_EVAL, fall_recall[0]), NUM_FR_PTS,
     "fallout_recall_%s", "- recall after %s nonrel docs retrieved",
     get_param_str_fall_recall},
    {"Interpolated R-based precision,",
     0, 0, 0, 0, 0, 1, offsetof(TREC_EVAL, int_R_prec_cut[0]), NUM_PREC_PTS-1,
     "int_%sR-prec", " after %s * R docs retrieved",
     get_param_str_Rcutoff},
    {"Time: Utility (a,b,c,d):",
     0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, av_time_utility), 1,
     "time_integral_utility_%s", " Coefficients %s ",
     get_param_str_utility},
    {"Time: num_rel at cutoff:",
     0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, time_num_rel[0]), NUM_TIME_PTS,
     "time_num_rel_%s", "    after %s seconds",
     get_param_str_time_cutoff},
    {"Time: num_nonrel at cutoff:",
     0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, time_num_nrel[0]), NUM_TIME_PTS,
     "time_num_nonrel_%s", "    after %s seconds",
     get_param_str_time_cutoff},
    {"Time: cumulative rel at cutoff:",
     0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, time_cum_rel[0]), NUM_TIME_PTS,
     "time_cum_rel_%s", "    after %s seconds",
     get_param_str_time_cutoff},
    {"Time: precision at time cutoff:",
     0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, time_precis[0]), NUM_TIME_PTS,
     "time_precis_%s", "    after %s seconds",
     get_param_str_time_cutoff},
    {"Time: precision at time cutoff:",
     0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, time_precis[0]), NUM_TIME_PTS,
     "time_precis_%s", "    after %s seconds",
     get_param_str_time_cutoff},
    {"Time: relative precision at time cutoff:",
     0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, time_relprecis[0]), NUM_TIME_PTS,
     "time_relative_precis_%s", "    after %s seconds",
     get_param_str_time_cutoff},
    {"Time: unranked precision at time cutoff:",
     0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, time_uap[0]), NUM_TIME_PTS,
     "time_uap_%s", "    after %s seconds",
     get_param_str_time_cutoff},
    {"Time: relative unranked precision at time cutoff:",
     0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, time_reluap[0]), NUM_TIME_PTS,
     "time_relative_uap_%s", "    after %s seconds",
     get_param_str_time_cutoff},
    {"Time: utility at time cutoff:",
     0, 0, 1, 0, 0, 1, offsetof(TREC_EVAL, time_utility[0]), NUM_TIME_PTS,
     "time_utility_%s", "    after %s seconds",
     get_param_str_time_utility_cutoff},
};

int num_param_meas = sizeof (param_meas) / sizeof (param_meas[0]);

MICRO_MEASURE micro_meas[] = {
    {"micro_prec", "Total relevant retrieved documents / Total retrieved documents",
     0, offsetof(TREC_EVAL, num_rel_ret), offsetof(TREC_EVAL, num_ret)},
    {"micro_recall", "Total relevant retrieved documents / Total relevant documents",
     0, offsetof(TREC_EVAL, num_rel_ret), offsetof(TREC_EVAL, num_rel)},
    {"micro_bpref", "Total correct preferences / Total possible preferences",
     0, offsetof(TREC_EVAL, bpref_num_correct), offsetof(TREC_EVAL, bpref_num_possible)},
};

int num_micro_meas = sizeof (micro_meas) / sizeof (micro_meas[0]);

