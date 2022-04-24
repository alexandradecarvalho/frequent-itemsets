from pyspark import SparkContext
from itertools import combinations

sc = SparkContext(appName="Assignment1")

data = sc.textFile('dataset/conditions.csv')
header = data.first()
data = data.filter(lambda line: line != header).map(lambda line: line.split(","))

cond_names_dict = data.map(lambda line: (line[-2], line[-1])).distinct().collectAsMap()

data = data.map(lambda line: (line[2], {line[-2]})).reduceByKey(lambda code1, code2: code1 | code2)

total = data.count()
def get_probability(code):
    if type(code) is str:
        return [code_support[1] for code_support in individual_counts if code_support[0] == code][0]/total
    elif type(code) is tuple and len(code) == 2:
        return [code_support[1] for code_support in pair_counts if code_support[0] == code][0]/total

def get_std_coef(i, j):
    return max(get_probability(i)+get_probability(j)-1, 1/total)/(get_probability(i)*get_probability(j))


max_k = 3
support_threshold = 1000

for k in range(1,max_k + 1):  
    
    if k == 1:
        counts = data.flatMap(lambda patient_condition: patient_condition[1]).map(lambda condition: (condition,1)).reduceByKey(lambda count1, count2: count1 + count2) 

        frequent_items = counts.filter(lambda condition_count: condition_count[1]  >= support_threshold)
        f_items = frequent_items.map(lambda condition_count: condition_count[0]).collect()
        frequent_data = data.map(lambda patient_conditions: (patient_conditions[0], {code for code in patient_conditions[1] if code in f_items}))

    else:
        candidate_combinations = frequent_data.map(lambda patient_conditions: (patient_conditions[0], {comb for comb in combinations(patient_conditions[1],k)}))
        frequents = candidate_combinations.flatMap(lambda patient_combinationset: patient_combinationset[1]).map(lambda combination: (combination,1)).reduceByKey(lambda count1, count2: count1 + count2).filter(lambda combination_count: combination_count[1]  >= support_threshold)
        
        if k == 2:
            frequent_pairs = frequents
        elif k == 3:
            frequent_trios = frequents

        print(frequents.sortBy(lambda triplet: triplet[1], ascending=False).map(lambda triplet: triplet[0]).take(10))

    k += 1

pair_counts = frequent_pairs.collect()
individual_counts = frequent_items.collect()

pair_subsets = frequent_trios.flatMap(lambda trio_support: [(comb,set(trio_support[0]) - set(comb),trio_support[1]) for comb in combinations(trio_support[0],2)])
subsets = pair_subsets.flatMap(lambda pair_j_tsupport: [pair_j_tsupport, ((pair_j_tsupport[0][0],), {pair_j_tsupport[0][1]}, [support[1] for support in pair_counts if support[0] == pair_j_tsupport[0]][0]), ((pair_j_tsupport[0][1],), {pair_j_tsupport[0][0]}, [support[1] for support in pair_counts if support[0] == pair_j_tsupport[0]][0])])

confidence = subsets.map(lambda pair_set_count : (pair_set_count[0], tuple(pair_set_count[1]), pair_set_count[2] / [pair_support[1] for pair_support in pair_counts if pair_support[0] == pair_set_count[0]][0]) if len(pair_set_count[0]) == 2 else (pair_set_count[0], tuple(pair_set_count[1]), pair_set_count[2] / [pair_support[1] for pair_support in individual_counts if pair_support[0] == pair_set_count[0][0]][0])).distinct()
interest = confidence.map(lambda i_j_confidence: i_j_confidence + (i_j_confidence[2] - get_probability(i_j_confidence[1][0]),))
lift = interest.map(lambda i_j_conf_int: i_j_conf_int + (i_j_conf_int[2] / get_probability(i_j_conf_int[1][0]),))
std_lift = lift.map(lambda ijcil: ijcil + ((ijcil[4] - get_std_coef(ijcil[0], ijcil[1][0])) / ((1/max(get_probability(ijcil[0]),get_probability(ijcil[1][0]))) - get_std_coef(ijcil[0], ijcil[1][0])),) if len(ijcil[0]) == 2 else ijcil + ((ijcil[4] - get_std_coef(ijcil[0][0], ijcil[1][0])) / ((1/max(get_probability(ijcil[0][0]),get_probability(ijcil[1][0]))) - get_std_coef(ijcil[0][0], ijcil[1][0])),))

conf_int_lift_stdlift = std_lift.map(lambda name_metrics: (cond_names_dict[name_metrics[0][0]] + ' -> ' + cond_names_dict[name_metrics[1][0]],) + name_metrics[2:] if len(name_metrics[0]) == 1 else (cond_names_dict[name_metrics[0][0]] + ' and ' + cond_names_dict[name_metrics[0][1]] + ' -> ' + cond_names_dict[name_metrics[1][0]],) + name_metrics[2:])
conf_int_lift_stdlift.filter(lambda metrics: metrics[-1] >= 0.2).sortBy(lambda metrics: metrics[-1]).saveAsTextFile('results/')