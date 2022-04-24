from pyspark import SparkContext
from itertools import combinations

sc = SparkContext(appName="Assignment1")
data = sc.textFile('dataset/conditions.csv')

header = data.first()
data = data.filter(lambda line: line != header).map(lambda line: line.split(","))

cond_names_dict = data.map(lambda line: (line[-2], line[-1])).distinct().collectAsMap()
data = data.map(lambda line: (line[2], {line[-2]})).reduceByKey(lambda code1, code2: code1 | code2)

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

pair_subsets = frequent_trios.flatMap(lambda trio_support: [(comb,list(set(trio_support[0]) - set(comb))[0],trio_support[1]) for comb in combinations(trio_support[0],2)])
single_subsets = frequent_pairs.flatMap(lambda pair_support: [((pair_support[0][0],),pair_support[0][1],pair_support[1]), (((pair_support[0][1],),pair_support[0][0],pair_support[1]))])
subsets = single_subsets.union(pair_subsets)

pair_counts = frequent_pairs.collect()
individual_counts = frequent_items.collect()

total = data.count()
def get_probability(code):
    if type(code) is tuple:
        if len(code) == 2:
            return [code_support[1] for code_support in pair_counts if code_support[0] == code][0]/total
        else:
            return [code_support[1] for code_support in individual_counts if code_support[0] == code[0]][0]/total
    else:
        return [code_support[1] for code_support in individual_counts if code_support[0] == code][0]/total

metrics = subsets.map(lambda pair_tup_count : (pair_tup_count[0], pair_tup_count[1], pair_tup_count[2] / [pair_support[1] for pair_support in pair_counts if pair_support[0] == pair_tup_count[0]][0]) if len(pair_tup_count[0]) == 2 else (pair_tup_count[0], pair_tup_count[1], pair_tup_count[2] / [pair_support[1] for pair_support in individual_counts if pair_support[0] == pair_tup_count[0][0]][0])).distinct().filter(lambda x : len(x[0]) == 2).map(lambda i_j_confidence: i_j_confidence + (i_j_confidence[2] - get_probability(i_j_confidence[1]),)).map(lambda i_j_conf_int: i_j_conf_int + (i_j_conf_int[2] / get_probability(i_j_conf_int[1]),))

def get_std_coef(i, j):
    return max(get_probability(i) + get_probability(j) - 1, 1/total)/(get_probability(i) * get_probability(j))

metrics = metrics.map(lambda ijcil :  ijcil + ( ((ijcil[4] - get_std_coef(ijcil[0], ijcil[1]))) / ((1/max(get_probability(ijcil[0]), get_probability(ijcil[1]))) - get_std_coef(ijcil[0], ijcil[1])) ,))
final = metrics.filter(lambda metrics: metrics[-1] >= 0.2)

f = final.collect()
f = [((x[0],x[1]),x[-1]) for x in f]

final = final.filter(lambda metrics: not any([1 if metrics[1] == x[0][0] and metrics[0] == x[0][1] and metrics[-1] < x[1] else 0 for x in f])).map(lambda name_metrics: (cond_names_dict[name_metrics[0][0]] + ' -> ' + cond_names_dict[name_metrics[1]],) + name_metrics[2:] if len(name_metrics[0]) == 1 else (cond_names_dict[name_metrics[0][0]] + ' and ' + cond_names_dict[name_metrics[0][1]] + ' -> ' + cond_names_dict[name_metrics[1]],) + name_metrics[2:])
final.sortBy(lambda metrics: metrics[-1]).saveAsTextFile('results/')