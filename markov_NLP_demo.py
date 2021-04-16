import numpy as np
import re
import copy
# Which a_i in b
def which_inds(a,b):
    n = len(a)
    res = [int()] * n
    for i in range(n):
        res[i] = np.where(a[i] == b)[0]

    try:
        res_out = np.concatenate(res)
    except ValueError:
        print("Last word is unique and has no next")

    return res_out

# Which b_i in a
def which_ind(a, b):
    n = len(b)
    res = ["NA"] * n
    for i in range(n):
        if a == b[i]:
            res[i] = True
        else:
            res[i] = False

    return np.where(res)[0]


# Frequency table
def table_frq(x,freq):

    array_x = np.array(x)
    unique_x = np.array(sorted(list(set(x))))
    n = len(unique_x)
    res = [float()] * n
    for i in range(n):

        res[i] = len(np.where(array_x == unique_x[i])[0])

    if freq is False:
        return res
    else:
        return np.divide(res,sum(res))

# Estimate transition matrix P
def estP(s):

    string_no_punctuation = re.sub(r'[^a-zA-Z0-9]+', ' ', s)
    string_no_punctuation = re.sub(' +', ' ', string_no_punctuation).lower()
    string_as_list = string_no_punctuation.split()
    string_as_list_next = copy.deepcopy(string_as_list)
    string_as_list_next.append(string_as_list[0])
    string_unique_values_sorted = sorted(set(string_as_list))
    string_as_array = np.asarray(string_as_list)
    string_as_array_next = np.asarray(string_as_list_next)

    n = len(string_unique_values_sorted)
    x_next_words = ['NA'] * n
    x_next_frequency = [float()] * n
    x_next_indices_in_unique = [int()] * n
    transition_matrix = np.empty([n, n], dtype=float)
    for i in range(n):
        x_next_words[i] = string_as_array_next[which_ind(string_unique_values_sorted[i], string_as_array)+1]
        x_next_frequency[i] = table_frq(x_next_words[i], True)
        x_next_words[i] = list(sorted(set(x_next_words[i])))
        x_next_indices_in_unique[i] = which_inds(x_next_words[i], np.asarray(string_unique_values_sorted))
        p_word = np.array([float(0)] * n)
        p_word[x_next_indices_in_unique[i]] = x_next_frequency[i]
        transition_matrix[i,] = p_word

    return list([transition_matrix, string_unique_values_sorted])

# Predict P
def predict_P(string_of_text, number_or_words, seed_word):


    P_and_names = estP(string_of_text)
    n = len(P_and_names[1])
    ind_of_one = which_ind(seed_word,P_and_names[1])
    pi_start = np.array([float(0)] * n)
    pi_start[ind_of_one] = 1

    y_words = [str()] * number_or_words
    for i in range(number_or_words):

        y_words[i] = np.random.choice(P_and_names[1], p = pi_start @ P_and_names[0])
        new_ind = which_ind(y_words[i], P_and_names[1])
        pi_start = np.array([float(0)] * n)
        pi_start[new_ind] = 1

    result = [seed_word] + y_words
    result = ' '.join(result)

    return result

# Test
example_string = "this is an example sentence. you could try any sentence you like. just give it a a string and you are fine. moreover, you should introduce a seed word which must be included in the string, and number of words, as an integer, the markov should predict."
test = predict_P(example_string, 3, 'you')
print(test)

