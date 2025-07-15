import re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu


def read_situations(fname):
    situations = []
    with open(fname, 'r', encoding='utf-8') as fin:
        situation = ""
        for l in fin:
            l = l.rstrip('\n')
            if 'script' in l and 'type=' in l:
                situation = ""
            elif l.startswith('</'):
                situations.append(situation)
            elif l.startswith('<u '):
                situation = l #Get last utterance
    return situations


def expand_matrix_basis(m, old_basis, new_basis):
    """
    Rewrite a matrix in a higher-dimensional basis, with potentially a different column order.
    Input:
    - m (numpy array): a matrix
    - old_basis (list of strings): a list of properties
    - new_basis (list of strings): a list of properties, longer than old_basis
    Returns:
    - new_m (np array): a new binary matrix
    """
    
    new_m = np.zeros((m.shape[0], len(new_basis)))
    old_in_new = [new_basis.index(prop) for prop in old_basis if prop in new_basis]
    new_m[:, old_in_new] = m
    return new_m


def retract_matrix_basis(m, old_basis, new_basis):
    """
    Rewrite a matrix in a lower-dimensional basis, with potentially a different column order.
    Input:
    - m (numpy array): a matrix
    - old_basis (list of strings): a list of properties
    - new_basis (list of strings): a list of properties, shorter than old_basis
    Returns:
    - new_m (np array): a new binary matrix
    """
    new_m = np.zeros((m.shape[0], len(new_basis)))
    new_in_old = [old_basis.index(prop) for prop in new_basis if prop in old_basis]
    new_m = m[:, new_in_old]
    return new_m
    

def row_by_row_similarity(x,y):
    einsum = np.einsum('ij,ij->i', x, y)
    l2_prod = (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    similarities = einsum / l2_prod
    similarities = np.nan_to_num(similarities, copy=True, nan=0.0, posinf=None, neginf=None)
    return similarities


def calculate_similarities(m1, m2, p1, p2, property_union=True):
    """Calculates cosine similarities between rows and columns."""
    p = []
    if property_union:
        p = list(set(p1+p2))
        m1 = expand_matrix_basis(m1, p1, p)
        m2 = expand_matrix_basis(m2, p2, p)
    else:
        p = list(set(p1) & set(p2))
        m1 = retract_matrix_basis(m1, p1, p)
        m2 = retract_matrix_basis(m2, p2, p)
    row_similarities = row_by_row_similarity(m1,m2)
    column_similarities = row_by_row_similarity(m1.T, m2.T)
    return row_similarities, column_similarities


def extract_entities_and_properties(logical_form):
      entities_properties = {}
      for lf in logical_form:
          entity = lf.split('.')[0]
          properties = lf.split()[1:]
          properties = [re.sub(r'\.\d+', '', prop) for prop in properties]
          properties.insert(0, entity)
          if entity not in entities_properties:
              entities_properties[entity] = set()
          for prop in properties:
              entities_properties[entity].add(prop)
      return entities_properties

def get_logical_forms(fname):
    logical_forms = [] # There will be one lf per situation
    with open(fname, 'r') as fin:
        situations = read_situations(fname)
    for sit in situations:
        sit = re.sub(r'<u speaker=[^>]*>','',sit)
        sit = re.sub(r'</u>','', sit)
        entities = [ent[1:-1] for ent in sit.split(', ')]
        logical_forms.append(entities)
    return logical_forms


def get_tokenized_utterances(fname):
    tokenized_utterances = []
    with open(fname, 'r') as fin:
        situations = read_situations(fname)
    for sit in situations:
        sit = re.sub(r'<u speaker=[^>]*>','',sit)
        sit = re.sub(r'</u>','', sit)
        tokenized_utterances.append(sit.lower().split()) #TODO REAL TOKENIZATION
    return tokenized_utterances


def mk_matrix(logical_form, verbose=False):
    """
    Creates a small entity-property matrix for a situation.
    Input:
    - logical_form (list): a list of lfs, one for each entity in the situation
    Returns:
    - m (np array): a binary matrix
    """
    
    entities_properties = extract_entities_and_properties(logical_form)

    # Create sorted list of unique properties and entities
    all_properties = sorted(set(prop for props in entities_properties.values() for prop in props))
    entities = list(entities_properties.keys())

    
    # Generate the activation matrix
    m = [[1 if prop in entities_properties[entity] else 0 for prop in all_properties] for entity in entities]
    
    if verbose:
        print("\nALL ENTS", entities)
        print("\nALL PROPS", all_properties)
        for i, row in enumerate(m):
            print(f"\nENT{i}", entities[i])
            print(row)
            for k,b in enumerate(row):
                if b == 1:
                    print(k, all_properties[k])

    m = np.array(m)
    return m, all_properties


def compute_bleu(reference, hypothesis):
    bleu_scores = []
    for i in range(1,4):
        weights = [1/i for _ in range(i)]
        bleu = sentence_bleu([reference], hypothesis, weights=weights)
        #print("REF", reference)
        #print("PRED", hypothesis)
        #print("BLEU",i, bleu)
        bleu_scores.append(bleu)
    return bleu_scores


if __name__ == "__main__":
    #fname_orig = "/home/aurelie/Projects/PossibleWorlds/seeds/vgnlp/botchen/data/testing/original/original_logic_to_logic.txt"
    #fname_pred = "/home/aurelie/Projects/PossibleWorlds/seeds/vgnlp/botchen/data/testing/predicted/predicted_logic_to_logic.txt"
    #fname_orig = "/home/aurelie/Projects/PossibleWorlds/seeds/vgnlp/botchen/data/testing/original/original_surface_to_logic.txt"
    #fname_pred = "/home/aurelie/Projects/PossibleWorlds/seeds/vgnlp/botchen/data/testing/predicted/predicted_surface_to_logic.txt"
    #fname_orig = "/home/aurelie/Projects/PossibleWorlds/seeds/vgnlp/botchen/data/testing/original/original_surface_to_surface.txt"
    #fname_pred = "/home/aurelie/Projects/PossibleWorlds/seeds/vgnlp/botchen/data/testing/predicted/predicted_surface_to_surface.txt"
    fname_orig = "/home/aurelie/Projects/PossibleWorlds/seeds/vgnlp/botchen/data/testing/original/original_logic_to_surface.txt"
    fname_pred = "/home/aurelie/Projects/PossibleWorlds/seeds/vgnlp/botchen/data/testing/predicted/predicted_logic_to_surface.txt"
  
    if fname_orig.endswith("to_logic.txt"):
        # One logical form per situation 
        logical_forms_orig = get_logical_forms(fname_orig)
        logical_forms_pred = get_logical_forms(fname_pred)

        situation_sims = []
        for i, lf in enumerate(logical_forms_orig):
            m_orig, props_orig = mk_matrix(logical_forms_orig[i])
            m_pred, props_pred = mk_matrix(logical_forms_pred[i])

            row_sims, col_sims = calculate_similarities(m_orig, m_pred, props_orig, props_pred, property_union=True)
            situation_sims.append(row_sims.mean())
        print(f"    Average Row Similarity: {np.array(situation_sims).mean():.3f}")

    else:
        utterances_orig = get_tokenized_utterances(fname_orig)
        utterances_pred = get_tokenized_utterances(fname_pred)

        bleu_scores = []
        for i,u in enumerate(utterances_orig):
            bleu_scores.append(compute_bleu(utterances_orig[i], utterances_pred[i]))

        bleus = [b[2] for b in bleu_scores] #trigrams
        print(f"    Average BLEU score (trigrams): {np.array(bleus).mean():.3f}")

