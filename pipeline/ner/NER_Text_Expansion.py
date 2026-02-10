def expand_text_spans(tokens, predicted_labels, tokenizer, window_size=0):
    
    ranges = []
    expanded_text_spans = []
    ner_text_spans = []
    ner_labels = []

    
    results = []
    i = 0
    
    while i < len(predicted_labels):
        if predicted_labels[i].startswith("B-"):
            # Found a B, now find the end of consecutive B's and following I's
            pattern_start = i
            pattern_end = pattern_start
            
            j = i + 1


            I_found = False
            # Then, consume any following I's
            while j < len(predicted_labels):
                # Consume following B's
                if not I_found and predicted_labels[j].startswith("B-"):
                    pattern_end = j
                    j += 1
                elif predicted_labels[j].startswith("I-"):
                    pattern_end = j
                    j += 1
                    I_found = True
                else:
                    break
            
           
            expanded_tokens_span = tokens[pattern_start:pattern_end+1]
            expanded_text_spans.append(tokenizer.convert_tokens_to_string(expanded_tokens_span))

         
            # Move past this pattern
            i = pattern_end + 1

            
        elif predicted_labels[i].startswith("I-"):
            # Found standalone I's (not preceded by B)
            pattern_start = i
            pattern_end = i
            
            # Consume consecutive I's
            j = i + 1
            while j < len(predicted_labels): 
                if predicted_labels[j].startswith("I-"):
                    pattern_end = j
                    j += 1
                    
                else:
                    break
            
           
            expanded_tokens_span = tokens[pattern_start:pattern_end+1]
            expanded_text_spans.append(tokenizer.convert_tokens_to_string(expanded_tokens_span))

            
            # Move past this pattern
            i = pattern_end + 1
        
 
            
        else:
            i += 1
            
    return ner_text_spans, expanded_text_spans
