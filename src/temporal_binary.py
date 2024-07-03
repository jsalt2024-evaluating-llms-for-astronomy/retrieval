import re
import json

def is_temporal_query(query):
    pattern = re.compile(r"""
        \d{4}(?:\s*-\s*\d{4})?|  # Years or year ranges
        \d{1,2}(?:st|nd|rd|th)\s+century|  # Century references
        (?:last|past|next|coming|future)\s+\d+\s+(?:year|decade|century)|  # Relative time references
        (?:recent|latest|current|ongoing|future|past|historical|ancient)|  # Time-related adjectives
        (?:evolution|changes?|advancements?|developments?|progress)|  # Process words
        (?:solar\s+eclipse|equinox|solstice)|  # Astronomical events
        (?:mission|probe|telescope|spacecraft)|  # Space missions
        (?:nobel\s+prize|discovery|detection)|  # Scientific milestones
        (?:before\s+and\s+after|compared\s+with|vs\.?)|  # Comparative phrases
        (?:since|until|prior\s+to|post-|pre-)|  # Temporal prepositions
        (?:AD|BC)|  # Historical era indicators
        (?:release|launch)  # Events related to data or missions
    """, re.VERBOSE | re.IGNORECASE)
    
    return bool(pattern.search(query))

def load_queries(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_accuracy(queries):
    correct = 0
    total = len(queries)
    
    for query in queries:
        predicted = is_temporal_query(query['query_text'])
        actual = query['has_temporal_aspect']
        if predicted == actual:
            correct += 1
    
    accuracy = correct / total
    return accuracy

# Load queries
queries = load_queries('../data/temporal.json')

# Calculate accuracy
accuracy = calculate_accuracy(queries)

# Print accuracy
print(f"Accuracy: {accuracy:.2%}")