import re

# Expanded English stopword list (can be further customized)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "of", "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "over", "under",
    "and", "but", "or", "nor", "so", "yet",
    "very", "just", "only", "also", "even", "still", "such", "no", "not",
    "too", "than", "then", "once", "here", "there", "when", "where", "why", "how",
    "this", "that", "these", "those", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "own", "same",
}

class FRED_Text_to_XMP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING",),
                "sentence_mode": ("BOOLEAN", {"default": False}),
                "replace_space_with_underscore": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("XMP_formatted_text",)
    CATEGORY = "FRED/Utils"
    FUNCTION = "convert_to_xmp"

    def convert_to_xmp(self, text: str, sentence_mode: bool, replace_space_with_underscore: bool):
        tags = self.extract_tags(text, sentence_mode, replace_space_with_underscore)
        xmp = self.tags_to_xmp(tags)
        return (xmp,)

    def extract_tags(self, text, sentence_mode, replace_space):
        if sentence_mode:
            # Remove punctuation, split into words, filter stopwords
            words = re.findall(r'\b\w+\b', text.lower())
            tags = [w for w in words if w not in STOPWORDS]
        else:
            # Split by comma, strip spaces
            tags = [t.strip() for t in text.split(",") if t.strip()]
        if replace_space:
            tags = [t.replace(" ", "_") for t in tags]
        return tags

    def tags_to_xmp(self, tags):
        li_elements = "\n".join(f"          <rdf:li>{tag}</rdf:li>" for tag in tags)
        return f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description xmlns:dc='http://purl.org/dc/elements/1.1/'>
      <dc:subject>
        <rdf:Bag>
{li_elements}
        </rdf:Bag>
      </dc:subject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FRED_Text_to_XMP": FRED_Text_to_XMP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_Text_to_XMP": "👑 FRED_Text_to_XMP"
}