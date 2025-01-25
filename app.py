import streamlit as st
import hashlib
import math
from bitarray import bitarray


# Bloom Filter Class
class BloomFilter:
    def __init__(self, bit_array_size, false_positive_probability, num_elements):
        self.size = bit_array_size
        self.probability = false_positive_probability
        self.num_elements = num_elements
        self.num_hashes = 1
        if num_elements != 0:
            self.num_hashes = int((bit_array_size / num_elements) * (math.log10(2)))
        self.bit_array = bitarray('0' * int(bit_array_size))

    def hash_sha256(self, item):
        encoded = hashlib.sha256(item.encode())
        return int(encoded.hexdigest(), base=16) % self.size

    def insert(self, item):
        for count in range(self.num_hashes):
            index = int(self.hash_sha256(item + str(count)) % self.size)
            self.bit_array[index] = 1

    def lookup(self, item):
        for count in range(self.num_hashes):
            check = int(self.hash_sha256(item + str(count)) % self.size)
            if self.bit_array[check] == 0:
                return False
        return True


# Streamlit Interface
st.title("Text Search Using Bloom Filters")

# Step 1: Upload a Document
uploaded_file = st.file_uploader("Upload a document (text file)", type=["txt"])

if uploaded_file is not None:
    # Load and process document
    document_data = uploaded_file.read().decode('utf-8').split()
    num_loaded_elements = len(document_data)
    desired_false_positive_probability = 0.01
    bit_array_size = int(-(num_loaded_elements * math.log10(desired_false_positive_probability)) / (math.log10(2) ** 2))
    bloom_filter = BloomFilter(bit_array_size, desired_false_positive_probability, num_loaded_elements)

    # Insert words into the Bloom filter
    for word in document_data:
        bloom_filter.insert(word)

    st.success("Document successfully loaded and processed.")

    # Step 2: Search Bar
    search_query = st.text_input("Enter a word to search:")

    if search_query:
        # Search for the word
        if bloom_filter.lookup(search_query):
            st.write(f"✅ **'{search_query}'** is in the document (possibly a true positive).")
        else:
            st.write(f"❌ **'{search_query}'** is NOT in the document (false positive or true negative).")

    # Step 3: Display Bloom Filter Stats
    with st.expander("Bloom Filter Statistics"):
        st.write(f"Number of elements added (n): {bloom_filter.num_elements}")
        st.write(f"Size of bit array (m): {bloom_filter.size}")
        st.write(f"Probability of false positives (p): {bloom_filter.probability}")
        st.write(f"Number of hash functions used (k): {bloom_filter.num_hashes}")
