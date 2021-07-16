

class LRUCache:
    def __init__(self, max_elems):
        self.max_elems = max_elems
        self.cache = []

    
    def add(self, action):
        if len(self.cache) == self.max_elems:
            self.cache.pop(0)
        self.cache.append(action)
        

    def cache_equal(self, action):
        result = self.cache.count(action) >= self.max_elems/2
        return result
