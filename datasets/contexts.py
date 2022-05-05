from typing import List

class ContextAllocateData:
    values: List[float]
    allocations: List[float]

    def __init__(self, values: List, allocations: List) -> None:
        if len(values) == 0:
            print("no context values!")
            return None

        if len(values) != len(allocations):
            print("can't allocate context values!")
            return None
        
        if sum(allocations) != 1:
            print("allocation invalid!")
            return None
        
        self.values = values
        self.allocations = allocations
